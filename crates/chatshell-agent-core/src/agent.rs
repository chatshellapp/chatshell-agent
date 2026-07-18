use futures::StreamExt;
use rig::OneOrMany;
use rig::agent::{Agent, AgentBuilder, MultiTurnStreamItem};
use rig::client::{CompletionClient, Nothing};
use rig::completion::{CompletionModel, Message};
use rig::message::{AssistantContent, Reasoning, UserContent};
use rig::providers::{anthropic, moonshot, ollama, openrouter};
use rig::streaming::{StreamedAssistantContent, StreamingChat};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::thinking::parse_thinking_content;
use crate::types::{
    AgentError, AgentEvent, AgentRequest, ChatMessage, MessageRole, ProviderKind,
    SamplingParameters,
};

type OpenAiCompatibleModel = moonshot::CompletionModel;
type AnthropicModel = anthropic::completion::CompletionModel;
type OpenRouterModel = openrouter::CompletionModel;
type OllamaModel = ollama::CompletionModel;

enum ProviderAgent {
    OpenAiCompatible(Agent<OpenAiCompatibleModel>),
    Anthropic(Agent<AnthropicModel>),
    OpenRouter(Agent<OpenRouterModel>),
    Ollama(Agent<OllamaModel>),
}

pub async fn run_generation(
    request: AgentRequest,
    sender: mpsc::Sender<Result<AgentEvent, AgentError>>,
    cancellation: CancellationToken,
) -> Result<(), AgentError> {
    let (prompt, history, system_prompt) = build_messages(&request.messages)?;
    let agent = create_agent(&request, system_prompt.as_deref())?;

    let result = match agent {
        ProviderAgent::OpenAiCompatible(agent) => {
            stream_agent(agent, prompt, history, &sender, &cancellation).await
        }
        ProviderAgent::Anthropic(agent) => {
            stream_agent(agent, prompt, history, &sender, &cancellation).await
        }
        ProviderAgent::OpenRouter(agent) => {
            stream_agent(agent, prompt, history, &sender, &cancellation).await
        }
        ProviderAgent::Ollama(agent) => {
            stream_agent(agent, prompt, history, &sender, &cancellation).await
        }
    };

    if let Err(error) = &result {
        let _ = sender.send(Err(error.clone())).await;
    }
    result
}

fn create_agent(
    request: &AgentRequest,
    system_prompt: Option<&str>,
) -> Result<ProviderAgent, AgentError> {
    let http_client = create_http_client();
    let model = request.model.as_str();
    let parameters = &request.parameters;

    match request.provider.kind {
        ProviderKind::OpenAi | ProviderKind::CustomOpenAi => {
            let key = required_key(request, "OpenAI-compatible")?;
            let base_url = request
                .provider
                .base_url
                .as_deref()
                .unwrap_or("https://api.openai.com/v1");
            let client = moonshot::Client::<reqwest::Client>::builder()
                .api_key(key)
                .base_url(base_url)
                .http_client(http_client)
                .build()
                .map_err(provider_error)?;
            Ok(ProviderAgent::OpenAiCompatible(build_agent(
                client.agent(model),
                system_prompt,
                parameters,
            )?))
        }
        ProviderKind::Anthropic => {
            let key = required_key(request, "Anthropic")?;
            let base_url = request
                .provider
                .base_url
                .as_deref()
                .unwrap_or("https://api.anthropic.com/v1");
            let client = anthropic::Client::<reqwest::Client>::builder()
                .api_key(key)
                .base_url(base_url)
                .http_client(http_client)
                .build()
                .map_err(provider_error)?;
            Ok(ProviderAgent::Anthropic(build_agent(
                AgentBuilder::new(client.completion_model(model).with_prompt_caching()),
                system_prompt,
                parameters,
            )?))
        }
        ProviderKind::OpenRouter => {
            let key = required_key(request, "OpenRouter")?;
            let base_url = request
                .provider
                .base_url
                .as_deref()
                .unwrap_or("https://openrouter.ai/api/v1");
            let client = openrouter::Client::<reqwest::Client>::builder()
                .api_key(key)
                .base_url(base_url)
                .http_client(http_client)
                .build()
                .map_err(provider_error)?;
            Ok(ProviderAgent::OpenRouter(build_agent(
                client.agent(model),
                system_prompt,
                parameters,
            )?))
        }
        ProviderKind::Ollama => {
            let base_url = request
                .provider
                .base_url
                .as_deref()
                .unwrap_or("http://127.0.0.1:11434");
            let client = ollama::Client::<reqwest::Client>::builder()
                .api_key(Nothing)
                .base_url(base_url)
                .http_client(http_client)
                .build()
                .map_err(provider_error)?;
            Ok(ProviderAgent::Ollama(build_agent(
                client.agent(model),
                system_prompt,
                parameters,
            )?))
        }
    }
}

fn create_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("ChatShell/0.1")
        .build()
        .unwrap_or_default()
}

fn required_key<'a>(
    request: &'a AgentRequest,
    provider_name: &'static str,
) -> Result<&'a str, AgentError> {
    request
        .provider
        .api_key
        .as_deref()
        .filter(|key| !key.is_empty())
        .ok_or(AgentError::MissingApiKey(provider_name))
}

fn build_agent<M: CompletionModel>(
    mut builder: AgentBuilder<M>,
    system_prompt: Option<&str>,
    parameters: &SamplingParameters,
) -> Result<Agent<M>, AgentError> {
    if let Some(prompt) = system_prompt {
        builder = builder.preamble(prompt);
    }
    if let Some(temperature) = parameters.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(max_tokens) = parameters.max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    if let Some(json) = parameters.additional_params_json.as_deref() {
        let value = serde_json::from_str(json)
            .map_err(|error| AgentError::InvalidProvider(error.to_string()))?;
        builder = builder.additional_params(value);
    }
    Ok(builder.build())
}

fn build_messages(
    messages: &[ChatMessage],
) -> Result<(Message, Vec<Message>, Option<String>), AgentError> {
    let system_prompt = messages
        .iter()
        .find(|message| message.role == MessageRole::System)
        .map(|message| message.content.clone());
    let last_user_index = messages
        .iter()
        .rposition(|message| message.role == MessageRole::User)
        .ok_or(AgentError::MissingUserMessage)?;

    let mut history = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role == MessageRole::System || index == last_user_index {
            continue;
        }
        history.push(to_rig_message(message));
    }

    Ok((
        to_rig_message(&messages[last_user_index]),
        history,
        system_prompt,
    ))
}

fn to_rig_message(message: &ChatMessage) -> Message {
    match message.role {
        MessageRole::User | MessageRole::System => Message::User {
            content: OneOrMany::one(UserContent::Text(message.content.clone().into())),
        },
        MessageRole::Assistant => {
            let mut content = Vec::new();
            if let Some(reasoning) = message.reasoning_content.as_deref() {
                if !reasoning.is_empty() {
                    content.push(AssistantContent::reasoning(reasoning));
                }
            }
            content.push(AssistantContent::Text(message.content.clone().into()));
            Message::Assistant {
                id: None,
                content: OneOrMany::many(content).expect("assistant content is never empty"),
            }
        }
    }
}

async fn stream_agent<M>(
    agent: Agent<M>,
    prompt: Message,
    history: Vec<Message>,
    sender: &mpsc::Sender<Result<AgentEvent, AgentError>>,
    cancellation: &CancellationToken,
) -> Result<(), AgentError>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: rig::completion::GetTokenUsage,
{
    let mut stream = agent.stream_chat(prompt, history).multi_turn(8).await;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    loop {
        let item = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                send_event(
                    sender,
                    AgentEvent::Completed {
                        content,
                        reasoning: (!reasoning.is_empty()).then_some(reasoning),
                        input_tokens,
                        output_tokens,
                        cancelled: true,
                    },
                ).await?;
                return Ok(());
            }
            item = stream.next() => match item {
                Some(item) => item,
                None => break,
            }
        };

        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text))) => {
                if !text.text.is_empty() {
                    content.push_str(&text.text);
                    send_event(sender, AgentEvent::TextDelta { text: text.text }).await?;
                }
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                Reasoning {
                    content: parts, ..
                },
            ))) => {
                let delta = parts
                    .iter()
                    .filter_map(|part| match part {
                        rig::message::ReasoningContent::Text { text, .. } => Some(text.as_str()),
                        rig::message::ReasoningContent::Summary(summary) => Some(summary.as_str()),
                        _ => None,
                    })
                    .collect::<String>();
                if !delta.is_empty() {
                    reasoning.push_str(&delta);
                    send_event(sender, AgentEvent::ReasoningDelta { text: delta }).await?;
                }
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta {
                    reasoning: delta, ..
                },
            )) => {
                if !delta.is_empty() {
                    reasoning.push_str(&delta);
                    send_event(sender, AgentEvent::ReasoningDelta { text: delta }).await?;
                }
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ToolCall { tool_call, .. },
            )) => {
                let arguments_json = serde_json::to_string(&tool_call.function.arguments)
                    .unwrap_or_else(|_| "{}".to_string());
                send_event(
                    sender,
                    AgentEvent::ToolCallStarted {
                        id: tool_call.id,
                        name: tool_call.function.name,
                        arguments_json,
                    },
                )
                .await?;
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Image(
                data_url,
            ))) => {
                send_event(sender, AgentEvent::Image { data_url }).await?;
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                let usage = response.usage();
                input_tokens = usage.input_tokens;
                output_tokens = usage.output_tokens;
            }
            Ok(_) => {}
            Err(error) => return Err(provider_error(error)),
        }
    }

    let parsed = parse_thinking_content(&content);
    if reasoning.is_empty() {
        reasoning = parsed.thinking_content.unwrap_or_default();
    }
    send_event(
        sender,
        AgentEvent::Completed {
            content: parsed.content,
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            input_tokens,
            output_tokens,
            cancelled: false,
        },
    )
    .await
}

async fn send_event(
    sender: &mpsc::Sender<Result<AgentEvent, AgentError>>,
    event: AgentEvent,
) -> Result<(), AgentError> {
    sender
        .send(Ok(event))
        .await
        .map_err(|_| AgentError::ConsumerClosed)
}

fn provider_error(error: impl std::fmt::Display) -> AgentError {
    AgentError::Provider(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProviderConfig;

    #[test]
    fn selects_last_user_message_as_prompt() {
        let messages = vec![
            ChatMessage {
                role: MessageRole::System,
                content: "Be concise".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: "First".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: MessageRole::Assistant,
                content: "Reply".into(),
                reasoning_content: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: "Second".into(),
                reasoning_content: None,
            },
        ];

        let (_, history, system) = build_messages(&messages).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(system.as_deref(), Some("Be concise"));
    }

    #[test]
    fn rejects_missing_api_key() {
        let request = AgentRequest {
            provider: ProviderConfig {
                kind: ProviderKind::OpenAi,
                api_key: None,
                base_url: None,
            },
            model: "test".into(),
            messages: vec![],
            parameters: SamplingParameters::default(),
        };
        assert_eq!(
            required_key(&request, "OpenAI-compatible"),
            Err(AgentError::MissingApiKey("OpenAI-compatible"))
        );
    }
}
