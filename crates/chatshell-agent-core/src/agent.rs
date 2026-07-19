use std::sync::Arc;
use futures::StreamExt;
use rig::OneOrMany;
use rig::agent::{Agent, AgentBuilder, MultiTurnStreamItem};
use rig::client::{CompletionClient, Nothing};
use rig::completion::{CompletionModel, GetTokenUsage, Message, ToolDefinition};
use rig::message::{
    AssistantContent, DocumentSourceKind, Image, ImageDetail, ImageMediaType, Reasoning,
    ToolCall, ToolFunction, UserContent,
};
use rig::providers::{anthropic, moonshot, ollama, openai, openrouter};
use rig::streaming::{StreamedAssistantContent, StreamingChat};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::thinking::parse_thinking_content;
use crate::types::{
    AgentError, AgentEvent, AgentRequest, ChatMessage, MessageRole, ProviderKind,
    SamplingParameters,
};

type OpenAiCompatibleModel = crate::openai_compat::CompletionModel;
type AnthropicModel = anthropic::completion::CompletionModel;
type OpenRouterModel = openrouter::CompletionModel;
type OllamaModel = ollama::CompletionModel;
type OpenAiResponsesModel = openai::responses_api::ResponsesCompletionModel;

/// Default max_tokens for Anthropic (required by the API, unlike OpenAI).
const ANTHROPIC_DEFAULT_MAX_TOKENS: u64 = 8192;

fn sampling_parameters_with_default_max_tokens(
    parameters: &SamplingParameters,
    default_max_tokens: u64,
) -> SamplingParameters {
    if parameters.max_tokens.is_some() {
        parameters.clone()
    } else {
        SamplingParameters {
            max_tokens: Some(default_max_tokens),
            ..parameters.clone()
        }
    }
}

enum ProviderAgent {
    OpenAiCompatible(Agent<OpenAiCompatibleModel>),
    OpenAi(Agent<OpenAiResponsesModel>),
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
        ProviderAgent::OpenAiCompatible(agent) => run_stream(
            agent, prompt, history, request.tools.clone(),
            system_prompt.as_deref(), &request.parameters, None, &sender, &cancellation,
        ).await,
        ProviderAgent::OpenAi(agent) => run_stream(
            agent, prompt, history, request.tools.clone(),
            system_prompt.as_deref(), &request.parameters, None, &sender, &cancellation,
        ).await,
        ProviderAgent::Anthropic(agent) => run_stream(
            agent, prompt, history, request.tools.clone(),
            system_prompt.as_deref(), &request.parameters, Some(ANTHROPIC_DEFAULT_MAX_TOKENS), &sender, &cancellation,
        ).await,
        ProviderAgent::OpenRouter(agent) => run_stream(
            agent, prompt, history, request.tools.clone(),
            system_prompt.as_deref(), &request.parameters, None, &sender, &cancellation,
        ).await,
        ProviderAgent::Ollama(agent) => run_stream(
            agent, prompt, history, request.tools.clone(),
            system_prompt.as_deref(), &request.parameters, None, &sender, &cancellation,
        ).await,
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
            if request.provider.api_style.as_deref() == Some("chat_completions") {
                // Chat Completions API - for OpenAI-compatible providers that
                // do not implement the Responses API (e.g. MiniMax, DeepSeek).
                let client = moonshot::Client::<reqwest::Client>::builder()
                    .api_key(key)
                    .base_url(base_url)
                    .http_client(http_client)
                    .build()
                    .map_err(provider_error)?;
                Ok(ProviderAgent::OpenAiCompatible(build_agent(
                    AgentBuilder::new(crate::openai_compat::CompletionModel::new(client, model)),
                    system_prompt,
                    parameters,
                )?))
            } else {
                // Responses API - default for OpenAI and custom_openai,
                // matching desktop's `create_openai_agent`.
                let client = openai::Client::<reqwest::Client>::builder()
                    .api_key(key)
                    .base_url(base_url)
                    .http_client(http_client)
                    .build()
                    .map_err(provider_error)?;
                Ok(ProviderAgent::OpenAi(build_agent(
                    client.agent(model),
                    system_prompt,
                    parameters,
                )?))
            }
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
            let anthropic_parameters =
                sampling_parameters_with_default_max_tokens(parameters, ANTHROPIC_DEFAULT_MAX_TOKENS);
            Ok(ProviderAgent::Anthropic(build_agent(
                AgentBuilder::new(client.completion_model(model).with_prompt_caching()),
                system_prompt,
                &anthropic_parameters,
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

    // The last non-system message is the prompt; everything else (except the
    // system message) becomes history. This supports both the normal path
    // (last message is the user's input) and the client-side tool path (last
    // message is a tool result).
    let last_prompt_index = messages
        .iter()
        .rposition(|message| message.role != MessageRole::System)
        .ok_or(AgentError::MissingUserMessage)?;

    let mut history = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role == MessageRole::System || index == last_prompt_index {
            continue;
        }
        history.push(to_rig_message(message));
    }

    Ok((
        to_rig_message(&messages[last_prompt_index]),
        history,
        system_prompt,
    ))
}

fn to_rig_message(message: &ChatMessage) -> Message {
    match message.role {
        MessageRole::User | MessageRole::System => {
            let mut contents: Vec<UserContent> = Vec::new();
            contents.push(UserContent::Text(message.content.clone().into()));
            for image in &message.images {
                if let Some(media_type) = mime_to_image_media_type(&image.media_type) {
                    contents.push(UserContent::Image(Image {
                        data: DocumentSourceKind::Base64(image.base64.clone()),
                        media_type: Some(media_type),
                        // OpenAI-compatible path (rig) requires `detail` to be
                        // `Some` for base64 images; MiniMax additionally
                        // rejects `"auto"` (error 2013) and accepts only
                        // `low` / `default` / `high`. `High` is the only
                        // value in rig's `ImageDetail` enum that satisfies both
                        // OpenAI and MiniMax, so we use it for all providers.
                        detail: Some(ImageDetail::High),
                        additional_params: None,
                    }));
                }
            }
            let content = if contents.len() == 1 {
                OneOrMany::one(contents.remove(0))
            } else {
                OneOrMany::many(contents).unwrap_or_else(|_| {
                    OneOrMany::one(UserContent::Text(message.content.clone().into()))
                })
            };
            Message::User { content }
        }
        MessageRole::Assistant => {
            let mut content = Vec::new();
            if let Some(reasoning) = message.reasoning_content.as_deref()
                && !reasoning.is_empty()
            {
                content.push(AssistantContent::reasoning(reasoning));
            }
            if !message.content.is_empty() {
                content.push(AssistantContent::Text(message.content.clone().into()));
            }
            for tool_call in &message.tool_calls {
                let arguments = serde_json::from_str(&tool_call.arguments_json)
                    .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                content.push(AssistantContent::ToolCall(ToolCall::new(
                    tool_call.id.clone(),
                    ToolFunction::new(tool_call.name.clone(), arguments),
                )));
            }
            if content.is_empty() {
                content.push(AssistantContent::Text(message.content.clone().into()));
            }
            Message::Assistant {
                id: None,
                content: OneOrMany::many(content).expect("assistant content is never empty"),
            }
        }
        MessageRole::Tool => {
            let id = message.tool_call_id.clone().unwrap_or_default();
            Message::tool_result(id, message.content.clone())
        }
    }
}

/// Convert a MIME string (e.g. `image/png`) to rig's `ImageMediaType`.
/// Returns `None` for unsupported types so unknown formats are skipped
/// rather than aborting the whole request.
fn mime_to_image_media_type(mime: &str) -> Option<ImageMediaType> {
    match mime.to_ascii_lowercase().as_str() {
        "image/jpeg" | "image/jpg" => Some(ImageMediaType::JPEG),
        "image/png" => Some(ImageMediaType::PNG),
        "image/gif" => Some(ImageMediaType::GIF),
        "image/webp" => Some(ImageMediaType::WEBP),
        _ => None,
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
                Reasoning { content: parts, .. },
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
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            })) => {
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

/// Route a generation to the normal multi-turn stream or the client-side tool
/// stream depending on whether tool definitions were provided.
async fn run_stream<M>(
    agent: Agent<M>,
    prompt: Message,
    history: Vec<Message>,
    tools: Vec<crate::types::ClientToolDefinition>,
    system_prompt: Option<&str>,
    parameters: &SamplingParameters,
    default_max_tokens: Option<u64>,
    sender: &mpsc::Sender<Result<AgentEvent, AgentError>>,
    cancellation: &CancellationToken,
) -> Result<(), AgentError>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
{
    let parameters = match default_max_tokens {
        Some(default) => sampling_parameters_with_default_max_tokens(parameters, default),
        None => parameters.clone(),
    };

    if tools.is_empty() {
        stream_agent(agent, prompt, history, sender, cancellation).await
    } else {
        stream_agent_with_client_tools(
            agent.model.clone(),
            prompt,
            history,
            tools,
            system_prompt,
            &parameters,
            sender,
            cancellation,
        )
        .await
    }
}

/// Stream a single completion turn with client-side tool definitions. Tool
/// calls are emitted as `ToolCallStarted` events and NOT executed server-side;
/// the caller must execute them and send a follow-up `AgentRequest` containing
/// the assistant tool-call message and a `Tool`-role result message. This
/// bypasses rig's multi-turn auto-execution so tool execution can happen in
/// the client (Swift), reusing the platform's native networking.
async fn stream_agent_with_client_tools<M>(
    model: Arc<M>,
    prompt: Message,
    history: Vec<Message>,
    tools: Vec<crate::types::ClientToolDefinition>,
    system_prompt: Option<&str>,
    parameters: &SamplingParameters,
    sender: &mpsc::Sender<Result<AgentEvent, AgentError>>,
    cancellation: &CancellationToken,
) -> Result<(), AgentError>
where
    M: CompletionModel + 'static,
    M::StreamingResponse: GetTokenUsage,
{
    let tool_defs: Vec<ToolDefinition> = tools
        .into_iter()
        .map(|t| {
            let parameters = serde_json::from_str(&t.parameters_json)
                .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
            ToolDefinition {
                name: t.name,
                description: t.description,
                parameters,
            }
        })
        .collect();

    let mut builder = model.as_ref().completion_request(prompt).messages(history);
    if let Some(p) = system_prompt {
        builder = builder.preamble(p.to_string());
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
    builder = builder.tools(tool_defs);

    let mut stream = builder.stream().await.map_err(provider_error)?;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;

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
                )
                .await?;
                return Ok(());
            }
            item = stream.next() => match item {
                Some(item) => item,
                None => break,
            }
        };

        match item {
            Ok(StreamedAssistantContent::Text(text)) => {
                if !text.text.is_empty() {
                    content.push_str(&text.text);
                    send_event(sender, AgentEvent::TextDelta { text: text.text }).await?;
                }
            }
            Ok(StreamedAssistantContent::Reasoning(Reasoning { content: parts, .. })) => {
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
            Ok(StreamedAssistantContent::ReasoningDelta { reasoning: delta, .. }) => {
                if !delta.is_empty() {
                    reasoning.push_str(&delta);
                    send_event(sender, AgentEvent::ReasoningDelta { text: delta }).await?;
                }
            }
            Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
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
                // Client-side execution: do NOT call the tool here. Continue
                // consuming the stream so trailing Final/usage is captured.
            }
            Ok(StreamedAssistantContent::Image(data_url)) => {
                send_event(sender, AgentEvent::Image { data_url }).await?;
            }
            Ok(StreamedAssistantContent::Final(final_resp)) => {
                if let Some(usage) = GetTokenUsage::token_usage(&final_resp) {
                    input_tokens = usage.input_tokens;
                    output_tokens = usage.output_tokens;
                }
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
                images: vec![],
                tool_calls: vec![],
                tool_call_id: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: "First".into(),
                reasoning_content: None,
                images: vec![],
                tool_calls: vec![],
                tool_call_id: None,
            },
            ChatMessage {
                role: MessageRole::Assistant,
                content: "Reply".into(),
                reasoning_content: None,
                images: vec![],
                tool_calls: vec![],
                tool_call_id: None,
            },
            ChatMessage {
                role: MessageRole::User,
                content: "Second".into(),
                reasoning_content: None,
                images: vec![],
                tool_calls: vec![],
                tool_call_id: None,
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
                api_style: None,
            },
            model: "test".into(),
            messages: vec![],
            parameters: SamplingParameters::default(),
            tools: vec![],
        };
        assert_eq!(
            required_key(&request, "OpenAI-compatible"),
            Err(AgentError::MissingApiKey("OpenAI-compatible"))
        );
    }

    #[tokio::test]
    async fn routes_openai_to_responses_by_default() {
        let request = AgentRequest {
            provider: ProviderConfig {
                kind: ProviderKind::OpenAi,
                api_key: Some("sk-test".into()),
                base_url: None,
                api_style: None,
            },
            model: "gpt-4o".into(),
            messages: vec![],
            parameters: SamplingParameters::default(),
            tools: vec![],
        };
        let agent = create_agent(&request, None).unwrap();
        assert!(matches!(agent, ProviderAgent::OpenAi(_)));
    }

    #[tokio::test]
    async fn routes_chat_completions_style_to_compat_model() {
        let request = AgentRequest {
            provider: ProviderConfig {
                kind: ProviderKind::CustomOpenAi,
                api_key: Some("sk-test".into()),
                base_url: Some("https://api.deepseek.com".into()),
                api_style: Some("chat_completions".into()),
            },
            model: "deepseek-chat".into(),
            messages: vec![],
            parameters: SamplingParameters::default(),
            tools: vec![],
        };
        let agent = create_agent(&request, None).unwrap();
        assert!(matches!(agent, ProviderAgent::OpenAiCompatible(_)));
    }

    #[test]
    fn converts_tool_result_message_to_user() {
        let message = ChatMessage {
            role: MessageRole::Tool,
            content: "result text".into(),
            reasoning_content: None,
            images: vec![],
            tool_calls: vec![],
            tool_call_id: Some("call_1".into()),
        };
        let rig_message = to_rig_message(&message);
        assert!(
            matches!(rig_message, Message::User { .. }),
            "tool result should map to a User message"
        );
    }

    #[test]
    fn converts_assistant_message_with_tool_calls() {
        let message = ChatMessage {
            role: MessageRole::Assistant,
            content: String::new(),
            reasoning_content: None,
            images: vec![],
            tool_calls: vec![crate::types::ChatToolCall {
                id: "call_1".into(),
                name: "web_search".into(),
                arguments_json: r#"{"query":"rust"}"#.into(),
            }],
            tool_call_id: None,
        };
        let rig_message = to_rig_message(&message);
        match rig_message {
            Message::Assistant { content, .. } => {
                let has_tool_call = content
                    .iter()
                    .any(|c| matches!(c, AssistantContent::ToolCall(_)));
                assert!(
                    has_tool_call,
                    "assistant tool-call message should carry the tool call"
                );
            }
            _ => panic!("expected Assistant message"),
        }
    }
}
