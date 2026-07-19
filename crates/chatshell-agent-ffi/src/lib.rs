use std::sync::Arc;

use chatshell_agent_core as core;
use once_cell::sync::Lazy;
use tokio::runtime::Runtime;
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("ChatShell agent runtime must initialize"));

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum ProviderKind {
    OpenAi,
    Anthropic,
    OpenRouter,
    Ollama,
    CustomOpenAi,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessageImage {
    pub base64: String,
    pub media_type: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub reasoning_content: Option<String>,
    /// Images attached to a user message. Empty for non-user roles.
    pub images: Vec<ChatMessageImage>,
}

#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct SamplingParameters {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params_json: Option<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct AgentRequest {
    pub provider: ProviderConfig,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub parameters: SamplingParameters,
}

#[derive(Debug, Clone, uniffi::Enum)]
pub enum AgentEvent {
    TextDelta {
        text: String,
    },
    ReasoningDelta {
        text: String,
    },
    ToolCallStarted {
        id: String,
        name: String,
        arguments_json: String,
    },
    ToolCallCompleted {
        id: String,
        output: String,
    },
    Image {
        data_url: String,
    },
    Completed {
        content: String,
        reasoning: Option<String>,
        input_tokens: u64,
        output_tokens: u64,
        cancelled: bool,
    },
}

#[derive(Debug, Clone, thiserror::Error, uniffi::Error)]
pub enum AgentError {
    #[error("API key is required: {message}")]
    MissingApiKey { message: String },
    #[error("No user message was provided")]
    MissingUserMessage,
    #[error("Invalid provider configuration: {message}")]
    InvalidProvider { message: String },
    #[error("Provider request failed: {message}")]
    Provider { message: String },
    #[error("Generation was cancelled")]
    Cancelled,
    #[error("The event consumer was closed")]
    ConsumerClosed,
}

#[derive(uniffi::Object)]
pub struct AgentCore;

#[uniffi::export]
impl AgentCore {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }

    pub fn start_generation(&self, request: AgentRequest) -> Arc<Generation> {
        let (sender, receiver) = mpsc::channel(64);
        let cancellation = CancellationToken::new();
        let task_cancellation = cancellation.clone();

        RUNTIME.spawn(async move {
            let _ = core::run_generation(request.into(), sender, task_cancellation).await;
        });

        Arc::new(Generation {
            receiver: Mutex::new(receiver),
            cancellation,
        })
    }
}

#[derive(uniffi::Object)]
pub struct Generation {
    receiver: Mutex<mpsc::Receiver<Result<core::AgentEvent, core::AgentError>>>,
    cancellation: CancellationToken,
}

#[uniffi::export]
impl Generation {
    pub async fn next_event(&self) -> Result<Option<AgentEvent>, AgentError> {
        match self.receiver.lock().await.recv().await {
            Some(Ok(event)) => Ok(Some(event.into())),
            Some(Err(error)) => Err(error.into()),
            None => Ok(None),
        }
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }
}

impl From<AgentRequest> for core::AgentRequest {
    fn from(value: AgentRequest) -> Self {
        Self {
            provider: value.provider.into(),
            model: value.model,
            messages: value.messages.into_iter().map(Into::into).collect(),
            parameters: value.parameters.into(),
        }
    }
}

impl From<ProviderConfig> for core::ProviderConfig {
    fn from(value: ProviderConfig) -> Self {
        Self {
            kind: value.kind.into(),
            api_key: value.api_key,
            base_url: value.base_url,
        }
    }
}

impl From<ProviderKind> for core::ProviderKind {
    fn from(value: ProviderKind) -> Self {
        match value {
            ProviderKind::OpenAi => Self::OpenAi,
            ProviderKind::Anthropic => Self::Anthropic,
            ProviderKind::OpenRouter => Self::OpenRouter,
            ProviderKind::Ollama => Self::Ollama,
            ProviderKind::CustomOpenAi => Self::CustomOpenAi,
        }
    }
}

impl From<ChatMessage> for core::ChatMessage {
    fn from(value: ChatMessage) -> Self {
        Self {
            role: value.role.into(),
            content: value.content,
            reasoning_content: value.reasoning_content,
            images: value
                .images
                .into_iter()
                .map(|image| core::ChatMessageImage {
                    base64: image.base64,
                    media_type: image.media_type,
                })
                .collect(),
        }
    }
}

impl From<MessageRole> for core::MessageRole {
    fn from(value: MessageRole) -> Self {
        match value {
            MessageRole::System => Self::System,
            MessageRole::User => Self::User,
            MessageRole::Assistant => Self::Assistant,
        }
    }
}

impl From<SamplingParameters> for core::SamplingParameters {
    fn from(value: SamplingParameters) -> Self {
        Self {
            temperature: value.temperature,
            max_tokens: value.max_tokens,
            additional_params_json: value.additional_params_json,
        }
    }
}

impl From<core::AgentEvent> for AgentEvent {
    fn from(value: core::AgentEvent) -> Self {
        match value {
            core::AgentEvent::TextDelta { text } => Self::TextDelta { text },
            core::AgentEvent::ReasoningDelta { text } => Self::ReasoningDelta { text },
            core::AgentEvent::ToolCallStarted {
                id,
                name,
                arguments_json,
            } => Self::ToolCallStarted {
                id,
                name,
                arguments_json,
            },
            core::AgentEvent::ToolCallCompleted { id, output } => {
                Self::ToolCallCompleted { id, output }
            }
            core::AgentEvent::Image { data_url } => Self::Image { data_url },
            core::AgentEvent::Completed {
                content,
                reasoning,
                input_tokens,
                output_tokens,
                cancelled,
            } => Self::Completed {
                content,
                reasoning,
                input_tokens,
                output_tokens,
                cancelled,
            },
        }
    }
}

impl From<core::AgentError> for AgentError {
    fn from(value: core::AgentError) -> Self {
        match value {
            core::AgentError::MissingApiKey(provider) => Self::MissingApiKey {
                message: provider.to_string(),
            },
            core::AgentError::MissingUserMessage => Self::MissingUserMessage,
            core::AgentError::InvalidProvider(message) => Self::InvalidProvider { message },
            core::AgentError::Provider(message) => Self::Provider { message },
            core::AgentError::Cancelled => Self::Cancelled,
            core::AgentError::ConsumerClosed => Self::ConsumerClosed,
        }
    }
}

uniffi::setup_scaffolding!();
