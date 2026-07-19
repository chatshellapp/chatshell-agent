use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OpenAi,
    Anthropic,
    OpenRouter,
    Ollama,
    CustomOpenAi,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// Base64-encoded image attached to a user message. Mirrors desktop `ImageData`:
/// `base64` contains raw base64 data (no `data:` URL prefix), `media_type` is a
/// MIME string such as `image/png`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessageImage {
    pub base64: String,
    pub media_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub reasoning_content: Option<String>,
    /// Optional images on user messages. Ignored for assistant/system messages.
    #[serde(default)]
    pub images: Vec<ChatMessageImage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingParameters {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRequest {
    pub provider: ProviderConfig,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub parameters: SamplingParameters,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum AgentError {
    #[error("API key is required for {0}")]
    MissingApiKey(&'static str),
    #[error("No user message was provided")]
    MissingUserMessage,
    #[error("Invalid provider configuration: {0}")]
    InvalidProvider(String),
    #[error("Provider request failed: {0}")]
    Provider(String),
    #[error("Generation was cancelled")]
    Cancelled,
    #[error("The event consumer was closed")]
    ConsumerClosed,
}
