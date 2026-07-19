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
    /// Selects the OpenAI API surface for `OpenAi` / `CustomOpenAi` providers.
    /// `"chat_completions"` forces the Chat Completions API; any other value
    /// (including `None`) selects the Responses API, matching desktop semantics.
    pub api_style: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Base64-encoded image attached to a user message. Mirrors desktop `ImageData`:
/// `base64` contains raw base64 data (no `data:` URL prefix), `media_type` is a
/// MIME string such as `image/png`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessageImage {
    pub base64: String,
    pub media_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatToolCall {
    pub id: String,
    /// OpenAI Responses API call identifier. Required when replaying tool calls
    /// back to the Responses API after client-side execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub reasoning_content: Option<String>,
    /// Optional images on user messages. Ignored for assistant/system messages.
    #[serde(default)]
    pub images: Vec<ChatMessageImage>,
    /// Tool calls emitted by the assistant. Set on assistant messages that
    /// request tool execution. Empty for non-tool-call messages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatToolCall>,
    /// Tool-call item identifier. Set on `Tool`-role messages to associate a
    /// tool result with the assistant's tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// OpenAI Responses API `call_id` for tool results. When set, takes
    /// precedence over `tool_call_id` when building provider requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result_call_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingParameters {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClientToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's parameters, as a JSON string.
    pub parameters_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRequest {
    pub provider: ProviderConfig,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub parameters: SamplingParameters,
    /// Tool definitions exposed to the model. When non-empty, the agent runs
    /// in client-side tool mode: tool calls are emitted as `ToolCallStarted`
    /// events and NOT executed server-side; the client must execute them and
    /// send a follow-up request with the tool results.
    #[serde(default)]
    pub tools: Vec<ClientToolDefinition>,
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
        call_id: Option<String>,
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
