use std::collections::HashMap;
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
    pub api_style: Option<String>,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessageImage {
    pub base64: String,
    pub media_type: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatToolCall {
    pub id: String,
    pub call_id: Option<String>,
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub reasoning_content: Option<String>,
    /// Images attached to a user message. Empty for non-user roles.
    pub images: Vec<ChatMessageImage>,
    /// Tool calls emitted by the assistant. Empty for non-tool-call messages.
    pub tool_calls: Vec<ChatToolCall>,
    /// Tool-call item identifier for `Tool`-role messages.
    pub tool_call_id: Option<String>,
    /// OpenAI Responses API `call_id` for tool results.
    pub tool_result_call_id: Option<String>,
}

#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct SamplingParameters {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params_json: Option<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ClientToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters_json: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct AgentRequest {
    pub provider: ProviderConfig,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub parameters: SamplingParameters,
    pub tools: Vec<ClientToolDefinition>,
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
/// Built-in provider template. Mirrors `chatshell_agent_core::BuiltinProvider`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct BuiltinProvider {
    pub id: String,
    pub name: String,
    pub default_base_url: String,
    pub requires_api_key: bool,
}

/// Output of thinking-content parsing. Mirrors
/// `chatshell_agent_core::ParsedContent`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ParsedContent {
    pub content: String,
    pub thinking_content: Option<String>,
    pub is_thinking_in_progress: bool,
}

/// Configuration for connecting to an MCP server via HTTP.
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpHttpConfig {
    pub endpoint: String,
    pub auth_token: Option<String>,
    pub custom_headers: HashMap<String, String>,
}

/// A tool exposed by an MCP server.
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
    pub input_schema_json: String,
}

/// Result of calling a tool on an MCP server.
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpToolResult {
    pub content: String,
    pub is_error: bool,
}

/// A fetched web resource with extracted content. Mirrors
/// `chatshell_agent_core::FetchedResource`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FetchedResource {
    pub url: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub content: String,
    pub mime_type: String,
    pub extraction_error: Option<String>,
}

/// A single web search result. Mirrors `chatshell_agent_core::SearchResult`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
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

// ---------------------------------------------------------------------------
// Shared data exports - provider registry, tool definitions, thinking parser.
// These are free functions so Swift can call them without an AgentCore handle.
// ---------------------------------------------------------------------------

/// All built-in provider templates, ordered for UI display.
#[uniffi::export]
pub fn builtin_providers() -> Vec<BuiltinProvider> {
    core::builtin_providers()
        .into_iter()
        .map(Into::into)
        .collect()
}

/// The sentinel row for creating a custom provider endpoint.
#[uniffi::export]
pub fn custom_provider() -> BuiltinProvider {
    core::custom_provider().into()
}

/// Whether a provider type supports auto-fetching models via API.
#[uniffi::export]
pub fn is_supported_fetch_provider(provider_type: String) -> bool {
    core::is_supported_fetch_provider(&provider_type)
}

/// Whether fetching models for this provider requires an API key.
#[uniffi::export]
pub fn requires_api_key_for_model_fetch(provider_type: String) -> bool {
    core::requires_api_key_for_model_fetch(&provider_type)
}

/// Tool definitions for the built-in `web_search` and `web_fetch` tools.
#[uniffi::export]
pub fn builtin_tool_definitions() -> Vec<ClientToolDefinition> {
    core::builtin_tool_definitions()
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Extract `<think>` / `<thinking>` / `<reasoning>` blocks from model output.
#[uniffi::export]
pub fn parse_thinking_content(text: String) -> ParsedContent {
    core::parse_thinking_content(&text).into()
}

// ---------------------------------------------------------------------------
// MCP HTTP transport. rmcp requires a tokio runtime, so async operations are
// spawned on RUNTIME and results are returned via oneshot channels. The Swift
// caller sees async functions; the actual work runs on the tokio runtime.
// ---------------------------------------------------------------------------

/// Connect to an MCP server via HTTP. Returns a handle that can list tools
/// and call them. Dropping the handle disconnects.
#[uniffi::export]
pub async fn connect_mcp_server(config: McpHttpConfig) -> Result<Arc<McpConnection>, AgentError> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let config_inner: core::McpHttpConfig = config.into();
    RUNTIME.spawn(async move {
        let result = core::McpConnection::connect(config_inner).await;
        let _ = tx.send(result);
    });
    let conn = rx
        .await
        .map_err(|e| AgentError::Provider {
            message: e.to_string(),
        })?
        .map_err(|e| AgentError::Provider {
            message: e.to_string(),
        })?;
    Ok(Arc::new(McpConnection {
        inner: parking_lot::Mutex::new(Some(conn)),
    }))
}

/// An active MCP server connection. Clone is cheap (Arc-based).
#[derive(uniffi::Object)]
pub struct McpConnection {
    inner: parking_lot::Mutex<Option<core::McpConnection>>,
}

#[uniffi::export]
impl McpConnection {
    /// List all tools exposed by the connected MCP server.
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>, AgentError> {
        let conn = self
            .inner
            .lock()
            .clone()
            .ok_or_else(|| AgentError::Provider {
                message: "Not connected".to_string(),
            })?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        RUNTIME.spawn(async move {
            let result = conn.list_tools().await;
            let _ = tx.send(result);
        });
        let tools = rx
            .await
            .map_err(|e| AgentError::Provider {
                message: e.to_string(),
            })?
            .map_err(|e| AgentError::Provider {
                message: e.to_string(),
            })?;
        Ok(tools.into_iter().map(Into::into).collect())
    }

    /// Call a tool on the connected MCP server.
    pub async fn call_tool(
        &self,
        name: String,
        arguments_json: String,
    ) -> Result<McpToolResult, AgentError> {
        let conn = self
            .inner
            .lock()
            .clone()
            .ok_or_else(|| AgentError::Provider {
                message: "Not connected".to_string(),
            })?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        RUNTIME.spawn(async move {
            let result = conn.call_tool(&name, &arguments_json).await;
            let _ = tx.send(result);
        });
        let result = rx
            .await
            .map_err(|e| AgentError::Provider {
                message: e.to_string(),
            })?
            .map_err(|e| AgentError::Provider {
                message: e.to_string(),
            })?;
        Ok(result.into())
    }

    /// Disconnect from the MCP server. Safe to call multiple times.
    pub async fn disconnect(&self) {
        *self.inner.lock() = None;
    }
}

// ---------------------------------------------------------------------------
// Web fetch - HTTP + Jina Reader. Same RUNTIME + oneshot pattern as MCP.
// ---------------------------------------------------------------------------

/// Fetch a web resource via HTTP and extract its main content. For HTML,
/// uses Readability + htmd (HTML-to-markdown). For JSON/XML/text, formats
/// as a code block. `max_chars` truncates the output (None = no limit).
#[uniffi::export]
pub async fn fetch_http(url: String, max_chars: Option<u64>) -> FetchedResource {
    let url_for_error = url.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    RUNTIME.spawn(async move {
        let result = core::web_fetch::fetch_http(&url, max_chars.map(|n| n as usize)).await;
        let _ = tx.send(result);
    });
    match rx.await {
        Ok(result) => result.into(),
        Err(_) => FetchedResource {
            url: url_for_error,
            title: None,
            description: None,
            content: String::new(),
            mime_type: String::new(),
            extraction_error: Some("Runtime dropped".to_string()),
        },
    }
}

/// Fetch a web resource using the Jina Reader API. Returns markdown content.
/// `api_key` is optional (Jina works without it, but rate limits apply).
#[uniffi::export]
pub async fn fetch_jina(url: String, api_key: Option<String>) -> FetchedResource {
    let url_for_error = url.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();
    RUNTIME.spawn(async move {
        let result = core::web_fetch::fetch_jina(&url, api_key.as_deref()).await;
        let _ = tx.send(result);
    });
    match rx.await {
        Ok(result) => result.into(),
        Err(_) => FetchedResource {
            url: url_for_error,
            title: None,
            description: None,
            content: String::new(),
            mime_type: String::new(),
            extraction_error: Some("Runtime dropped".to_string()),
        },
    }
}

// ---------------------------------------------------------------------------
// Web search - Brave API (async) + Yahoo/Baidu parsing (sync, pure functions).
// ---------------------------------------------------------------------------

/// Search via the Brave Search API. Requires an API key.
#[uniffi::export]
pub async fn search_brave(
    query: String,
    api_key: String,
    max_results: u64,
) -> Result<Vec<SearchResult>, AgentError> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    RUNTIME.spawn(async move {
        let result = core::web_search::search_brave(&query, &api_key, max_results as usize).await;
        let _ = tx.send(result);
    });
    rx.await
        .map_err(|e| AgentError::Provider {
            message: e.to_string(),
        })?
        .map_err(|e| AgentError::Provider { message: e })
        .map(|results| results.into_iter().map(Into::into).collect())
}

/// Parse Yahoo HTML search results. Pure function - caller fetches the HTML.
#[uniffi::export]
pub fn parse_yahoo_results(html: String, max: u64) -> Vec<SearchResult> {
    core::web_search::parse_yahoo_results(&html, max as usize)
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Parse Baidu HTML search results. Pure function - caller fetches the HTML.
#[uniffi::export]
pub fn parse_baidu_results(html: String, max: u64) -> Vec<SearchResult> {
    core::web_search::parse_baidu_results(&html, max as usize)
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Decode Yahoo's redirect URL wrapper to extract the real URL.
#[uniffi::export]
pub fn resolve_yahoo_redirect_url(href: String) -> String {
    core::web_search::resolve_yahoo_redirect_url(&href)
}

impl From<AgentRequest> for core::AgentRequest {
    fn from(value: AgentRequest) -> Self {
        Self {
            provider: value.provider.into(),
            model: value.model,
            messages: value.messages.into_iter().map(Into::into).collect(),
            parameters: value.parameters.into(),
            tools: value
                .tools
                .into_iter()
                .map(|t| core::ClientToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters_json: t.parameters_json,
                })
                .collect(),
        }
    }
}

impl From<ProviderConfig> for core::ProviderConfig {
    fn from(value: ProviderConfig) -> Self {
        Self {
            kind: value.kind.into(),
            api_key: value.api_key,
            base_url: value.base_url,
            api_style: value.api_style,
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
            tool_calls: value
                .tool_calls
                .into_iter()
                .map(|tc| core::ChatToolCall {
                    id: tc.id,
                    call_id: tc.call_id,
                    name: tc.name,
                    arguments_json: tc.arguments_json,
                })
                .collect(),
            tool_call_id: value.tool_call_id,
            tool_result_call_id: value.tool_result_call_id,
        }
    }
}

impl From<MessageRole> for core::MessageRole {
    fn from(value: MessageRole) -> Self {
        match value {
            MessageRole::System => Self::System,
            MessageRole::User => Self::User,
            MessageRole::Assistant => Self::Assistant,
            MessageRole::Tool => Self::Tool,
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
                call_id,
                name,
                arguments_json,
            } => Self::ToolCallStarted {
                id,
                call_id,
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

impl From<core::BuiltinProvider> for BuiltinProvider {
    fn from(value: core::BuiltinProvider) -> Self {
        Self {
            id: value.id,
            name: value.name,
            default_base_url: value.default_base_url,
            requires_api_key: value.requires_api_key,
        }
    }
}

impl From<core::ParsedContent> for ParsedContent {
    fn from(value: core::ParsedContent) -> Self {
        Self {
            content: value.content,
            thinking_content: value.thinking_content,
            is_thinking_in_progress: value.is_thinking_in_progress,
        }
    }
}

impl From<core::ClientToolDefinition> for ClientToolDefinition {
    fn from(value: core::ClientToolDefinition) -> Self {
        Self {
            name: value.name,
            description: value.description,
            parameters_json: value.parameters_json,
        }
    }
}

impl From<McpHttpConfig> for core::McpHttpConfig {
    fn from(value: McpHttpConfig) -> Self {
        Self {
            endpoint: value.endpoint,
            auth_token: value.auth_token,
            custom_headers: value.custom_headers,
        }
    }
}

impl From<core::McpToolInfo> for McpToolInfo {
    fn from(value: core::McpToolInfo) -> Self {
        Self {
            name: value.name,
            description: value.description,
            input_schema_json: value.input_schema_json,
        }
    }
}

impl From<core::McpToolResult> for McpToolResult {
    fn from(value: core::McpToolResult) -> Self {
        Self {
            content: value.content,
            is_error: value.is_error,
        }
    }
}

impl From<core::FetchedResource> for FetchedResource {
    fn from(value: core::FetchedResource) -> Self {
        Self {
            url: value.url,
            title: value.title,
            description: value.description,
            content: value.content,
            mime_type: value.mime_type,
            extraction_error: value.extraction_error,
        }
    }
}

impl From<core::SearchResult> for SearchResult {
    fn from(value: core::SearchResult) -> Self {
        Self {
            title: value.title,
            url: value.url,
            snippet: value.snippet,
        }
    }
}

uniffi::setup_scaffolding!();

// ---------------------------------------------------------------------------
// Snapshot sync (ADR 01). The engine runs on the caller's database file
// through a dedicated side connection; the host keeps its own async access.
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum SyncError {
    #[error("{message}")]
    Failed { message: String },
    #[error("sync engine unavailable (poisoned lock)")]
    Unavailable,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct SyncOutcome {
    pub action: String,
    pub rows_merged: u64,
    pub republished: bool,
}

impl From<core::sync::SyncOutcome> for SyncOutcome {
    fn from(outcome: core::sync::SyncOutcome) -> Self {
        Self {
            action: outcome.action,
            rows_merged: outcome.rows_merged as u64,
            republished: outcome.republished,
        }
    }
}

#[derive(uniffi::Object)]
pub struct SyncEngine {
    inner: std::sync::Mutex<core::sync::SyncEngine>,
}

#[uniffi::export]
impl SyncEngine {
    /// Open a sync side connection. `cloud_dir` must already exist (the
    /// device that owns the iCloud container provisions it).
    #[uniffi::constructor]
    pub fn new(db_path: String, cloud_dir: String) -> Result<Self, SyncError> {
        let engine = core::sync::SyncEngine::new(
            std::path::Path::new(&db_path),
            std::path::PathBuf::from(cloud_dir),
        )
        .map_err(|e| SyncError::Failed {
            message: e.to_string(),
        })?;
        Ok(Self {
            inner: std::sync::Mutex::new(engine),
        })
    }

    /// Publish the current state to the cloud snapshot.
    pub fn publish(&self) -> Result<(), SyncError> {
        self.inner
            .lock()
            .map_err(|_| SyncError::Unavailable)?
            .publish()
            .map_err(|e| SyncError::Failed {
                message: e.to_string(),
            })
    }

    /// One sync pass: merge remote changes, then republish when needed.
    pub fn sync_now(&self) -> Result<SyncOutcome, SyncError> {
        self.inner
            .lock()
            .map_err(|_| SyncError::Unavailable)?
            .sync_now()
            .map(Into::into)
            .map_err(|e| SyncError::Failed {
                message: e.to_string(),
            })
    }
}
