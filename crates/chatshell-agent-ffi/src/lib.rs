#[cfg(target_os = "android")]
mod android_tls;

use std::collections::HashMap;
use std::sync::Arc;

use chatshell_agent_core as core;
use once_cell::sync::Lazy;
use tokio::runtime::Runtime;
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("ChatShell agent runtime must initialize"));

/// Outcome of [`Generation::try_next_event`].
#[derive(Debug, uniffi::Enum)]
pub enum PollEvent {
    Event { event: AgentEvent },
    Done,
    Pending,
}

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

/// Client-construction family a provider routes to. Mirrors
/// `chatshell_agent_core::provider_catalog::Wire`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum Wire {
    Responses,
    OpenAiCompat,
    Anthropic,
    OpenRouter,
    Ollama,
    Azure,
    Gemini,
    RigNative,
}

/// Routing data for one provider type. Mirrors
/// `chatshell_agent_core::provider_catalog::ProviderRoute`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ProviderRoute {
    pub wire: Wire,
    pub kind: Option<ProviderKind>,
    pub default_base_url: Option<String>,
    pub key_required: bool,
    pub key_display: Option<String>,
    pub local_no_key: bool,
    pub string_content_only: bool,
    pub default_api_style: Option<String>,
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
            // Surface terminal errors as events: dropping the sender without
            // one would make clients see a silent, empty stream.
            let result =
                core::run_generation(request.into(), sender.clone(), task_cancellation).await;
            if let Err(error) = result {
                let _ = sender.send(Err(error.into())).await;
            }
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

    /// Synchronous, timeout-bounded alternative to [`Generation::next_event`].
    ///
    /// The UniFFI foreign-future callback path used by `next_event` can stall
    /// in some client environments; polling from the client side avoids it.
    /// A timeout yields `PollEvent::Pending`, not an error: callers simply
    /// poll again until `Done` or an error arrives.
    pub fn try_next_event(&self, timeout_ms: u64) -> Result<PollEvent, AgentError> {
        let result = RUNTIME.block_on(async {
            let mut receiver = self.receiver.lock().await;
            tokio::time::timeout(
                std::time::Duration::from_millis(timeout_ms),
                receiver.recv(),
            )
            .await
        });
        match result {
            Ok(Some(Ok(event))) => Ok(PollEvent::Event {
                event: event.into(),
            }),
            Ok(Some(Err(error))) => Err(error.into()),
            Ok(None) => Ok(PollEvent::Done),
            Err(_) => Ok(PollEvent::Pending),
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

/// Resolve routing data for a provider type (wire family, kind, default
/// URL, key rules, string-content quirk, default api_style). `None` for
/// unknown provider types.
#[uniffi::export]
pub fn resolve_provider_route(provider_type: String) -> Option<ProviderRoute> {
    core::provider_catalog::resolve(&provider_type).map(Into::into)
}

/// Expand flattened history rows (a JSON-encoded `HistoryRow` array) into
/// the replayable `ChatMessage` sequence. The assistant/tool/final-text
/// pairing protocol lives in core; clients only flatten persistence rows.
#[uniffi::export]
pub fn pair_history(rows_json: String) -> Result<Vec<ChatMessage>, AgentError> {
    let rows: Vec<core::history::HistoryRow> =
        serde_json::from_str(&rows_json).map_err(|error| AgentError::InvalidProvider {
            message: format!("invalid history rows JSON: {error}"),
        })?;
    Ok(core::history::pair_history(&rows)
        .into_iter()
        .map(Into::into)
        .collect())
}

/// Provider-key encryption errors surfaced to Swift.
#[derive(Debug, Clone, thiserror::Error, uniffi::Error)]
pub enum CryptoError {
    #[error("Crypto operation failed: {message}")]
    Failure { message: String },
}

impl CryptoError {
    fn from_anyhow(error: anyhow::Error) -> Self {
        Self::Failure {
            message: format!("{error:#}"),
        }
    }
}

/// Fresh random 32-byte master key, base64-encoded. Callers store it in the
/// platform keychain; it never persists anywhere else.
#[uniffi::export]
pub fn generate_master_key_b64() -> String {
    use base64::{Engine as _, engine::general_purpose};
    general_purpose::STANDARD.encode(core::crypto::generate_master_key())
}

/// Encrypt under a base64 master key; returns `base64(nonce ‖ ciphertext)`,
/// the exact format desktop's `providers.api_key` column carries.
#[uniffi::export]
pub fn encrypt_with_master_key(
    master_key_b64: String,
    plaintext: String,
) -> Result<String, CryptoError> {
    decode_master_key(&master_key_b64)
        .and_then(|key| core::crypto::encrypt(&key, &plaintext))
        .map_err(CryptoError::from_anyhow)
}

/// Decrypt `providers.api_key`-format ciphertext under a base64 master key.
#[uniffi::export]
pub fn decrypt_with_master_key(
    master_key_b64: String,
    ciphertext_b64: String,
) -> Result<String, CryptoError> {
    decode_master_key(&master_key_b64)
        .and_then(|key| core::crypto::decrypt(&key, &ciphertext_b64))
        .map_err(CryptoError::from_anyhow)
}

/// BLAKE3 hex digest of `data` — identical to desktop's
/// `storage::hash_bytes`. Names the content-addressed blob sidecar files and
/// fills `files.content_hash` on platforms without a local blake3 binding.
#[uniffi::export]
pub fn blake3_hex(data: Vec<u8>) -> String {
    core::crypto::blake3_hex(&data)
}

/// How a device should settle its master key (pure policy, shared with the
/// desktop app): the iCloud-synchronizable item is the authority; a device
/// that sees it adopts it (demoting a different local key to the decrypt
/// fallback); with nothing anywhere, mint.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum MasterKeyResolution {
    /// The existing local key stands (item equals local, or no item).
    KeepCurrent,
    /// The synchronizable item wins; `demote_current` is true when a
    /// different local key must be kept as the decrypt fallback.
    AdoptSync { demote_current: bool },
    /// No key exists anywhere: mint one, persist locally, write the item
    /// best-effort.
    Mint,
}

/// Master-key resolution policy for one device. `sync_item` is the
/// iCloud-synchronizable keychain item value (None when unreadable or
/// absent); `local_key` is this device's local copy (None when absent).
#[uniffi::export]
pub fn resolve_master_key(
    sync_item: Option<String>,
    local_key: Option<String>,
) -> MasterKeyResolution {
    match core::crypto::resolve_master_key(sync_item.as_deref(), local_key.as_deref()) {
        core::crypto::MasterKeyResolution::KeepCurrent => MasterKeyResolution::KeepCurrent,
        core::crypto::MasterKeyResolution::AdoptSync { demote_current } => {
            MasterKeyResolution::AdoptSync { demote_current }
        }
        core::crypto::MasterKeyResolution::Mint => MasterKeyResolution::Mint,
    }
}

fn decode_master_key(master_key_b64: &str) -> anyhow::Result<[u8; core::crypto::MASTER_KEY_LEN]> {
    use base64::{Engine as _, engine::general_purpose};
    let bytes = general_purpose::STANDARD
        .decode(master_key_b64)
        .map_err(|e| anyhow::anyhow!("invalid master key base64: {e}"))?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("master key must be {} bytes", core::crypto::MASTER_KEY_LEN))
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

/// Parse DuckDuckGo HTML search results. Pure function - caller fetches
/// the HTML.
#[uniffi::export]
pub fn parse_duckduckgo_results(html: String, max: u64) -> Vec<SearchResult> {
    core::web_search::parse_duckduckgo_results(&html, max as usize)
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Resolve DuckDuckGo's redirect wrapper to the actual URL.
#[uniffi::export]
pub fn resolve_duckduckgo_redirect_url(href: String) -> String {
    core::web_search::resolve_duckduckgo_redirect_url(&href)
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

impl From<core::ProviderKind> for ProviderKind {
    fn from(value: core::ProviderKind) -> Self {
        match value {
            core::ProviderKind::OpenAi => Self::OpenAi,
            core::ProviderKind::Anthropic => Self::Anthropic,
            core::ProviderKind::OpenRouter => Self::OpenRouter,
            core::ProviderKind::Ollama => Self::Ollama,
            core::ProviderKind::CustomOpenAi => Self::CustomOpenAi,
        }
    }
}

impl From<core::provider_catalog::Wire> for Wire {
    fn from(value: core::provider_catalog::Wire) -> Self {
        match value {
            core::provider_catalog::Wire::Responses => Self::Responses,
            core::provider_catalog::Wire::OpenAiCompat => Self::OpenAiCompat,
            core::provider_catalog::Wire::Anthropic => Self::Anthropic,
            core::provider_catalog::Wire::OpenRouter => Self::OpenRouter,
            core::provider_catalog::Wire::Ollama => Self::Ollama,
            core::provider_catalog::Wire::Azure => Self::Azure,
            core::provider_catalog::Wire::Gemini => Self::Gemini,
            core::provider_catalog::Wire::RigNative => Self::RigNative,
        }
    }
}

impl From<core::provider_catalog::ProviderRoute> for ProviderRoute {
    fn from(value: core::provider_catalog::ProviderRoute) -> Self {
        Self {
            wire: value.wire.into(),
            kind: value.kind.map(Into::into),
            default_base_url: value.default_base_url.map(str::to_string),
            key_required: value.key_required,
            key_display: value.key_display.map(str::to_string),
            local_no_key: value.local_no_key,
            string_content_only: value.string_content_only,
            default_api_style: value.default_api_style.map(str::to_string),
        }
    }
}

impl From<core::MessageRole> for MessageRole {
    fn from(value: core::MessageRole) -> Self {
        match value {
            core::MessageRole::System => Self::System,
            core::MessageRole::User => Self::User,
            core::MessageRole::Assistant => Self::Assistant,
            core::MessageRole::Tool => Self::Tool,
        }
    }
}

impl From<core::ChatMessageImage> for ChatMessageImage {
    fn from(value: core::ChatMessageImage) -> Self {
        Self {
            base64: value.base64,
            media_type: value.media_type,
        }
    }
}

impl From<core::ChatToolCall> for ChatToolCall {
    fn from(value: core::ChatToolCall) -> Self {
        Self {
            id: value.id,
            call_id: value.call_id,
            name: value.name,
            arguments_json: value.arguments_json,
        }
    }
}

impl From<core::ChatMessage> for ChatMessage {
    fn from(value: core::ChatMessage) -> Self {
        Self {
            role: value.role.into(),
            content: value.content,
            reasoning_content: value.reasoning_content,
            images: value.images.into_iter().map(Into::into).collect(),
            tool_calls: value.tool_calls.into_iter().map(Into::into).collect(),
            tool_call_id: value.tool_call_id,
            tool_result_call_id: value.tool_result_call_id,
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
    /// The cloud artifact rejected this device's key material (wrong or
    /// missing content key — sync group reset elsewhere, rotation we never
    /// learned). Hosts drop the engine and re-enter the acquisition ladder
    /// at the passphrase rung (ADR 04 §3 rule 2) — never an error loop.
    #[error("content key rejected: {message}")]
    KeyMaterial { message: String },
    #[error("sync engine unavailable (poisoned lock)")]
    Unavailable,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct SyncOutcome {
    pub action: String,
    pub rows_merged: u64,
    pub republished: bool,
}

/// Map an engine error, preserving the key-material classification that
/// core's typed `SyncCryptoError` carries through anyhow.
fn classify_sync_error(e: anyhow::Error) -> SyncError {
    if let Some(core_err) = e.downcast_ref::<core::sync_crypto::SyncCryptoError>()
        && matches!(
            core_err,
            core::sync_crypto::SyncCryptoError::Auth
                | core::sync_crypto::SyncCryptoError::UnknownKeyVersion(_)
        )
    {
        return SyncError::KeyMaterial {
            message: e.to_string(),
        };
    }
    SyncError::Failed {
        message: e.to_string(),
    }
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

/// The canonical sync schema as one idempotent SQL batch (CREATE TABLE IF
/// NOT EXISTS, parents-first). Clients whose schema predates the shared
/// registry apply this in their own migration tooling (e.g. GRDB on iOS) so
/// every device carries the same table/column shape for snapshot merging.
#[uniffi::export]
pub fn current_sync_schema_sql() -> String {
    core::sync_schema::SYNC_SCHEMA_SQL.to_string()
}

#[derive(uniffi::Object)]
pub struct SyncEngine {
    inner: std::sync::Mutex<core::sync::SyncEngine>,
}

#[uniffi::export]
impl SyncEngine {
    /// Open a sync side connection. `cloud_dir` must already exist (the
    /// device that owns the iCloud container provisions it). `cache_dir` is
    /// the app-private staging directory for decrypted snapshots — never a
    /// shared temp dir (ADR 04 §1). `crypto` enables encrypted transport;
    /// `None` keeps the legacy plaintext artifacts (debug staging only).
    #[uniffi::constructor]
    pub fn new(
        db_path: String,
        cloud_dir: String,
        cache_dir: String,
        crypto: Option<std::sync::Arc<SyncCrypto>>,
    ) -> Result<Self, SyncError> {
        let engine = core::sync::SyncEngine::new(
            std::path::Path::new(&db_path),
            std::path::PathBuf::from(cloud_dir),
            std::path::PathBuf::from(cache_dir),
            crypto.map(|c| c.inner.lock().expect("crypto lock").clone()),
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
            .map_err(classify_sync_error)
    }

    /// One sync pass: merge remote changes, then republish when needed.
    pub fn sync_now(&self) -> Result<SyncOutcome, SyncError> {
        self.inner
            .lock()
            .map_err(|_| SyncError::Unavailable)?
            .sync_now()
            .map(Into::into)
            .map_err(classify_sync_error)
    }

    /// The engine's current content-key chain (may have gained carried
    /// slots after a pass saw a rotated remote header), serialized for the
    /// device-local cache file. `None` when this engine does not encrypt.
    pub fn sync_crypto_chain(&self) -> Option<Vec<u8>> {
        self.inner
            .lock()
            .ok()
            .and_then(|engine| engine.crypto().map(|c| c.keys().encode()))
    }
}

// ---------------------------------------------------------------------------
// Transport encryption (ADR 04). One cryptographic implementation shared by
// every platform; the host owns Keychain access and file IO.
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum SyncCryptoError {
    #[error("{message}")]
    Failed { message: String },
    #[error("artifact is corrupt or foreign: {message}")]
    Corrupt { message: String },
    #[error("unsupported transport format version {version}")]
    UnsupportedFormat { version: u16 },
    #[error("content key version {version} is not held on this device")]
    UnknownKeyVersion { version: u32 },
    #[error("decryption failed (wrong key or tampered data)")]
    Auth,
    #[error("wrong passphrase")]
    WrongPassphrase,
}

impl From<core::sync_crypto::SyncCryptoError> for SyncCryptoError {
    fn from(e: core::sync_crypto::SyncCryptoError) -> Self {
        match e {
            core::sync_crypto::SyncCryptoError::Corrupt(m) => {
                SyncCryptoError::Corrupt { message: m }
            }
            core::sync_crypto::SyncCryptoError::UnsupportedFormat(v) => {
                SyncCryptoError::UnsupportedFormat { version: v }
            }
            core::sync_crypto::SyncCryptoError::UnknownKeyVersion(v) => {
                SyncCryptoError::UnknownKeyVersion { version: v }
            }
            core::sync_crypto::SyncCryptoError::Auth => SyncCryptoError::Auth,
            core::sync_crypto::SyncCryptoError::WrongPassphrase => SyncCryptoError::WrongPassphrase,
            other => SyncCryptoError::Failed {
                message: other.to_string(),
            },
        }
    }
}

/// Generate a high-entropy diceware-style sync passphrase (shown once with
/// save guidance; never stored by the app — ADR 04 §3).
#[uniffi::export]
pub fn generate_sync_passphrase() -> String {
    core::sync_crypto::generate_passphrase()
}

/// Serialize a single-entry chain (Keychain adoption: the item holds only
/// the current key; its version comes from the artifact header). Keeps the
/// cache format owned by Rust.
#[uniffi::export]
pub fn sync_chain_from_single_key(
    version: u32,
    key_b64: String,
) -> Result<Vec<u8>, SyncCryptoError> {
    use base64::{Engine as _, engine::general_purpose};
    let key: [u8; core::sync_crypto::CK_LEN] = general_purpose::STANDARD
        .decode(key_b64)
        .map_err(|_| SyncCryptoError::Failed {
            message: "content key must be base64 of 32 bytes".into(),
        })?
        .try_into()
        .map_err(|_| SyncCryptoError::Failed {
            message: "content key must be base64 of 32 bytes".into(),
        })?;
    let keys = core::sync_crypto::ContentKeys::from_entries(vec![(version, key)])?;
    Ok(keys.encode())
}

/// Parsed Apple-slot synchronizable-item value (ADR 05): the key version
/// plus the raw CK. Hosts adopt the item ONLY on an exact version match
/// with the artifact header's `key_version` — a stale item must fall
/// through to the cached-chain rung, never install a wrong-key engine.
#[derive(Debug, Clone, uniffi::Record)]
pub struct SyncSynchronizableItem {
    pub version: u32,
    pub key_b64: String,
}

/// Parse a synchronizable-item value (`v<version>:<base64 key>`). Returns
/// `None` for legacy unversioned values and anything malformed; hosts
/// treat `None` as "no item" (ladder rung 1 miss).
#[uniffi::export]
pub fn parse_synchronizable_item(value: String) -> Option<SyncSynchronizableItem> {
    use base64::{Engine as _, engine::general_purpose};
    let (version, key) = core::sync_crypto::parse_synchronizable_item(&value)?;
    Some(SyncSynchronizableItem {
        version,
        key_b64: general_purpose::STANDARD.encode(key),
    })
}

/// Plaintext header of an encrypted artifact, minus the slot payloads
/// (hosts never need raw slot bytes — `SyncCrypto.from_chain` re-carries
/// them from the artifact path itself).
#[derive(Debug, Clone, uniffi::Record)]
pub struct ArtifactHeaderInfo {
    pub sync_version: String,
    pub schema_version: u64,
    pub key_version: u32,
    pub icloud_slot: bool,
}

#[uniffi::export]
pub fn read_artifact_header_info(path: String) -> Result<ArtifactHeaderInfo, SyncCryptoError> {
    let header = core::sync_crypto::read_artifact_header(std::path::Path::new(&path))?;
    Ok(ArtifactHeaderInfo {
        sync_version: header.sync_version,
        schema_version: header.schema_version,
        key_version: header.key_version,
        icloud_slot: header.icloud_slot,
    })
}

/// Transport-encryption state for one sync group: the content-key chain
/// plus the Argon2id slot bytes carried in artifact headers. Hosts keep the
/// chain in their own cache (`chain_bytes`) and the current key in a
/// dedicated iCloud Keychain synchronizable item (`synchronizable_item_value`).
#[derive(uniffi::Object)]
pub struct SyncCrypto {
    inner: std::sync::Mutex<core::sync_crypto::SyncCrypto>,
}

#[uniffi::export]
impl SyncCrypto {
    /// First device of a sync group: mint a content key and wrap it under
    /// the group passphrase (required even on Apple platforms so the
    /// fallback exists when the Keychain slot fails — ADR 04 §3 rule 1).
    #[uniffi::constructor]
    pub fn bootstrap(passphrase: String) -> Result<Self, SyncCryptoError> {
        let inner = core::sync_crypto::SyncCrypto::bootstrap(&passphrase)?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    /// Rebuild from the device-local chain cache. When `artifact_path`
    /// points at an existing remote artifact, its Argon2id slots are
    /// carried forward verbatim so passphrase-only peers keep their
    /// fallback (Apple platforms always set the marker slot).
    #[uniffi::constructor]
    pub fn from_chain(
        chain: Vec<u8>,
        artifact_path: Option<String>,
    ) -> Result<Self, SyncCryptoError> {
        let keys = core::sync_crypto::ContentKeys::decode(&chain)?;
        let slots = match artifact_path.as_deref().filter(|p| !p.is_empty()) {
            Some(path) => {
                let header = core::sync_crypto::read_artifact_header(std::path::Path::new(path))?;
                header.slots
            }
            None => Vec::new(),
        };
        let inner = core::sync_crypto::SyncCrypto::from_keys(keys, slots, true)?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    /// Passphrase rung of the acquisition ladder: unwrap every slot of the
    /// remote artifact header into the full chain (unioned with a locally
    /// cached chain when supplied), e.g. after a rotation elsewhere or a
    /// sync-group reset.
    #[uniffi::constructor]
    pub fn unlock_from_artifact(
        artifact_path: String,
        passphrase: String,
        known_chain: Option<Vec<u8>>,
    ) -> Result<Self, SyncCryptoError> {
        let header = core::sync_crypto::read_artifact_header(std::path::Path::new(&artifact_path))?;
        let known = match known_chain {
            Some(bytes) => Some(core::sync_crypto::ContentKeys::decode(&bytes)?),
            None => None,
        };
        let inner = core::sync_crypto::SyncCrypto::from_passphrase(
            &header,
            &passphrase,
            known.as_ref(),
            true,
        )?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    /// Serialized chain for the device-local cache file.
    pub fn chain_bytes(&self) -> Vec<u8> {
        self.inner.lock().expect("crypto lock").keys().encode()
    }

    /// Highest chain version — used for every new write.
    pub fn current_key_version(&self) -> u32 {
        self.inner
            .lock()
            .expect("crypto lock")
            .keys()
            .current_version()
    }

    /// Explicit user action (suspected compromise, lost device): mint the
    /// next CK under the passphrase. Forward-only revocation — ADR 04 §5.
    /// The carried slots come from the artifact the object was built with,
    /// so the caller must publish promptly after rotating.
    pub fn rotate(&self, passphrase: String) -> Result<(), SyncCryptoError> {
        self.inner
            .lock()
            .expect("crypto lock")
            .rotate(passphrase.trim())
            .map_err(Into::into)
    }

    /// Value for the Apple-slot synchronizable item, or `None` once the
    /// group has rotated (ADR 05 freeze rule): the item may only ever
    /// carry the bootstrap-era (v1) key. Refreshing it after a rotation
    /// would re-deliver the rotated CK to every device under the Apple ID —
    /// including a lost one — defeating forward-only revocation. Hosts
    /// write the item only when this returns a value.
    pub fn synchronizable_item_value(&self) -> Option<String> {
        self.inner
            .lock()
            .expect("crypto lock")
            .synchronizable_item_value()
    }

    /// Encrypt attachment bytes for the blob sidecar (`BlobStore` stays
    /// IO-only; this is the single crypto implementation).
    pub fn encrypt_blob(
        &self,
        digest_hex: String,
        plaintext: Vec<u8>,
    ) -> Result<Vec<u8>, SyncCryptoError> {
        self.inner
            .lock()
            .expect("crypto lock")
            .encrypt_blob(&digest_hex, &plaintext)
            .map_err(Into::into)
    }

    /// Decrypt a blob sidecar (verifies the content hash after decrypt).
    pub fn decrypt_blob(
        &self,
        digest_hex: String,
        ciphertext: Vec<u8>,
    ) -> Result<Vec<u8>, SyncCryptoError> {
        self.inner
            .lock()
            .expect("crypto lock")
            .decrypt_blob(&digest_hex, &ciphertext)
            .map_err(Into::into)
    }
}
