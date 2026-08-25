mod agent;
pub mod compaction;
pub mod context_pruning;
pub mod edit_anchor;
pub mod fuzzy_match;
pub mod history;
pub mod mcp;
pub mod openai_compat;
pub mod provider_catalog;
pub mod provider_registry;
pub mod snapcompact;
pub mod sync;
pub mod thinking;
pub mod tool_definitions;
pub mod types;
pub mod web_fetch;
pub mod web_search;

pub use agent::{create_http_client, run_generation};
pub use history::{HistoryRole, HistoryRow, ToolCallRow, pair_history};
pub use mcp::{McpConnection, McpHttpConfig, McpToolInfo, McpToolResult};
pub use provider_registry::{
    BuiltinProvider, builtin_providers, custom_provider, is_supported_fetch_provider,
    requires_api_key_for_model_fetch,
};
pub use thinking::{ParsedContent, parse_thinking_content};
pub use tool_definitions::{builtin_tool_definitions, web_fetch_definition, web_search_definition};
pub use types::{
    AgentError, AgentEvent, AgentRequest, ChatMessage, ChatMessageImage, ChatToolCall,
    ClientToolDefinition, MessageRole, ProviderConfig, ProviderKind, SamplingParameters,
};
pub use web_fetch::FetchedResource;
pub use web_search::SearchResult;
