//! MCP (Model Context Protocol) client layer shared across desktop and iOS.
//!
//! Only the HTTP (Streamable HTTP) transport is constructed here; STDIO stays
//! desktop-only (iOS cannot spawn child processes) but reuses the handler,
//! listing, call, and result helpers. OAuth flow stays platform-specific
//! (desktop uses system browser + callback server, iOS uses
//! ASWebAuthenticationSession).
//!
//! Shared pieces:
//! - [`ChatShellClientHandler`]: client identity + server notifications
//!   (`tools/list_changed` debounced, progress routing, server logs).
//! - [`list_all_tools_bounded`]: cursor-following listing with guards.
//! - [`call_tool_with_options`]: deadline, progress, cancel-on-drop.
//! - [`render_call_tool_result`]: content + `structuredContent` rendering.
//! - [`render_read_resource_result`]: resource contents (blobs spilled).
//!
//! Auth token (bearer) is passed in via `McpHttpConfig`; the caller is
//! responsible for obtaining it (plaintext, keychain, OAuth, etc.).

mod call;
mod connection;
mod elicitation;
mod handler;
mod listing;
mod prompt;
mod resource;
mod result;
mod types;

pub use call::{
    DEFAULT_CALL_TIMEOUT_SECS, DEFAULT_MAX_TOTAL_TIMEOUT_SECS, McpCallOptions,
    call_tool_with_options, read_resource_with_options,
};
pub use connection::{McpClientService, McpConnection, McpHttpConfig, connect_http_service};
pub use elicitation::{
    ELICITATION_TIMEOUT, McpElicitationAnswer, McpElicitationPrompter, McpElicitationRequest,
    scan_answer, validate_form_content,
};
pub use handler::{
    ChatShellClientHandler, LIST_CHANGED_DEBOUNCE, McpNotificationSink, chatshell_client_info,
    prepare_client_peer,
};
pub use listing::{
    ListTruncation, MAX_CURSOR_BYTES, MAX_LIST_ITEMS, MAX_LIST_PAGES, Paginated,
    list_all_resource_templates_bounded, list_all_resources_bounded, list_all_tools_bounded,
    paginate, supports_resources,
};
pub use prompt::{
    MAX_PROMPT_CHARS, McpPromptArgument, McpPromptInfo, get_prompt_text, list_all_prompts_bounded,
    render_prompt, scan_prompt_arguments, supports_prompts,
};
pub use resource::{
    MAX_RESOURCE_BLOB_BYTES, MAX_RESOURCE_TEXT_CHARS, render_read_resource_result,
    scan_resource_uri,
};
pub use result::{MAX_STRUCTURED_CHARS, RenderedToolResult, render_call_tool_result};
pub use types::{McpResourceInfo, McpResourceTemplateInfo, McpToolInfo, McpToolResult};
