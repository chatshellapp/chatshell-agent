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
mod handler;
mod listing;
mod resource;
mod result;
mod types;

pub use call::{
    DEFAULT_CALL_TIMEOUT_SECS, DEFAULT_MAX_TOTAL_TIMEOUT_SECS, McpCallOptions,
    call_tool_with_options, read_resource_with_options,
};
pub use connection::{McpClientService, McpConnection, McpHttpConfig, connect_http_service};
pub use handler::{
    ChatShellClientHandler, LIST_CHANGED_DEBOUNCE, McpNotificationSink, chatshell_client_info,
};
pub use listing::{
    ListTruncation, MAX_CURSOR_BYTES, MAX_LIST_ITEMS, MAX_LIST_PAGES, Paginated,
    list_all_resource_templates_bounded, list_all_resources_bounded, list_all_tools_bounded,
    paginate, supports_resources,
};
pub use resource::{MAX_RESOURCE_BLOB_BYTES, MAX_RESOURCE_TEXT_CHARS, render_read_resource_result};
pub use result::{MAX_STRUCTURED_CHARS, RenderedToolResult, render_call_tool_result};
pub use types::{McpToolInfo, McpToolResult};
