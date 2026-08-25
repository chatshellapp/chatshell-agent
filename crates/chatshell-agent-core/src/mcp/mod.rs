//! MCP (Model Context Protocol) HTTP transport - shared across desktop and iOS.
//!
//! Extracted from desktop's `src-tauri/src/mcp/manager.rs`. Only the HTTP
//! (Streamable HTTP) transport is shared; STDIO transport stays desktop-only
//! (iOS cannot spawn child processes). OAuth flow stays platform-specific
//! (desktop uses system browser + callback server, iOS uses ASWebAuthenticationSession).
//!
//! The shared module handles: HTTP connection, tool listing, tool calling.
//! Auth token (bearer) is passed in via `McpHttpConfig` - the caller is
//! responsible for obtaining it (plaintext, keychain, OAuth, etc.).

mod connection;
mod types;

pub use connection::{McpConnection, McpHttpConfig};
pub use types::{McpToolInfo, McpToolResult};
