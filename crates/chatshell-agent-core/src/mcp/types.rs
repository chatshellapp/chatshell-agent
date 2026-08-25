use serde::{Deserialize, Serialize};

/// A tool exposed by an MCP server. Mirrors `rmcp::model::Tool` but with
/// FFI-friendly owned types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema describing the tool's parameters, as a JSON string.
    pub input_schema_json: String,
}

/// Result of calling a tool on an MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolResult {
    /// Text content returned by the tool. Non-text content (images, etc.)
    /// is currently skipped - only text is surfaced to the LLM.
    pub content: String,
    pub is_error: bool,
}
