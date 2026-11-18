use serde::{Deserialize, Serialize};

/// A tool exposed by an MCP server. Mirrors `rmcp::model::Tool` but with
/// FFI-friendly owned types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema describing the tool's parameters, as a JSON string.
    pub input_schema_json: String,
    /// JSON Schema of the tool's `structuredContent`, if declared.
    pub output_schema_json: Option<String>,
    /// Tool annotations (`title`, `readOnlyHint`, ...) as a JSON string.
    /// Server-supplied and untrusted: descriptive only.
    pub annotations_json: Option<String>,
    /// `readOnlyHint: true` was declared.
    pub read_only_hint: bool,
    /// `destructiveHint: true` was declared (explicitly; the spec default
    /// is not applied).
    pub destructive_hint: bool,
}

impl McpToolInfo {
    pub fn from_tool(t: &rmcp::model::Tool) -> Self {
        Self {
            name: t.name.to_string(),
            description: t.description.as_ref().map(|d| d.to_string()),
            input_schema_json: serde_json::to_string(&t.input_schema)
                .unwrap_or_else(|_| "{}".to_string()),
            output_schema_json: t
                .output_schema
                .as_ref()
                .and_then(|s| serde_json::to_string(s).ok()),
            annotations_json: t
                .annotations
                .as_ref()
                .and_then(|a| serde_json::to_string(a).ok()),
            read_only_hint: t
                .annotations
                .as_ref()
                .is_some_and(|a| a.read_only_hint == Some(true)),
            destructive_hint: t
                .annotations
                .as_ref()
                .is_some_and(|a| a.destructive_hint == Some(true)),
        }
    }
}

/// A resource listed by an MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpResourceInfo {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

/// A resource template (RFC 6570 URI template) listed by an MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpResourceTemplateInfo {
    pub uri_template: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

/// Result of calling a tool on an MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolResult {
    /// Rendered tool output: content blocks (text, data URIs for images and
    /// blobs) followed by the `structuredContent` JSON block when no text
    /// block already mirrors it.
    pub content: String,
    pub is_error: bool,
}
