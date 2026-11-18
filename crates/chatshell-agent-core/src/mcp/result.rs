//! Rendering `tools/call` results into model-facing text.
//!
//! Shared by desktop (`mcp_tool_use`) and iOS (`McpConnection::call_tool`) so
//! both platforms surface the same content for the same result.

use rmcp::model::{CallToolResult, ContentBlock, ResourceContents};

/// Maximum characters of the rendered `structuredContent` JSON block. The
/// payload comes from a server we do not control.
pub const MAX_STRUCTURED_CHARS: usize = 32_000;

/// A rendered tool result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedToolResult {
    pub text: String,
    pub is_error: bool,
}

fn render_block(block: &ContentBlock) -> Option<String> {
    match block {
        ContentBlock::Text(raw) => Some(raw.text.clone()),
        ContentBlock::Image(raw) => Some(format!("data:{};base64,{}", raw.mime_type, raw.data)),
        ContentBlock::Audio(raw) => Some(format!("data:{};base64,{}", raw.mime_type, raw.data)),
        ContentBlock::Resource(raw) => Some(match &raw.resource {
            ResourceContents::TextResourceContents {
                uri,
                mime_type,
                text,
                ..
            } => format!(
                "{}{uri}:{text}",
                mime_type
                    .as_ref()
                    .map(|m| format!("data:{m};"))
                    .unwrap_or_default(),
            ),
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } => format!(
                "{}{uri}:{blob}",
                mime_type
                    .as_ref()
                    .map(|m| format!("data:{m};"))
                    .unwrap_or_default(),
            ),
            _ => String::new(),
        }),
        ContentBlock::ResourceLink(link) => Some(format!("[resource link] {}", link.uri)),
        _ => None,
    }
}

/// Whether some text block already carries `structured` (the spec asks
/// servers to mirror structured output as serialized JSON text). Compared
/// as parsed JSON so whitespace and key order do not matter.
fn text_mirrors_structured(content: &[ContentBlock], structured: &serde_json::Value) -> bool {
    content.iter().any(|block| match block {
        ContentBlock::Text(raw) => serde_json::from_str::<serde_json::Value>(raw.text.trim())
            .is_ok_and(|parsed| &parsed == structured),
        _ => false,
    })
}

fn truncate_chars(mut s: String, max: usize) -> (String, bool) {
    if s.len() <= max {
        return (s, false);
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    (s, true)
}

fn render_structured(structured: &serde_json::Value) -> String {
    let json = serde_json::to_string_pretty(structured).unwrap_or_else(|_| structured.to_string());
    let (json, truncated) = truncate_chars(json, MAX_STRUCTURED_CHARS);
    if truncated {
        format!(
            "<structured_content truncated_at_chars=\"{MAX_STRUCTURED_CHARS}\">\n{json}\n</structured_content>"
        )
    } else {
        format!("<structured_content>\n{json}\n</structured_content>")
    }
}

/// Render a `CallToolResult`: every content block in order, then the
/// `structuredContent` JSON block when no text block already mirrors it.
/// `is_error` is always preserved.
pub fn render_call_tool_result(result: &CallToolResult) -> RenderedToolResult {
    let mut parts: Vec<String> = result.content.iter().filter_map(render_block).collect();
    if let Some(structured) = &result.structured_content
        && !text_mirrors_structured(&result.content, structured)
    {
        parts.push(render_structured(structured));
    }
    RenderedToolResult {
        text: parts.join("\n"),
        is_error: result.is_error.unwrap_or(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn result(value: serde_json::Value) -> CallToolResult {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn text_only() {
        let r = render_call_tool_result(&result(json!({
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        })));
        assert_eq!(r.text, "a\nb");
        assert!(!r.is_error);
    }

    #[test]
    fn structured_only_renders_json_block() {
        let r = render_call_tool_result(&result(json!({
            "content": [],
            "structuredContent": {"temperature": 22.5}
        })));
        assert!(r.text.starts_with("<structured_content>"));
        assert!(r.text.contains("\"temperature\": 22.5"));
    }

    #[test]
    fn structured_without_content_field() {
        let r = render_call_tool_result(&result(json!({
            "structuredContent": {"ok": true}
        })));
        assert!(r.text.contains("\"ok\": true"));
    }

    #[test]
    fn mirrored_structured_is_not_duplicated() {
        let r = render_call_tool_result(&result(json!({
            "content": [{"type": "text", "text": "{ \"b\": 2,\n \"a\": 1 }"}],
            "structuredContent": {"a": 1, "b": 2}
        })));
        assert_eq!(r.text, "{ \"b\": 2,\n \"a\": 1 }");
    }

    #[test]
    fn divergent_structured_is_appended() {
        let r = render_call_tool_result(&result(json!({
            "content": [{"type": "text", "text": "Weather fetched"}],
            "structuredContent": {"temperature": 22.5}
        })));
        assert!(r.text.starts_with("Weather fetched\n<structured_content>"));
    }

    #[test]
    fn is_error_is_preserved_with_structured() {
        let r = render_call_tool_result(&result(json!({
            "content": [],
            "structuredContent": {"error": "bad input"},
            "isError": true
        })));
        assert!(r.is_error);
        assert!(r.text.contains("bad input"));
    }

    #[test]
    fn oversized_structured_is_truncated() {
        let big = "x".repeat(MAX_STRUCTURED_CHARS * 2);
        let r = render_call_tool_result(&result(json!({
            "structuredContent": {"blob": big}
        })));
        assert!(r.text.contains("truncated_at_chars"));
        assert!(r.text.len() < MAX_STRUCTURED_CHARS + 200);
    }

    #[test]
    fn image_and_resource_blocks() {
        let r = render_call_tool_result(&result(json!({
            "content": [
                {"type": "image", "data": "AAA", "mimeType": "image/png"},
                {"type": "resource", "resource": {"uri": "file:///a.txt", "mimeType": "text/plain", "text": "hi"}}
            ]
        })));
        assert_eq!(
            r.text,
            "data:image/png;base64,AAA\ndata:text/plain;file:///a.txt:hi"
        );
    }
}
