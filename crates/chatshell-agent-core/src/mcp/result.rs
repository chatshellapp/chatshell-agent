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

/// Render an image block as a `data:` line the hosts lift into an image
/// block. Raster images are decoded and normalized first
/// ([`crate::image_fit::fit_untrusted_image`]): a defective image passes
/// our decoders but fails the whole provider request, and an oversized one
/// exceeds provider limits. An image that cannot be decoded becomes a text
/// note instead. SVG stays as is (the model reads it as source).
fn render_image(mime_type: &str, data: &str) -> String {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    if mime_type.eq_ignore_ascii_case("image/svg+xml") {
        return format!("data:{mime_type};base64,{data}");
    }
    let compact: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let fitted = STANDARD
        .decode(compact.as_bytes())
        .map_err(|e| format!("invalid base64: {e}"))
        .and_then(|bytes| crate::image_fit::fit_untrusted_image(&bytes));
    match fitted {
        Ok(image) => format!(
            "data:{};base64,{}",
            image.mime_type,
            STANDARD.encode(&image.bytes)
        ),
        Err(reason) => format!(
            "[The tool returned an image ({mime_type}) that could not be decoded: {reason}]"
        ),
    }
}

fn render_block(block: &ContentBlock) -> Option<String> {
    match block {
        ContentBlock::Text(raw) => Some(raw.text.clone()),
        ContentBlock::Image(raw) => Some(render_image(&raw.mime_type, &raw.data)),
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
/// as parsed JSON so whitespace and key order do not matter; any JSON value
/// is allowed (2026-07-28), and a string value also matches its bare text.
fn text_mirrors_structured(content: &[ContentBlock], structured: &serde_json::Value) -> bool {
    content.iter().any(|block| match block {
        ContentBlock::Text(raw) => {
            let text = raw.text.trim();
            // A string value is usually echoed bare, not as a JSON literal.
            structured
                .as_str()
                .is_some_and(|value| value.trim() == text)
                || serde_json::from_str::<serde_json::Value>(text)
                    .is_ok_and(|parsed| &parsed == structured)
        }
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
    fn non_object_structured_values_are_matched() {
        for (text, structured) in [
            ("[1, 2]", json!([1, 2])),
            ("42", json!(42)),
            ("\"sunny\"", json!("sunny")),
            ("sunny", json!("sunny")),
        ] {
            let r = render_call_tool_result(&result(json!({
                "content": [{"type": "text", "text": text}],
                "structuredContent": structured
            })));
            assert_eq!(r.text, text, "{structured}");
        }
        let r = render_call_tool_result(&result(json!({"structuredContent": [1, 2]})));
        assert!(r.text.contains("<structured_content>"));
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

    fn png_base64(width: u32, height: u32) -> String {
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        let image = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            width,
            height,
            image::Rgba([10, 20, 30, 255]),
        ));
        let mut out = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        STANDARD.encode(out)
    }

    #[test]
    fn image_and_resource_blocks() {
        let r = render_call_tool_result(&result(json!({
            "content": [
                {"type": "image", "data": png_base64(2, 2), "mimeType": "image/png"},
                {"type": "resource", "resource": {"uri": "file:///a.txt", "mimeType": "text/plain", "text": "hi"}}
            ]
        })));
        let (image_line, resource_line) = r.text.split_once('\n').unwrap();
        assert!(image_line.starts_with("data:image/png;base64,"));
        assert!(crate::tool_images::split_image_lines(image_line).is_some());
        assert_eq!(resource_line, "data:text/plain;file:///a.txt:hi");
    }

    #[test]
    fn undecodable_images_become_a_note() {
        let r = render_call_tool_result(&result(json!({
            "content": [{"type": "image", "data": "AAA", "mimeType": "image/png"}]
        })));
        assert!(!r.text.contains("data:image/"), "{}", r.text);
        assert!(r.text.contains("could not be decoded"));
    }

    /// PNG chunk CRC (CRC-32/ISO-HDLC), bitwise.
    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    /// The MCP reference server's `getTinyImage` PNG has a bad `iCCP` CRC;
    /// strict providers (xAI) reject it. Set `MCP_TINY_IMAGE_B64` to a file
    /// holding its base64 to check the rendered image is clean.
    #[test]
    fn reference_server_tiny_image_is_rendered_clean() {
        let Ok(path) = std::env::var("MCP_TINY_IMAGE_B64") else {
            return;
        };
        let data = std::fs::read_to_string(path).unwrap();
        let r = render_call_tool_result(&result(json!({
            "content": [{"type": "image", "data": data.trim(), "mimeType": "image/png"}]
        })));
        let (_, images) = crate::tool_images::split_image_lines(&r.text).unwrap();
        use base64::{Engine as _, engine::general_purpose::STANDARD};
        let bytes = STANDARD.decode(&images[0].base64).unwrap();
        // Every chunk CRC must verify.
        let mut i = 8;
        while i < bytes.len() {
            let len = u32::from_be_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
            let crc = u32::from_be_bytes(bytes[i + 8 + len..i + 12 + len].try_into().unwrap());
            assert_eq!(crc32(&bytes[i + 4..i + 8 + len]), crc, "chunk at {i}");
            i += 12 + len;
        }
    }
}
