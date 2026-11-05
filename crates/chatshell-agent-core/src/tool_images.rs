//! Image payloads inside text tool results.
//!
//! File tools return `String`, and images ride inside it as a line of the
//! form `data:image/<type>;base64,<payload>` (the read tool's image and PDF
//! page paths). Sent as-is, that line reaches the provider as base64 text:
//! the model cannot see the image and pays for every character. Hosts turn
//! such results into multimodal tool output: the text without the payload
//! lines, followed by one image block per line.

use rig::message::{ImageMediaType, MimeType, ToolResultContent};

/// One image lifted out of a tool result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolImage {
    pub mime_type: String,
    pub base64: String,
}

/// Placeholder left in the text where an image line was.
pub const IMAGE_PLACEHOLDER: &str = "[image attached below]";

/// Split `text` into its non-image text and the image payload lines.
/// Returns `None` when there is no image line (keep the text as is).
pub fn split_image_lines(text: &str) -> Option<(String, Vec<ToolImage>)> {
    if !text.contains("data:image/") {
        return None;
    }
    let mut images = Vec::new();
    let mut kept: Vec<&str> = Vec::new();
    for line in text.lines() {
        match parse_image_line(line.trim()) {
            Some(image) => {
                images.push(image);
                kept.push(IMAGE_PLACEHOLDER);
            }
            None => kept.push(line),
        }
    }
    (!images.is_empty()).then(|| (kept.join("\n"), images))
}

fn parse_image_line(line: &str) -> Option<ToolImage> {
    let rest = line.strip_prefix("data:")?;
    let (mime, payload) = rest.split_once(";base64,")?;
    let mime = mime.to_ascii_lowercase();
    // SVG is text the model reads as source; only raster images split.
    let raster = matches!(
        mime.as_str(),
        "image/png" | "image/jpeg" | "image/gif" | "image/webp"
    );
    let base64_ok = !payload.is_empty()
        && payload
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'=');
    (raster && base64_ok).then(|| ToolImage {
        mime_type: mime,
        base64: payload.to_string(),
    })
}

/// Multimodal tool-result content for `text`: `None` when it carries no
/// image line.
pub fn tool_result_content(text: &str) -> Option<Vec<ToolResultContent>> {
    let (text, images) = split_image_lines(text)?;
    let mut content = vec![ToolResultContent::text(text)];
    for image in images {
        content.push(ToolResultContent::image_base64(
            image.base64,
            ImageMediaType::from_mime_type(&image.mime_type),
            None,
        ));
    }
    Some(content)
}

/// Inverse of [`tool_result_content`]: the text form a host persists.
/// Text blocks are joined; each base64 image fills the next placeholder
/// (or is appended) as a `data:` line, so replay restores the same blocks.
pub fn tool_result_text(content: &[ToolResultContent]) -> String {
    use rig::message::DocumentSourceKind;
    let mut text: Vec<String> = Vec::new();
    let mut images: Vec<String> = Vec::new();
    for block in content {
        match block {
            ToolResultContent::Text(t) => text.push(t.text.clone()),
            ToolResultContent::Image(image) => {
                if let DocumentSourceKind::Base64(data) = &image.data {
                    let mime = image
                        .media_type
                        .as_ref()
                        .map(|m| m.to_mime_type())
                        .unwrap_or("image/png");
                    images.push(format!("data:{mime};base64,{data}"));
                }
            }
            _ => {}
        }
    }
    let mut images = images.into_iter();
    let mut lines: Vec<String> = text
        .join("\n")
        .lines()
        .map(|line| {
            if line == IMAGE_PLACEHOLDER {
                images.next().unwrap_or_else(|| line.to_string())
            } else {
                line.to_string()
            }
        })
        .collect();
    lines.extend(images);
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_lines_split_out_and_leave_a_placeholder() {
        let text = "[Image: PNG, 2x2, 70 B]\ndata:image/png;base64,iVBORw0KGgo=\nafter";
        let (kept, images) = split_image_lines(text).unwrap();
        assert_eq!(
            kept,
            "[Image: PNG, 2x2, 70 B]\n[image attached below]\nafter"
        );
        assert_eq!(
            images,
            vec![ToolImage {
                mime_type: "image/png".to_string(),
                base64: "iVBORw0KGgo=".to_string(),
            }]
        );
    }

    #[test]
    fn plain_text_svg_and_inline_mentions_are_left_alone() {
        assert!(split_image_lines("no images here").is_none());
        assert!(split_image_lines("data:image/svg+xml;base64,PHN2Zy8+").is_none());
        // Not a whole line: a markdown image inside page text.
        assert!(split_image_lines("![x](data:image/png;base64,AAAA)").is_none());
        assert!(split_image_lines("data:image/png;base64,not base64!").is_none());
    }

    #[test]
    fn persisted_text_round_trips_through_content() {
        let text = "[Image: PNG]\ndata:image/png;base64,AAAA\nmiddle\ndata:image/jpeg;base64,BBBB";
        let content = tool_result_content(text).unwrap();
        assert_eq!(tool_result_text(&content), text);
        let plain = vec![ToolResultContent::text("just text")];
        assert_eq!(tool_result_text(&plain), "just text");
    }

    #[test]
    fn content_has_text_first_then_one_block_per_image() {
        let text = "pages\ndata:image/jpeg;base64,AAAA\ndata:image/png;base64,BBBB";
        let content = tool_result_content(text).unwrap();
        assert_eq!(content.len(), 3);
        assert!(matches!(content[0], ToolResultContent::Text(_)));
        assert!(matches!(content[1], ToolResultContent::Image(_)));
        assert!(matches!(content[2], ToolResultContent::Image(_)));
    }
}
