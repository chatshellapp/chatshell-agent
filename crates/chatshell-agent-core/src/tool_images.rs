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

/// Move image blocks out of tool results into the user content that
/// follows them, in place.
///
/// Chat Completions-style wires (OpenAI-compatible endpoints, OpenRouter,
/// Ollama, xAI, Azure, ...) only accept text in `tool` messages, and rig
/// rejects the whole request when a tool result carries an image. Those
/// wires do accept images in `user` messages, and rig splits one user
/// message into its `tool` messages followed by a single `user` message for
/// the remaining content. So each image is appended to the same message,
/// after a label naming the call it came from, and the tool result keeps
/// its text (with a placeholder if no text is left). Returns whether
/// anything moved.
pub fn move_tool_result_images(messages: &mut [rig::message::Message]) -> bool {
    use rig::message::{Message, UserContent};
    let mut moved = false;
    for message in messages.iter_mut() {
        let Message::User { content } = message else {
            continue;
        };
        let mut lifted: Vec<UserContent> = Vec::new();
        for block in content.iter_mut() {
            let UserContent::ToolResult(result) = block else {
                continue;
            };
            if !result
                .content
                .iter()
                .any(|c| matches!(c, ToolResultContent::Image(_)))
            {
                continue;
            }
            let (images, mut rest): (Vec<_>, Vec<_>) = std::mem::take(&mut result.content)
                .into_iter()
                .partition(|c| matches!(c, ToolResultContent::Image(_)));
            if rest.is_empty() {
                rest.push(ToolResultContent::text(IMAGE_PLACEHOLDER));
            }
            result.content = rest;
            let source = if result.name.is_empty() {
                format!("tool call {}", result.wire_call_id())
            } else {
                format!("`{}` (call {})", result.name, result.wire_call_id())
            };
            lifted.push(UserContent::text(format!(
                "[Image output of {source}, attached here because tool results cannot carry images on this provider]"
            )));
            lifted.extend(images.into_iter().filter_map(|c| match c {
                ToolResultContent::Image(image) => Some(UserContent::Image(image)),
                _ => None,
            }));
        }
        if !lifted.is_empty() {
            content.extend(lifted);
            moved = true;
        }
    }
    moved
}

/// Completion model wrapper for wires that reject images in tool results:
/// applies [`move_tool_result_images`] to every request before the inner
/// model converts it.
#[derive(Clone)]
pub struct ToolResultImagesAsUserContent<M> {
    inner: M,
}

impl<M> ToolResultImagesAsUserContent<M> {
    pub fn new(inner: M) -> Self {
        Self { inner }
    }
}

impl<M: rig::completion::CompletionModel> rig::completion::CompletionModel
    for ToolResultImagesAsUserContent<M>
{
    async fn completion(
        &self,
        mut request: rig::completion::CompletionRequest,
    ) -> Result<rig::completion::CompletionResponse, rig::completion::CompletionError> {
        move_tool_result_images(&mut request.chat_history);
        self.inner.completion(request).await
    }

    async fn stream(
        &self,
        mut request: rig::completion::CompletionRequest,
    ) -> Result<rig::streaming::StreamingCompletionResponse, rig::completion::CompletionError> {
        move_tool_result_images(&mut request.chat_history);
        self.inner.stream(request).await
    }

    fn capabilities(&self) -> rig::completion::ProviderCapabilities {
        self.inner.capabilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_result_message(content: Vec<ToolResultContent>) -> rig::message::Message {
        rig::message::Message::User {
            content: vec![rig::message::UserContent::tool_result(
                "tc_1", "shot", content,
            )],
        }
    }

    #[test]
    fn tool_result_images_move_after_the_results() {
        use rig::message::{Message, UserContent};
        let content = tool_result_content("[Image: PNG]\ndata:image/png;base64,AAAA").unwrap();
        let mut messages = vec![tool_result_message(content)];
        assert!(move_tool_result_images(&mut messages));
        let Message::User { content } = &messages[0] else {
            panic!("expected a user message");
        };
        assert_eq!(content.len(), 3, "tool result, label, image");
        let UserContent::ToolResult(result) = &content[0] else {
            panic!("tool result should stay first");
        };
        assert!(
            result
                .content
                .iter()
                .all(|c| matches!(c, ToolResultContent::Text(_)))
        );
        assert!(matches!(&content[1], UserContent::Text(t) if t.text.contains("`shot`")));
        assert!(matches!(content[2], UserContent::Image(_)));
    }

    #[test]
    fn image_only_tool_result_keeps_a_text_placeholder() {
        use rig::message::{Message, UserContent};
        let image = ToolResultContent::image_base64("AAAA", Some(ImageMediaType::PNG), None);
        let mut messages = vec![tool_result_message(vec![image])];
        assert!(move_tool_result_images(&mut messages));
        let Message::User { content } = &messages[0] else {
            panic!("expected a user message");
        };
        let UserContent::ToolResult(result) = &content[0] else {
            panic!("tool result should stay first");
        };
        assert!(
            matches!(&result.content[..], [ToolResultContent::Text(t)] if t.text == IMAGE_PLACEHOLDER)
        );
    }

    #[test]
    fn text_only_tool_results_are_untouched() {
        let mut messages = vec![tool_result_message(vec![ToolResultContent::text("ok")])];
        let before = messages.clone();
        assert!(!move_tool_result_images(&mut messages));
        assert_eq!(messages, before);
    }

    #[test]
    fn moved_images_convert_on_the_chat_completions_wire() {
        use rig::providers::openai::completion::{
            Message as WireMessage, user_content_to_messages,
        };
        let content = tool_result_content("[Image: PNG]\ndata:image/png;base64,AAAA").unwrap();
        let mut messages = vec![tool_result_message(content)];
        move_tool_result_images(&mut messages);
        let rig::message::Message::User { content } = messages.remove(0) else {
            panic!("expected a user message");
        };
        let wire = user_content_to_messages(content).expect("converts without error");
        assert_eq!(wire.len(), 2, "one tool message, then one user message");
        assert!(matches!(wire[0], WireMessage::ToolResult { .. }));
        assert!(matches!(wire[1], WireMessage::User { .. }));
    }

    #[test]
    fn moved_images_convert_on_the_openrouter_wire() {
        // rig keys OpenRouter's request conversion on this shared type.
        use rig::providers::openai::completion::CompletionRequest as OpenrouterCompletionRequest;
        let content = tool_result_content("[Image: PNG]\ndata:image/png;base64,AAAA").unwrap();
        let request = |chat_history| rig::completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history,
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };
        let mut messages = vec![tool_result_message(content)];
        assert!(
            OpenrouterCompletionRequest::try_from(("m", request(messages.clone()))).is_err(),
            "OpenRouter rejects images inside tool results"
        );
        move_tool_result_images(&mut messages);
        OpenrouterCompletionRequest::try_from(("m", request(messages)))
            .expect("converts once the images moved out");
    }

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
