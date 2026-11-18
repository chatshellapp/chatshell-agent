//! Rendering `resources/read` results into model-facing text.
//!
//! Text contents are inlined with a size cap. Binary (blob) contents are
//! never inlined as base64: they are decoded and written to the host's
//! spill store (`tool-output://...` on desktop), and the model is told
//! where the file is, its MIME type, and its size.

use std::path::Path;

use base64::Engine;
use rmcp::model::{ReadResourceResult, ResourceContents};

use crate::path_policy;
use crate::permission::file_uri_path;
use crate::tools::SpillTarget;

/// INV-1 for resource reads: scan the URI like an MCP tool argument, and
/// put a `file://` URI through the read path policy exactly like the
/// `read` tool (deny-tier paths are refused; ask-tier ones were already
/// prompted by the permission gate).
pub fn scan_resource_uri(uri: &str) -> Result<(), String> {
    path_policy::check_blocked_path(uri)?;
    if let Some(path) = file_uri_path(uri) {
        path_policy::check_blocked_path(&path.to_string_lossy())?;
        path_policy::check_read(&path, None)?;
    }
    Ok(())
}

/// Maximum characters of one text content inlined for the model.
pub const MAX_RESOURCE_TEXT_CHARS: usize = 64_000;
/// Largest decoded blob written to the spill store.
pub const MAX_RESOURCE_BLOB_BYTES: usize = 20 * 1024 * 1024;

fn truncate_chars(text: &str, max: usize) -> (&str, bool) {
    if text.len() <= max {
        return (text, false);
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (&text[..end], true)
}

/// File extension for a blob's MIME type (best effort).
fn extension_for(mime: Option<&str>) -> &'static str {
    match mime.unwrap_or_default() {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/svg+xml" => "svg",
        "application/pdf" => "pdf",
        "application/json" => "json",
        "application/zip" => "zip",
        m if m.starts_with("text/") => "txt",
        _ => "bin",
    }
}

fn spill_blob(target: &SpillTarget, bytes: &[u8], mime: Option<&str>) -> std::io::Result<String> {
    std::fs::create_dir_all(&target.dir)?;
    let name = format!(
        "mcp-resource-{}.{}",
        uuid::Uuid::now_v7(),
        extension_for(mime)
    );
    let path = target.dir.join(&name);
    std::fs::write(&path, bytes)?;
    Ok(match &target.reference_scheme {
        Some(scheme) => crate::tools::reference::reference(scheme, &name),
        None => display_path(&path),
    })
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}

fn render_blob(uri: &str, mime: Option<&str>, blob: &str, spill: Option<&SpillTarget>) -> String {
    let mime_label = mime.unwrap_or("application/octet-stream");
    let bytes = match base64::engine::general_purpose::STANDARD.decode(blob.trim()) {
        Ok(bytes) => bytes,
        Err(_) => {
            return format!("[resource {uri} ({mime_label}): invalid base64 blob, not shown]");
        }
    };
    let size = bytes.len();
    if size > MAX_RESOURCE_BLOB_BYTES {
        return format!(
            "[resource {uri} ({mime_label}, {size} bytes): larger than the {MAX_RESOURCE_BLOB_BYTES}-byte limit, not saved]"
        );
    }
    match spill {
        Some(target) => match spill_blob(target, &bytes, mime) {
            Ok(location) => {
                format!("[resource {uri} ({mime_label}, {size} bytes) saved to {location}]")
            }
            Err(e) => {
                tracing::warn!("Could not save MCP resource blob: {e}");
                format!("[resource {uri} ({mime_label}, {size} bytes): could not be saved: {e}]")
            }
        },
        None => format!(
            "[resource {uri} ({mime_label}, {size} bytes): binary content not shown (no file store configured)]"
        ),
    }
}

/// Render every content of a read result. `spill` receives blobs.
pub fn render_read_resource_result(
    result: &ReadResourceResult,
    spill: Option<&SpillTarget>,
) -> String {
    let parts: Vec<String> = result
        .contents
        .iter()
        .map(|content| match content {
            ResourceContents::TextResourceContents {
                uri,
                mime_type,
                text,
                ..
            } => {
                let (shown, truncated) = truncate_chars(text, MAX_RESOURCE_TEXT_CHARS);
                let mime = mime_type.as_deref().unwrap_or("text/plain");
                let note = if truncated {
                    format!(
                        " truncated_at_chars=\"{MAX_RESOURCE_TEXT_CHARS}\" total_chars=\"{}\"",
                        text.len()
                    )
                } else {
                    String::new()
                };
                format!("<resource uri=\"{uri}\" mime=\"{mime}\"{note}>\n{shown}\n</resource>")
            }
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } => render_blob(uri, mime_type.as_deref(), blob, spill),
            _ => "[resource content of an unsupported kind, not shown]".to_string(),
        })
        .collect();
    if parts.is_empty() {
        "[the server returned no content for this resource]".to_string()
    } else {
        parts.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn result(value: serde_json::Value) -> ReadResourceResult {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn plain_uris_pass_the_scan() {
        assert!(scan_resource_uri("mem://notes/today").is_ok());
        assert!(scan_resource_uri("https://example.com/doc").is_ok());
    }

    #[test]
    fn denied_file_uri_and_blocked_paths_fail_the_scan() {
        let home = dirs::home_dir().unwrap();
        let uri = url::Url::from_file_path(home.join(".ssh/id_rsa"))
            .unwrap()
            .to_string();
        assert!(scan_resource_uri(&uri).is_err());
        assert!(scan_resource_uri("~/.ssh/id_rsa").is_err());
    }

    #[test]
    fn text_is_inlined_with_uri_and_mime() {
        let out = render_read_resource_result(
            &result(json!({"contents": [
                {"uri": "file:///a.md", "mimeType": "text/markdown", "text": "# hi"}
            ]})),
            None,
        );
        assert_eq!(
            out,
            "<resource uri=\"file:///a.md\" mime=\"text/markdown\">\n# hi\n</resource>"
        );
    }

    #[test]
    fn long_text_is_truncated() {
        let text = "y".repeat(MAX_RESOURCE_TEXT_CHARS + 10);
        let out = render_read_resource_result(
            &result(json!({"contents": [{"uri": "x://a", "text": text}]})),
            None,
        );
        assert!(out.contains("truncated_at_chars"));
        assert!(out.len() < MAX_RESOURCE_TEXT_CHARS + 300);
    }

    #[test]
    fn blob_is_spilled_never_inlined() {
        let dir = tempfile::tempdir().unwrap();
        let target = SpillTarget {
            dir: dir.path().to_path_buf(),
            reference_scheme: Some("tool-output".to_string()),
        };
        let png = base64::engine::general_purpose::STANDARD.encode([0x89, b'P', b'N', b'G']);
        let out = render_read_resource_result(
            &result(json!({"contents": [
                {"uri": "x://img", "mimeType": "image/png", "blob": png}
            ]})),
            Some(&target),
        );
        assert!(out.contains("(image/png, 4 bytes) saved to tool-output://mcp-resource-"));
        assert!(out.contains(".png]"));
        assert!(!out.contains(&png));
        let files: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(files.len(), 1);
        let written = std::fs::read(files[0].as_ref().unwrap().path()).unwrap();
        assert_eq!(written, [0x89, b'P', b'N', b'G']);
    }

    #[test]
    fn blob_without_store_is_described_only() {
        let out = render_read_resource_result(
            &result(json!({"contents": [{"uri": "x://b", "blob": "AAAA"}]})),
            None,
        );
        assert!(out.contains("application/octet-stream, 3 bytes"));
        assert!(out.contains("not shown"));
    }

    #[test]
    fn invalid_blob_is_reported() {
        let out = render_read_resource_result(
            &result(json!({"contents": [{"uri": "x://b", "blob": "!!not base64!!"}]})),
            None,
        );
        assert!(out.contains("invalid base64"));
    }

    #[test]
    fn empty_contents() {
        let out = render_read_resource_result(&result(json!({"contents": []})), None);
        assert!(out.contains("no content"));
    }
}
