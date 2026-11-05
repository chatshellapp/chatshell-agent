//! Host-resolved file references (`attachment://…`, `tool-output://…`).
//!
//! Some files the model must read live in host-private storage: chat
//! attachments and spilled tool output sit under the app's data or cache
//! directory, which the read policy denies wholesale so the agent cannot
//! browse other conversations. Hosts hand those files to the model as
//! references instead of absolute paths. The read tool resolves a
//! reference through a host-injected [`ReferenceResolver`], which vouches
//! only for files that belong to the current conversation; a vouched path
//! skips the private-directory floor (attaching a file is consent to read
//! it) while every other path keeps the normal policy.
//!
//! References are also stable across devices: persisted tool text and
//! manifests carry `attachment://<name>`, not one device's absolute path.

use std::path::PathBuf;

use futures::future::BoxFuture;

/// Scheme for chat attachments (`attachment://<hash>.<ext>`).
pub const ATTACHMENT_SCHEME: &str = "attachment";
/// Scheme for spilled tool output (`tool-output://<file>`).
pub const TOOL_OUTPUT_SCHEME: &str = "tool-output";

/// Resolves host references to local paths.
pub trait ReferenceResolver: Send + Sync {
    /// Resolve `reference` (the full `scheme://rest` string). `Err` carries
    /// a model-facing explanation (unknown scheme, not part of this
    /// conversation, not downloaded on this device yet).
    fn resolve(&self, reference: &str) -> BoxFuture<'static, Result<PathBuf, String>>;
}

/// The scheme of a host reference, or `None` for plain paths and web URLs.
/// A reference is `scheme://rest` with a lowercase RFC 3986 scheme;
/// `http`, `https`, and `file` are not references.
pub fn reference_scheme(path: &str) -> Option<&str> {
    let (scheme, rest) = path.split_once("://")?;
    let mut chars = scheme.chars();
    let first = chars.next()?;
    if !first.is_ascii_lowercase()
        || !chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "+-.".contains(c))
        || rest.is_empty()
    {
        return None;
    }
    match scheme {
        "http" | "https" | "file" => None,
        _ => Some(scheme),
    }
}

/// Build `scheme://name`.
pub fn reference(scheme: &str, name: &str) -> String {
    format!("{scheme}://{name}")
}

/// The single-component name of a reference (`attachment://x.pdf` →
/// `x.pdf`). Rejects empty names and anything that could walk out of the
/// host's directory (`/`, `\`, `..`).
pub fn reference_name<'a>(reference: &'a str, scheme: &str) -> Option<&'a str> {
    let name = reference.strip_prefix(scheme)?.strip_prefix("://")?.trim();
    let safe = !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains('\0');
    safe.then_some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemes_are_detected_and_web_urls_are_not_references() {
        assert_eq!(reference_scheme("attachment://abc.pdf"), Some("attachment"));
        assert_eq!(reference_scheme("tool-output://x.txt"), Some("tool-output"));
        assert_eq!(reference_scheme("https://example.com"), None);
        assert_eq!(reference_scheme("http://example.com"), None);
        assert_eq!(reference_scheme("file:///etc/hosts"), None);
        assert_eq!(reference_scheme("/tmp/a://b"), None);
        assert_eq!(reference_scheme("Attachment://x"), None);
        assert_eq!(reference_scheme("attachment://"), None);
        assert_eq!(reference_scheme("relative/path.txt"), None);
    }

    #[test]
    fn names_are_single_safe_components() {
        assert_eq!(
            reference_name("attachment://abc.pdf", ATTACHMENT_SCHEME),
            Some("abc.pdf")
        );
        assert_eq!(reference_name("attachment://../x", ATTACHMENT_SCHEME), None);
        assert_eq!(reference_name("attachment://a/b", ATTACHMENT_SCHEME), None);
        assert_eq!(reference_name("attachment://a\\b", ATTACHMENT_SCHEME), None);
        assert_eq!(reference_name("attachment://..", ATTACHMENT_SCHEME), None);
        assert_eq!(reference_name("tool-output://x", ATTACHMENT_SCHEME), None);
        assert_eq!(
            reference("tool-output", "a.txt"),
            "tool-output://a.txt".to_string()
        );
    }
}
