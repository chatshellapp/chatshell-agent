//! Edit tool for LLM agents
//!
//! Precise string replacement with upstream-style anchor safety:
//! - content tags minted by read/write (`[path#TAG]` headers) can be passed
//!   as `tag` to detect stale content before editing (upstream snapshot-store
//!   semantics; a stale tag recovers via exact match or demands a re-read);
//! - exact match first, whitespace/indentation-tolerant fuzzy fallback
//!   through the shared core matcher;
//! - byte-identical no-op edits get a soft hint, escalating to a hard error
//!   when the same payload repeats (upstream noop-loop-guard);
//! - ambiguous matches report numbered preview windows;
//! - every successful edit returns a fresh tag plus a numbered context
//!   window so the next edit re-grounds without a fresh read;
//! - a missing path resolves by unique workspace suffix when a project root
//!   is configured;
//! - edits are exclusive: one read-modify-write at a time.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use rig::agent::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::anchor_store as anchor_state;
use crate::edit_anchor as anchor;
use crate::fuzzy_match;
use crate::path_policy;
use crate::tool_fs::{Fs, StdFs, WalkFilters};

/// Context lines around the replaced region in the re-grounding window.
const REGROUND_CONTEXT_LINES: usize = 3;
/// Line clip for re-grounding windows.
const REGROUND_MAX_LINE_CHARS: usize = 200;
/// Context lines around each ambiguity preview window (5-line windows).
const PREVIEW_CONTEXT_LINES: usize = 2;
/// Line clip for preview windows (upstream: 80-char cap).
const PREVIEW_MAX_LINE_CHARS: usize = 80;
/// Max preview windows rendered (upstream: 5).
const MAX_PREVIEW_MATCHES: usize = 5;
/// Suffix-resolution walk limits.
const SUFFIX_WALK_MAX_DEPTH: usize = 20;
const SUFFIX_WALK_MAX_FILES: usize = 10_000;
const SUFFIX_MAX_CANDIDATES: usize = 5;

#[derive(Debug, Clone, Deserialize)]
pub struct EditArgs {
    /// Absolute path to the file to edit
    pub path: String,
    /// The exact text to find and replace (must be unique unless replace_all is true)
    pub old_string: String,
    /// The replacement text (must differ from old_string)
    pub new_string: String,
    /// Replace all occurrences instead of requiring uniqueness (default: false)
    #[serde(default)]
    pub replace_all: Option<bool>,
    /// Optional content tag from the latest `[path#TAG]` read/write/edit
    /// header for this file; detects stale anchors
    #[serde(default)]
    pub tag: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("Edit error: {0}")]
pub struct EditError(String);

fn default_fs() -> Arc<dyn Fs> {
    Arc::new(StdFs)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EditTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    project_root: Option<PathBuf>,
    #[serde(skip, default = "default_fs")]
    fs: Arc<dyn Fs>,
}

impl Default for EditTool {
    fn default() -> Self {
        Self::new()
    }
}

impl EditTool {
    pub fn new() -> Self {
        Self {
            project_root: None,
            fs: Arc::new(StdFs),
        }
    }

    pub fn with_project_root(root: PathBuf) -> Self {
        Self {
            project_root: Some(root),
            ..Self::new()
        }
    }

    /// Route IO through a custom [`Fs`] implementation.
    pub fn with_fs(mut self, fs: Arc<dyn Fs>) -> Self {
        self.fs = fs;
        self
    }
}

/// Outcome of locating the requested file, with upstream-style path tolerance.
enum Located {
    Found(PathBuf),
    /// The path was missing but exactly one workspace file matches its
    /// trailing components.
    Resolved(PathBuf),
    NotFound,
    /// Multiple workspace files match; listing is attached to the error.
    Ambiguous(Vec<PathBuf>),
}

fn normal_components(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(os) => Some(os.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect()
}

/// upstream path tolerance (`findUniqueWorkspaceSuffix`): when the requested path
/// does not exist but a project root is configured, resolve the file whose
/// relative path uniquely ends with the request's trailing components (the
/// model wrote `src/foo.rs`, the actual file is `packages/x/src/foo.rs`).
fn locate_file(fs: &dyn Fs, requested: &Path, root: Option<&Path>) -> Located {
    if fs.exists(requested) {
        return Located::Found(requested.to_path_buf());
    }
    let root = root.filter(|root| fs.metadata(root).is_ok_and(|meta| meta.is_dir));
    let Some(root) = root else {
        return Located::NotFound;
    };
    let suffix = normal_components(requested);
    if suffix.is_empty() {
        return Located::NotFound;
    }

    // One gitignore-aware walk, same settings as the glob tool.
    let mut files: Vec<Vec<String>> = Vec::new();
    let walked = fs
        .walk(
            root,
            &WalkFilters {
                max_depth: Some(SUFFIX_WALK_MAX_DEPTH),
                max_files: Some(SUFFIX_WALK_MAX_FILES),
                ..Default::default()
            },
        )
        .unwrap_or_default();
    for path in walked {
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let components = normal_components(relative);
        if !components.is_empty() {
            files.push(components);
        }
        if files.len() >= SUFFIX_WALK_MAX_FILES {
            break;
        }
    }

    // Longest suffix first: the more specific pattern wins.
    for length in (1..=suffix.len()).rev() {
        let pattern = &suffix[suffix.len() - length..];
        let mut matches: Vec<PathBuf> = files
            .iter()
            .filter(|components| {
                components.len() >= length && components[components.len() - length..] == *pattern
            })
            .map(|components| root.join(components.iter().collect::<PathBuf>()))
            .collect();
        if matches.len() == 1 {
            return Located::Resolved(matches.remove(0));
        }
        if matches.len() > 1 {
            matches.truncate(SUFFIX_MAX_CANDIDATES);
            return Located::Ambiguous(matches);
        }
    }
    Located::NotFound
}

/// Warnings appended to a successful edit when anchor state warrants them.
fn anchor_warnings(
    path_note: Option<&str>,
    tag_stale: bool,
    drifted_since_served: bool,
    given_tag: Option<&str>,
    current_tag: &str,
) -> String {
    let mut warnings = String::new();
    if let Some(note) = path_note {
        warnings.push_str(note);
    }
    if tag_stale {
        warnings.push_str(&format!(
            " NOTE: content tag {} is stale (current {}) - the file changed since that read; \
             applied by exact content match and re-anchored below.",
            given_tag.unwrap_or("?"),
            current_tag
        ));
    } else if drifted_since_served {
        warnings.push_str(
            " NOTE: the file changed since it was last read/written by this app; \
             anchored on the current content below.",
        );
    }
    warnings
}

/// The re-grounding section returned with every successful edit: fresh
/// `[path#TAG]` header plus a numbered context window around the edit.
fn reground_section(
    path_display: &str,
    new_content: &str,
    new_tag: Option<String>,
    start_line: usize,
    region_lines: usize,
) -> String {
    let mut section = String::new();
    if let Some(tag) = new_tag {
        section.push_str(&anchor::tag_header(path_display, &tag));
        section.push('\n');
    }
    let end_line = start_line + region_lines.saturating_sub(1);
    section.push_str(&anchor::render_context_window(
        new_content,
        start_line,
        end_line,
        REGROUND_CONTEXT_LINES,
        REGROUND_MAX_LINE_CHARS,
    ));
    section
}

impl Tool for EditTool {
    const NAME: &'static str = "edit";

    type Error = EditError;
    type Args = EditArgs;
    type Output = String;

    fn description(&self) -> String {
        "Make precise text replacements in a file. \
                Specify the exact text to find (old_string) and its replacement (new_string). \
                The old_string must match the file content exactly, including whitespace and indentation; \
                when no exact match exists, a whitespace/indentation-tolerant fuzzy match is attempted \
                (reported with its similarity when it succeeds). \
                By default old_string must appear exactly once in the file; use replace_all to change every occurrence. \
                Pass the tag from the latest [path#TAG] header for this file (read/write/edit output) as `tag` \
                to detect stale content: a fresh tag validates the file is unchanged, a stale one triggers \
                recovery or a re-read demand. \
                The response includes a fresh [path#TAG] header plus the edited region with surrounding \
                line numbers - anchor follow-up edits on it. \
                Always use the `read` tool first to see the current file content before editing."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find in the file. Must match including whitespace and indentation."
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace old_string with. Must differ from old_string."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace all occurrences. If false (default), old_string must appear exactly once."
                },
                "tag": {
                    "type": "string",
                    "description": "Optional content tag from the latest [path#TAG] header for this file. Detects stale anchors."
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::agent::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let replace_all = args.replace_all.unwrap_or(false);

        tracing::info!(
            "🔧 [tool-call] edit: path=\"{}\" old_len={} new_len={} replace_all={} tag={:?}",
            args.path,
            args.old_string.len(),
            args.new_string.len(),
            replace_all,
            args.tag
        );

        if args.old_string == args.new_string {
            return Err(EditError(
                "old_string and new_string are identical; nothing to change".into(),
            ));
        }

        // One edit at a time (upstream: edit concurrency = "exclusive"). The whole
        // read-modify-write below is synchronous, so a std mutex held across
        // it is safe.
        let _exclusive = anchor_state::EDIT_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let (path, path_note) = match locate_file(
            self.fs.as_ref(),
            Path::new(&args.path),
            self.project_root.as_deref(),
        ) {
            Located::Found(path) => (path, None),
            Located::Resolved(path) => {
                let note = format!(
                    " (resolved missing {} by unique workspace suffix)",
                    args.path
                );
                (path, Some(note))
            }
            Located::NotFound => {
                return Err(EditError(format!("File not found: {}", args.path)));
            }
            Located::Ambiguous(candidates) => {
                let listing = candidates
                    .iter()
                    .map(|candidate| format!("  {}", candidate.display()))
                    .collect::<Vec<_>>()
                    .join("\n");
                return Err(EditError(format!(
                    "File not found: {}. Possible matches under the workspace root:\n{}",
                    args.path, listing
                )));
            }
        };

        // The write policy gates the final target; a suffix-resolved path is
        // under the workspace root and must pass like any other.
        path_policy::check_write(&path, self.project_root.as_deref()).map_err(EditError)?;
        if !self.fs.metadata(&path).is_ok_and(|meta| meta.is_file) {
            return Err(EditError(format!("Not a file: {}", path.display())));
        }

        let content = String::from_utf8(
            self.fs
                .read(&path)
                .map_err(|e| EditError(format!("Failed to read file: {}", e)))?,
        )
        .map_err(|e| EditError(format!("File is not valid UTF-8: {}", e)))?;

        // Anchor validation (upstream snapshot-tag semantics): the tag hashes the
        // live content, so a mismatch means the file changed since the read
        // that minted the tag.
        let current_tag = anchor::content_tag(&content);
        let tag_stale = args.tag.as_ref().is_some_and(|tag| tag != &current_tag);
        let drifted_since_served = matches!(
            anchor_state::check_anchor(&path, &content),
            anchor::AnchorStatus::Drifted
        );

        let count = content.matches(&args.old_string).count();

        // Stale tag AND the remembered text is gone: refuse and demand a
        // re-read instead of fuzzy-editing a misremembered file.
        if count == 0 && tag_stale {
            return Err(EditError(format!(
                "Content anchor is stale: the file no longer hashes to tag {} (current {}). \
                 old_string was not found either - the file changed since it was read. \
                 Re-read the file and retry with the fresh tag.",
                args.tag.as_deref().unwrap_or("?"),
                current_tag
            )));
        }

        if count == 0 {
            // Exact miss: fall back to the shared fuzzy matcher before failing.
            let outcome = fuzzy_match::find_match(
                &content,
                &args.old_string,
                true,
                fuzzy_match::DEFAULT_FUZZY_THRESHOLD,
            );
            let Some(matched) = outcome.matched.clone() else {
                let mut message = fuzzy_match::format_match_error(
                    &args.path,
                    &args.old_string,
                    &outcome,
                    true,
                    (fuzzy_match::DEFAULT_FUZZY_THRESHOLD * 100.0) as i32,
                );
                if let Some(closest) = outcome.closest.as_ref() {
                    message.push_str("\n\nClosest candidate:\n");
                    message.push_str(&anchor::render_preview_windows(
                        &content,
                        &[closest.start_line],
                        PREVIEW_CONTEXT_LINES,
                        PREVIEW_MAX_LINE_CHARS,
                    ));
                }
                return Err(EditError(message));
            };

            let end = matched.start_index + matched.actual_text.len();
            let mut new_content = String::with_capacity(content.len() + args.new_string.len());
            new_content.push_str(&content[..matched.start_index]);
            new_content.push_str(&args.new_string);
            new_content.push_str(&content[end..]);

            if new_content == content {
                // Byte-identical result: the model re-issued (a fuzzy variant
                // of) what is already there. Soft hint first; identical
                // repeats escalate (upstream no-op loop guard).
                let verdict = anchor_state::record_noop(
                    &path,
                    anchor::edit_input_fingerprint(&args.old_string, &args.new_string),
                );
                return Err(EditError(if verdict.escalate {
                    format!(
                        "This identical no-op edit was already attempted {} times. \
                         Stop retrying: re-read the file, or state that no change is needed.",
                        verdict.count
                    )
                } else {
                    "The edit would leave the file byte-identical: the matched text already \
                     equals new_string. Re-read the file and edit the actual current content."
                        .to_string()
                }));
            }

            self.fs
                .write(&path, new_content.as_bytes())
                .map_err(|e| EditError(format!("Failed to write file: {}", e)))?;
            anchor_state::reset_noop(&path);
            let new_tag = anchor_state::record_anchor(&path, &new_content);

            tracing::info!(
                "🔧 [tool-result] edit: fuzzy-replaced 1 occurrence in \"{}\" ({:.0}% similar, line {})",
                path.display(),
                matched.confidence * 100.0,
                matched.start_line
            );

            let warnings = anchor_warnings(
                path_note.as_deref(),
                false,
                drifted_since_served,
                args.tag.as_deref(),
                &current_tag,
            );
            let mut message = format!(
                "Replaced 1 fuzzy-matched occurrence in {} (line {}, {:.0}% similar to old_string; \
                 whitespace/indentation differences were tolerated){}",
                args.path,
                matched.start_line,
                matched.confidence * 100.0,
                warnings
            );
            message.push('\n');
            message.push_str(&reground_section(
                &args.path,
                &new_content,
                new_tag,
                matched.start_line,
                args.new_string.lines().count().max(1),
            ));
            return Ok(message);
        }

        if !replace_all && count > 1 {
            let occurrence_lines: Vec<usize> = content
                .match_indices(&args.old_string)
                .take(MAX_PREVIEW_MATCHES)
                .map(|(index, _)| content[..index].split('\n').count())
                .collect();
            return Err(EditError(format!(
                "old_string appears {} times in {}. Provide more surrounding context to make it unique, \
                 or set replace_all to true. Occurrences:\n{}",
                count,
                path.display(),
                anchor::render_preview_windows(
                    &content,
                    &occurrence_lines,
                    PREVIEW_CONTEXT_LINES,
                    PREVIEW_MAX_LINE_CHARS
                )
            )));
        }

        let new_content = if replace_all {
            content.replace(&args.old_string, &args.new_string)
        } else {
            content.replacen(&args.old_string, &args.new_string, 1)
        };

        self.fs
            .write(&path, new_content.as_bytes())
            .map_err(|e| EditError(format!("Failed to write file: {}", e)))?;
        anchor_state::reset_noop(&path);
        let new_tag = anchor_state::record_anchor(&path, &new_content);

        let replacements = if replace_all { count } else { 1 };
        let first_line = content
            .find(&args.old_string)
            .map(|index| content[..index].split('\n').count())
            .unwrap_or(1);

        tracing::info!(
            "🔧 [tool-result] edit: replaced {} occurrence(s) in \"{}\"",
            replacements,
            path.display()
        );

        let warnings = anchor_warnings(
            path_note.as_deref(),
            tag_stale,
            drifted_since_served,
            args.tag.as_deref(),
            &current_tag,
        );
        let mut message = format!(
            "Successfully replaced {} occurrence(s) in {}{}",
            replacements, args.path, warnings
        );
        message.push('\n');
        message.push_str(&reground_section(
            &args.path,
            &new_content,
            new_tag,
            first_line,
            args.new_string.lines().count().max(1),
        ));
        Ok(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::read::ReadTool;

    const INDENT_FILE: &str = "fn main() {\n        let x = 1;\n        let y = 2;\n}\n";

    fn temp_fixture(name: &str, content: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "chatshell-edit-test-{}-{}",
            name,
            std::process::id()
        ));
        std::fs::write(&path, content).unwrap();
        path
    }

    fn temp_root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "chatshell-edit-root-{}-{}",
            name,
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    async fn run_edit(
        tool: &EditTool,
        path: &str,
        old: &str,
        new: &str,
    ) -> Result<String, EditError> {
        run_edit_full(tool, path, old, new, None, None).await
    }

    async fn run_edit_full(
        tool: &EditTool,
        path: &str,
        old: &str,
        new: &str,
        replace_all: Option<bool>,
        tag: Option<String>,
    ) -> Result<String, EditError> {
        use rig::agent::tool::Tool as _;
        let args = EditArgs {
            path: path.to_string(),
            old_string: old.to_string(),
            new_string: new.to_string(),
            replace_all,
            tag,
        };
        let mut ctx = rig::agent::tool::ToolContext::default();
        tool.call(&mut ctx, args).await
    }

    async fn run_read(read_tool: &ReadTool, path: &str) -> String {
        use rig::agent::tool::Tool as _;
        let mut ctx = rig::agent::tool::ToolContext::default();
        read_tool
            .call(
                &mut ctx,
                crate::tools::read::ReadArgs {
                    path: path.to_string(),
                    offset: None,
                    limit: None,
                    question: None,
                },
            )
            .await
            .unwrap()
    }

    /// Extract the 4-hex tag from a `[path#TAG]` header line.
    fn tag_from_header(output: &str) -> String {
        let header = output
            .lines()
            .find(|line| line.starts_with('[') && line.contains('#'))
            .expect("output carries a [path#TAG] header");
        header
            .split('#')
            .nth(1)
            .and_then(|rest| rest.strip_suffix(']'))
            .expect("well-formed header")
            .to_string()
    }

    #[test]
    fn test_edit_tool_creation() {
        let tool = EditTool::new();
        assert_eq!(EditTool::NAME, "edit");
        let _ = tool;
    }

    #[test]
    fn test_edit_args_basic() {
        let args: EditArgs = serde_json::from_str(
            r#"{"path": "/tmp/test.rs", "old_string": "foo", "new_string": "bar"}"#,
        )
        .unwrap();
        assert_eq!(args.path, "/tmp/test.rs");
        assert_eq!(args.old_string, "foo");
        assert_eq!(args.new_string, "bar");
        assert_eq!(args.replace_all, None);
        assert_eq!(args.tag, None);
    }

    #[test]
    fn test_edit_args_with_tag() {
        let args: EditArgs = serde_json::from_str(
            r#"{"path": "/tmp/t.rs", "old_string": "a", "new_string": "b", "tag": "1a2b"}"#,
        )
        .unwrap();
        assert_eq!(args.tag.as_deref(), Some("1a2b"));
    }

    #[tokio::test]
    async fn test_exact_edit_unchanged() {
        let path = temp_fixture("exact", "alpha\nbeta\ngamma\n");
        let out = run_edit(&EditTool::new(), path.to_str().unwrap(), "beta", "BETA")
            .await
            .unwrap();
        assert!(out.contains("1 occurrence"), "{out}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "alpha\nBETA\ngamma\n"
        );
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_fuzzy_fallback_tolerates_indent_shift() {
        let path = temp_fixture("indent", INDENT_FILE);
        let old = "fn main() {\n    let x = 1;\n    let y = 2;\n}";
        let out = run_edit(
            &EditTool::new(),
            path.to_str().unwrap(),
            old,
            "fn main() {\n    let x = 10;\n    let y = 20;\n}",
        )
        .await
        .unwrap();
        assert!(out.contains("fuzzy-matched"), "{out}");
        let updated = std::fs::read_to_string(&path).unwrap();
        assert!(updated.contains("let x = 10;"), "{updated}");
        assert!(updated.contains("let y = 20;"), "{updated}");
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_fuzzy_no_match_returns_diagnostics() {
        let path = temp_fixture("nomatch", "one\ntwo\nthree\n");
        let err = run_edit(
            &EditTool::new(),
            path.to_str().unwrap(),
            "zzzzzzzz nope",
            "x",
        )
        .await
        .unwrap_err();
        let msg = err.0;
        assert!(
            msg.contains("No exact match found") || msg.contains("No match found"),
            "{msg}"
        );
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_ambiguous_exact_lists_preview_windows() {
        let path = temp_fixture("dup", "header\nsame\nmiddle\nsame\ntrailer\n");
        let err = run_edit(&EditTool::new(), path.to_str().unwrap(), "same", "diff")
            .await
            .unwrap_err();
        assert!(err.0.contains("2 times"), "{}", err.0);
        assert!(err.0.contains("Occurrences:"), "{}", err.0);
        // Preview windows mark the match starts.
        assert!(err.0.contains(">    2 | same"), "{}", err.0);
        assert!(err.0.contains(">    4 | same"), "{}", err.0);
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_reground_window_in_success_output() {
        let path = temp_fixture("reground", "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\n");
        let out = run_edit(&EditTool::new(), path.to_str().unwrap(), "l4", "L4")
            .await
            .unwrap();
        assert!(
            out.lines()
                .any(|line| line.starts_with('[') && line.contains('#')),
            "fresh header: {out}"
        );
        // Numbered context lines in read-tool format.
        assert!(out.contains("     1\tl1"), "{out}");
        assert!(out.contains("     4\tL4"), "{out}");
        assert!(out.contains("     7\tl7"), "{out}");
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_tag_roundtrip_read_edit() {
        let path = temp_fixture("taground", "fn a() {}\nfn b() {}\n");
        let read_tool = ReadTool::new();
        let read_output = run_read(&read_tool, path.to_str().unwrap()).await;
        assert!(
            read_output.starts_with('[') && read_output.contains('#'),
            "{read_output}"
        );
        let tag = tag_from_header(&read_output);

        let out = run_edit_full(
            &EditTool::new(),
            path.to_str().unwrap(),
            "fn b() {}",
            "fn b() { return; }",
            None,
            Some(tag),
        )
        .await
        .unwrap();
        // Fresh tag differs from the consumed one? Not necessarily (4-hex
        // space), but the header must be present and the edit applied.
        assert!(out.contains('#'), "{out}");
        assert!(
            std::fs::read_to_string(&path).unwrap().contains("return"),
            "edit applied"
        );
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_stale_tag_with_exact_match_recovers() {
        let path = temp_fixture("stagerec", "value = 1\n");
        let read_tool = ReadTool::new();
        let read_output = run_read(&read_tool, path.to_str().unwrap()).await;
        let stale_tag = tag_from_header(&read_output);

        // External modification after the read.
        std::fs::write(&path, "value = 2\nother = 9\n").unwrap();

        let out = run_edit_full(
            &EditTool::new(),
            path.to_str().unwrap(),
            "value = 2",
            "value = 3",
            None,
            Some(stale_tag),
        )
        .await
        .unwrap();
        assert!(out.contains("stale"), "recovery warning present: {out}");
        assert!(out.contains("re-anchored"), "{out}");
        assert!(
            std::fs::read_to_string(&path)
                .unwrap()
                .contains("value = 3")
        );
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_stale_tag_without_match_refuses() {
        let path = temp_fixture("stageref", "alpha\nbeta\n");
        let read_tool = ReadTool::new();
        let read_output = run_read(&read_tool, path.to_str().unwrap()).await;
        let stale_tag = tag_from_header(&read_output);

        std::fs::write(&path, "completely different content\n").unwrap();

        let err = run_edit_full(
            &EditTool::new(),
            path.to_str().unwrap(),
            "beta",
            "gamma",
            None,
            Some(stale_tag),
        )
        .await
        .unwrap_err();
        assert!(err.0.contains("Content anchor is stale"), "{}", err.0);
        assert!(err.0.contains("Re-read"), "{}", err.0);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "completely different content\n"
        );
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_drift_warning_without_tag() {
        let path = temp_fixture("drift", "one\ntwo\n");
        let read_tool = ReadTool::new();
        let _ = run_read(&read_tool, path.to_str().unwrap()).await;

        std::fs::write(&path, "one\nTWO\nthree\n").unwrap();

        let out = run_edit(&EditTool::new(), path.to_str().unwrap(), "TWO", "two")
            .await
            .unwrap();
        assert!(
            out.contains("changed since it was last read/written"),
            "{out}"
        );
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_noop_guard_soft_then_hard() {
        let path = temp_fixture("noop", INDENT_FILE);
        let tool = EditTool::new();
        // old_string with 4-space indent; new_string equals the actual
        // 8-space content: fuzzy match maps onto existing bytes.
        let old = "fn main() {\n    let x = 1;\n    let y = 2;\n}";
        let new = "fn main() {\n        let x = 1;\n        let y = 2;\n}";

        let first = run_edit(&tool, path.to_str().unwrap(), old, new)
            .await
            .unwrap_err();
        assert!(first.0.contains("byte-identical"), "{}", first.0);

        let second = run_edit(&tool, path.to_str().unwrap(), old, new)
            .await
            .unwrap_err();
        assert!(second.0.contains("byte-identical"), "{}", second.0);

        let third = run_edit(&tool, path.to_str().unwrap(), old, new)
            .await
            .unwrap_err();
        assert!(third.0.contains("already attempted 3 times"), "{}", third.0);

        // The file was never touched.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), INDENT_FILE);
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn test_suffix_resolution_unique() {
        let root = temp_root("suffixu");
        let nested = root.join("packages/x/src/foo.rs");
        std::fs::create_dir_all(nested.parent().unwrap()).unwrap();
        std::fs::write(&nested, "const X: u8 = 1;\n").unwrap();

        let requested = root.join("src/foo.rs"); // does not exist
        let tool = EditTool::with_project_root(root.clone());
        let out = run_edit(
            &tool,
            requested.to_str().unwrap(),
            "const X: u8 = 1;",
            "const X: u8 = 2;",
        )
        .await
        .unwrap();
        assert!(out.contains("resolved missing"), "{out}");
        assert!(out.contains("unique workspace suffix"), "{out}");
        assert!(std::fs::read_to_string(&nested).unwrap().contains("u8 = 2"));
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn test_suffix_resolution_ambiguous() {
        let root = temp_root("suffixa");
        for dir in ["a/src", "b/src"] {
            let file = root.join(dir).join("foo.rs");
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, "const X: u8 = 1;\n").unwrap();
        }

        let requested = root.join("src/foo.rs");
        let tool = EditTool::with_project_root(root.clone());
        let err = run_edit(
            &tool,
            requested.to_str().unwrap(),
            "const X: u8 = 1;",
            "const X: u8 = 2;",
        )
        .await
        .unwrap_err();
        assert!(err.0.contains("Possible matches"), "{}", err.0);
        assert!(err.0.contains("a/src/foo.rs"), "{}", err.0);
        assert!(err.0.contains("b/src/foo.rs"), "{}", err.0);
        std::fs::remove_dir_all(root).ok();
    }
}
