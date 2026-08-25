//! Glob tool for LLM agents
//!
//! A file pattern matching tool following established agent-harness glob
//! conventions.
//! Uses the `ignore` crate for directory walking (respects .gitignore)
//! and returns results sorted by modification time (most recent first).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use rig::agent::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::path_policy;
use crate::tool_fs::{Fs, StdFs, WalkFilters};

const MAX_RESULTS: usize = 1000;
const MAX_COLLECT: usize = 100_000;
const GLOB_TIMEOUT_SECS: u64 = 15;

#[derive(Debug, Clone, Deserialize)]
pub struct GlobArgs {
    /// Glob pattern to match files (e.g. "**/*.rs", "src/**/*.tsx")
    pub pattern: String,
    /// Root directory to search from
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("Glob error: {0}")]
pub struct GlobError(String);

fn default_fs() -> Arc<dyn Fs> {
    Arc::new(StdFs)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GlobTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    default_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_root: Option<PathBuf>,
    #[serde(skip, default = "default_fs")]
    fs: Arc<dyn Fs>,
}

impl Default for GlobTool {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobTool {
    pub fn new() -> Self {
        Self {
            default_path: None,
            project_root: None,
            fs: Arc::new(StdFs),
        }
    }

    /// Route IO through a custom [`Fs`] implementation.
    pub fn with_fs(mut self, fs: Arc<dyn Fs>) -> Self {
        self.fs = fs;
        self
    }

    pub fn with_working_directory(path: String) -> Self {
        Self {
            default_path: Some(path),
            ..Self::new()
        }
    }

    pub fn with_project_root(mut self, root: PathBuf) -> Self {
        self.project_root = Some(root);
        self
    }
}

impl Tool for GlobTool {
    const NAME: &'static str = "glob";

    type Error = GlobError;
    type Args = GlobArgs;
    type Output = String;

    fn description(&self) -> String {
        "Find files matching a glob pattern. \
                Returns a list of matching file paths sorted by modification time (most recent first). \
                Use this to discover files by name patterns, \
                find source files, configuration files, or any files \
                in a directory tree. Supports recursive patterns like **/*.rs. \
                Respects .gitignore rules automatically."
                .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        let path_description = if let Some(ref default_dir) = self.default_path {
            format!(
                "Root directory to search from (defaults to: {})",
                default_dir
            )
        } else {
            "Root directory to search from (defaults to home directory)".to_string()
        };
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g. \"**/*.rs\", \"src/**/*.tsx\", \"*.json\", \"*.config.{js,ts}\")"
                },
                "path": {
                    "type": "string",
                    "description": path_description
                }
            },
            "required": ["pattern"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::agent::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        tracing::info!(
            "🔧 [tool-call] glob: pattern=\"{}\" path={:?}",
            args.pattern,
            args.path
        );

        let base_dir = args.path.or(self.default_path.clone()).unwrap_or_else(|| {
            dirs::home_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| ".".to_string())
        });

        path_policy::check_read(Path::new(&base_dir), self.project_root.as_deref())
            .map_err(GlobError)?;

        let pattern = args.pattern;
        let fs = Arc::clone(&self.fs);

        tokio::time::timeout(
            std::time::Duration::from_secs(GLOB_TIMEOUT_SECS),
            tokio::task::spawn_blocking(move || run_glob(fs, &pattern, &base_dir)),
        )
        .await
        .map_err(|_| {
            GlobError(format!(
                "Glob timed out after {}s (directory may be too large)",
                GLOB_TIMEOUT_SECS
            ))
        })?
        .map_err(|e| GlobError(format!("Glob task failed: {}", e)))?
    }
}

fn run_glob(fs: Arc<dyn Fs>, pattern: &str, base_dir: &str) -> Result<String, GlobError> {
    let base_path = Path::new(base_dir);
    if !fs.exists(base_path) {
        return Err(GlobError(format!("Directory not found: {}", base_dir)));
    }

    let paths = fs
        .walk(
            base_path,
            &WalkFilters {
                include: vec![pattern.to_string()],
                max_depth: Some(20),
                max_files: Some(MAX_COLLECT),
                ..Default::default()
            },
        )
        .map_err(|e| GlobError(format!("Failed to walk directory: {}", e)))?;

    let mut entries: Vec<(String, SystemTime)> = Vec::new();
    for path in paths {
        let mtime = fs
            .metadata(&path)
            .ok()
            .and_then(|m| m.modified)
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let display_path = path
            .strip_prefix(base_path)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        entries.push((display_path, mtime));
    }

    entries.sort_by(|a, b| b.1.cmp(&a.1));
    let total_found = entries.len();
    let capped = total_found >= MAX_RESULTS;
    entries.truncate(MAX_RESULTS);

    if entries.is_empty() {
        return Ok("No files found matching the pattern.".to_string());
    }

    let files_shown = entries.len();
    let paths: Vec<&str> = entries.iter().map(|(p, _)| p.as_str()).collect();
    let mut output = paths.join("\n");

    // Footer always states the window and, when capped, how to narrow it.
    if capped {
        output.push_str(&format!(
            "\n\n[results capped at {} files — narrow the pattern (add a directory \
             level or a more specific extension) or set path closer to the targets; \
             {} files found, relative to {}]",
            MAX_RESULTS, total_found, base_dir
        ));
    } else {
        output.push_str(&format!(
            "\n\n({} files found, relative to {})",
            files_shown, base_dir
        ));
    }

    tracing::info!("🔧 [tool-result] glob: found {} files", total_found);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_result_footer_counts_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        std::fs::write(dir.path().join("b.txt"), "x").unwrap();
        let dir_str = dir.path().to_string_lossy().to_string();

        let out = run_glob(default_fs(), "*.txt", &dir_str).unwrap();

        assert!(out.contains("2 files found"), "footer: {out}");
        assert!(!out.contains("capped"), "no cap note: {out}");
    }

    #[test]
    fn cap_is_flagged_with_narrowing_hint() {
        let dir = tempfile::tempdir().unwrap();
        // MAX_RESULTS=1000: exceed it with cheap empty files.
        for i in 0..(MAX_RESULTS + 5) {
            std::fs::write(dir.path().join(format!("f{:05}.txt", i)), "").unwrap();
        }
        let dir_str = dir.path().to_string_lossy().to_string();

        let out = run_glob(default_fs(), "*.txt", &dir_str).unwrap();

        assert!(
            out.contains("results capped at 1000 files"),
            "cap footer: {}",
            &out[out.len().saturating_sub(300)..]
        );
        assert!(out.contains("narrow the pattern"));
        assert!(
            out.contains("1005 files found"),
            "pre-truncate count: {out}"
        );
        // Exactly MAX_RESULTS paths listed, not more.
        assert_eq!(out.matches(".txt\n").count(), MAX_RESULTS);
    }
}
