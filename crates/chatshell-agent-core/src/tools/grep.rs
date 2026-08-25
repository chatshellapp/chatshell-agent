//! Grep tool for LLM agents
//!
//! A content search tool powered by ripgrep's core crates (grep-regex, grep-searcher, ignore).
//! Performs fast, in-process regex searching without requiring external binaries.

use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use grep_regex::RegexMatcherBuilder;
use grep_searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkContext, SinkMatch};
use rig::agent::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::path_policy;
use crate::tool_fs::{Fs, StdFs, WalkFilters};

const MAX_OUTPUT_CHARS: usize = 50_000;
const SEARCH_TIMEOUT_SECS: u64 = 30;
/// Per-file match cap in content mode when searching a directory:
/// breadth-first — one hot file must not crowd out a multi-file search.
const MULTI_FILE_MATCHES_PER_FILE: usize = 20;
/// Per-file match cap when searching a single explicit file.
const SINGLE_FILE_MATCHES: usize = 200;
/// Files larger than this are skipped in directory searches (and reported),
/// mirroring ripgrep's mmap-cap behavior for oversized inputs.
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;
/// How many skipped-file paths to list in the footer.
const MAX_SKIPPED_PATHS_LISTED: usize = 5;

#[derive(Debug, Clone, Deserialize)]
pub struct GrepArgs {
    pub pattern: String,
    #[serde(default)]
    pub path: Option<String>,
    /// Glob pattern to filter files (e.g. "*.rs", "*.{ts,tsx}")
    #[serde(default)]
    pub glob: Option<String>,
    /// File type to filter by (e.g. "rust", "js", "py"). Uses ripgrep's built-in type definitions.
    #[serde(default)]
    pub file_type: Option<String>,
    #[serde(default)]
    pub case_insensitive: Option<bool>,
    /// Enable multiline matching (`.` matches `\n`, patterns can span lines)
    #[serde(default)]
    pub multiline: Option<bool>,
    /// Context lines around each match (sets both before and after)
    #[serde(default)]
    pub context_lines: Option<usize>,
    /// Lines to show before each match (overrides context_lines for before)
    #[serde(default)]
    pub before_context: Option<usize>,
    /// Lines to show after each match (overrides context_lines for after)
    #[serde(default)]
    pub after_context: Option<usize>,
    /// Skip the first N matching files (file-based pagination). Use the
    /// value from the previous result's footer to fetch the next page.
    #[serde(default)]
    pub skip: Option<usize>,
    /// Maximum total number of matches to return across all files
    #[serde(default)]
    pub head_limit: Option<usize>,
    /// "content" (default), "files_with_matches", or "count"
    #[serde(default)]
    pub output_mode: Option<String>,
    /// Show line numbers in content output mode (default: true)
    #[serde(default)]
    pub line_numbers: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
#[error("Grep error: {0}")]
pub struct GrepError(String);

fn default_fs() -> Arc<dyn Fs> {
    Arc::new(StdFs)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GrepTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    default_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_root: Option<PathBuf>,
    #[serde(skip, default = "default_fs")]
    fs: Arc<dyn Fs>,
}

impl Default for GrepTool {
    fn default() -> Self {
        Self::new()
    }
}

impl GrepTool {
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

impl Tool for GrepTool {
    const NAME: &'static str = "grep";

    type Error = GrepError;
    type Args = GrepArgs;
    type Output = String;

    fn description(&self) -> String {
        "Search file contents using regular expressions. \
                Returns matching lines with file paths and line numbers. \
                Supports multiple output modes, file type filtering, multiline matching, \
                and context lines. Powered by ripgrep's core engine for fast in-process searching."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        let path_description = if let Some(ref default_dir) = self.default_path {
            format!(
                "File or directory to search in (defaults to: {})",
                default_dir
            )
        } else {
            "File or directory to search in (defaults to home directory)".to_string()
        };
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for. Uses ripgrep syntax (not grep). Literal braces need escaping."
                },
                "path": {
                    "type": "string",
                    "description": path_description
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. \"*.rs\", \"*.{ts,tsx}\")"
                },
                "file_type": {
                    "type": "string",
                    "description": "File type to search (e.g. \"rust\", \"js\", \"py\", \"ts\", \"go\", \"java\", \"cpp\"). More efficient than glob for standard file types."
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Enable case-insensitive search. Defaults to false."
                },
                "multiline": {
                    "type": "boolean",
                    "description": "Enable multiline matching where . matches newlines and patterns can span lines. Defaults to false."
                },
                "context_lines": {
                    "type": "number",
                    "description": "Number of context lines to show before and after each match"
                },
                "before_context": {
                    "type": "number",
                    "description": "Number of lines to show before each match (overrides context_lines)"
                },
                "after_context": {
                    "type": "number",
                    "description": "Number of lines to show after each match (overrides context_lines)"
                },
                "head_limit": {
                    "type": "number",
                    "description": "Maximum total number of matches to return across all files"
                },
                "skip": {
                    "type": "number",
                    "description": "Skip the first N matching files (file-based pagination). Use the skip value from the previous result's footer to fetch the next page."
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode: \"content\" shows matching lines with context (default), \"files_with_matches\" shows only file paths containing matches, \"count\" shows match counts per file"
                },
                "line_numbers": {
                    "type": "boolean",
                    "description": "Show line numbers in content output mode. Defaults to true."
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
            "🔧 [tool-call] grep: pattern=\"{}\" path={:?} glob={:?} type={:?} mode={:?}",
            args.pattern,
            args.path,
            args.glob,
            args.file_type,
            args.output_mode
        );

        let search_path = args
            .path
            .clone()
            .or(self.default_path.clone())
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| ".".to_string())
            });

        path_policy::check_read(Path::new(&search_path), self.project_root.as_deref())
            .map_err(GrepError)?;

        let fs = Arc::clone(&self.fs);
        tokio::time::timeout(
            std::time::Duration::from_secs(SEARCH_TIMEOUT_SECS),
            tokio::task::spawn_blocking(move || run_search(fs, &args, &search_path)),
        )
        .await
        .map_err(|_| {
            GrepError(format!(
                "Search timed out after {}s (search scope may be too large)",
                SEARCH_TIMEOUT_SECS
            ))
        })?
        .map_err(|e| GrepError(format!("Search task failed: {}", e)))?
    }
}

// ---------------------------------------------------------------------------
// Core search logic (runs on a blocking thread)
// ---------------------------------------------------------------------------

fn run_search(fs: Arc<dyn Fs>, args: &GrepArgs, search_path: &str) -> Result<String, GrepError> {
    let mut matcher_builder = RegexMatcherBuilder::new();
    if args.case_insensitive.unwrap_or(false) {
        matcher_builder.case_insensitive(true);
    }
    if args.multiline.unwrap_or(false) {
        matcher_builder.multi_line(true);
        matcher_builder.dot_matches_new_line(true);
    }

    let matcher = matcher_builder
        .build(&args.pattern)
        .map_err(|e| GrepError(format!("Invalid regex pattern: {}", e)))?;

    let mut searcher_builder = SearcherBuilder::new();
    searcher_builder.line_number(true);
    searcher_builder.binary_detection(BinaryDetection::quit(b'\x00'));

    if args.multiline.unwrap_or(false) {
        searcher_builder.multi_line(true);
    }

    let before = args.before_context.or(args.context_lines).unwrap_or(0);
    let after = args.after_context.or(args.context_lines).unwrap_or(0);
    if before > 0 {
        searcher_builder.before_context(before);
    }
    if after > 0 {
        searcher_builder.after_context(after);
    }

    let mut searcher = searcher_builder.build();
    let output_mode = args.output_mode.as_deref().unwrap_or("content");
    let head_limit = args.head_limit.unwrap_or(usize::MAX);
    let path = Path::new(search_path);

    let path_meta = fs
        .metadata(path)
        .map_err(|_| GrepError(format!("Path not found: {}", search_path)))?;

    let skip = args.skip.unwrap_or(0);
    let show_line_numbers = args.line_numbers.unwrap_or(true);

    let mut run = SearchRun {
        output_mode,
        total_matches: 0,
        head_limit,
        show_line_numbers,
        per_file_match_limit: MULTI_FILE_MATCHES_PER_FILE,
        skip,
        files_matched: 0,
        files_skipped_by_page: 0,
        skipped_oversized: Vec::new(),
        hit_head_limit: false,
        hit_output_cap: false,
    };
    let mut output = String::new();

    if path_meta.is_file {
        run.per_file_match_limit = if args.head_limit.is_some() {
            head_limit
        } else {
            SINGLE_FILE_MATCHES
        };
        search_file(
            fs.as_ref(),
            &matcher,
            &mut searcher,
            path,
            &mut run,
            &mut output,
        );
    } else {
        run.per_file_match_limit = if args.head_limit.is_some() {
            head_limit
        } else {
            MULTI_FILE_MATCHES_PER_FILE
        };
        let files = fs
            .walk(
                path,
                &WalkFilters {
                    include: args.glob.clone().into_iter().collect(),
                    types: args.file_type.clone().into_iter().collect(),
                    ..Default::default()
                },
            )
            .map_err(|e| GrepError(format!("Failed to walk search path: {}", e)))?;

        for file in files {
            if run.total_matches >= run.head_limit {
                break; // hit_head_limit already flagged by the sink
            }
            if output.len() > MAX_OUTPUT_CHARS {
                run.hit_output_cap = true;
                break;
            }
            if let Ok(meta) = fs.metadata(&file)
                && meta.len > MAX_FILE_BYTES
            {
                run.skipped_oversized
                    .push(file.to_string_lossy().to_string());
                continue;
            }
            search_file(
                fs.as_ref(),
                &matcher,
                &mut searcher,
                &file,
                &mut run,
                &mut output,
            );
        }
    }

    let files_shown = run.files_matched - run.files_skipped_by_page;
    let next_skip = skip + files_shown;

    if output.is_empty() {
        if run.files_skipped_by_page > 0 {
            return Ok(format!(
                "No matches found after skipping {} matching file(s) (skip={}). \
                 All matching files were skipped; use skip=0 to start from the beginning.",
                run.files_skipped_by_page, skip
            ));
        }
        return Ok("No matches found.".to_string());
    }

    let mut footers: Vec<String> = Vec::new();

    if run.hit_head_limit {
        let suggestion = run.total_matches.saturating_mul(2).max(1);
        footers.push(format!(
            "[match limit reached at {} match(es) — more may exist. \
             Use head_limit={} to see more, or narrow path/glob/file_type.]",
            run.total_matches, suggestion
        ));
    }
    if output.len() > MAX_OUTPUT_CHARS {
        truncate_char_safe(&mut output, MAX_OUTPUT_CHARS);
        footers.push(format!(
            "[output truncated at {} chars — narrow the pattern/path, or page with skip/head_limit.]",
            MAX_OUTPUT_CHARS
        ));
    }
    if run.files_skipped_by_page > 0 {
        footers.push(format!(
            "[{} matching file(s) skipped by skip={}; showing files {}-{} of {}. \
             Use skip={} for the next page.]",
            run.files_skipped_by_page,
            skip,
            skip + 1,
            next_skip,
            run.files_matched,
            next_skip
        ));
    } else if run.files_matched > 0 && (run.hit_head_limit || run.hit_output_cap) {
        footers.push(format!(
            "[showing files 1-{} of {}+. Use skip={} for the next page.]",
            files_shown, run.files_matched, next_skip
        ));
    }
    if !run.skipped_oversized.is_empty() {
        let listed: Vec<&str> = run
            .skipped_oversized
            .iter()
            .take(MAX_SKIPPED_PATHS_LISTED)
            .map(|s| s.as_str())
            .collect();
        let extra = run.skipped_oversized.len() - listed.len();
        let mut note = format!(
            "[skipped {} file(s) larger than {} MiB: {}",
            run.skipped_oversized.len(),
            MAX_FILE_BYTES / (1024 * 1024),
            listed.join(", ")
        );
        if extra > 0 {
            note.push_str(&format!(", +{} more", extra));
        }
        note.push_str(" — read them directly if needed.]");
        footers.push(note);
    }

    if !footers.is_empty() {
        output.push_str("\n\n");
        output.push_str(&footers.join("\n"));
    }

    tracing::info!(
        "🔧 [tool-result] grep: {} matches across {} files, {} chars",
        run.total_matches,
        files_shown,
        output.len()
    );

    Ok(output)
}

/// Truncate to `max_bytes` on a UTF-8 char boundary (String::truncate
/// panics mid-codepoint).
fn truncate_char_safe(s: &mut String, max_bytes: usize) {
    if s.len() <= max_bytes {
        return;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
}

/// Accumulated per-run search state threaded through `search_file`.
///
/// `files_matched` counts files that produced at least one match (whether or
/// not the page window suppressed them); `files_skipped_by_page` counts the
/// ones the `skip` window dropped.
struct SearchRun<'a> {
    output_mode: &'a str,
    total_matches: usize,
    head_limit: usize,
    show_line_numbers: bool,
    /// Match cap applied within a single file (breadth-first budgets).
    per_file_match_limit: usize,
    /// Number of leading matching files suppressed (pagination).
    skip: usize,
    files_matched: usize,
    files_skipped_by_page: usize,
    skipped_oversized: Vec<String>,
    hit_head_limit: bool,
    hit_output_cap: bool,
}

/// Search one file into a per-file buffer, then either append it to the run
/// output (inside the page window) or account it as page-skipped.
fn search_file(
    fs: &dyn Fs,
    matcher: &grep_regex::RegexMatcher,
    searcher: &mut Searcher,
    path: &Path,
    run: &mut SearchRun<'_>,
    output: &mut String,
) {
    let output_mode = run.output_mode;
    let show_line_numbers = run.show_line_numbers;
    let path_display = path.to_string_lossy();

    let Ok(reader) = fs.open(path) else {
        return; // unreadable file: skip, matching search_path's silent skip
    };

    let mut file_output = String::new();
    let mut file_matches: usize = 0;
    let mut file_capped = false;

    match output_mode {
        "files_with_matches" => {
            let mut found = false;
            let _ = searcher.search_reader(matcher, reader, FileMatchSink { found: &mut found });
            if found {
                let _ = writeln!(file_output, "{}", path_display);
                file_matches = 1;
                run.total_matches += 1;
            }
        }
        "count" => {
            let mut count: u64 = 0;
            let _ = searcher.search_reader(matcher, reader, CountSink { count: &mut count });
            if count > 0 {
                let _ = writeln!(file_output, "{}:{}", path_display, count);
                file_matches = count as usize;
                run.total_matches += count as usize;
            }
        }
        _ => {
            let _ = searcher.search_reader(
                matcher,
                reader,
                ContentSink {
                    path: &path_display,
                    output: &mut file_output,
                    file_matches: &mut file_matches,
                    file_capped: &mut file_capped,
                    total_matches: &mut run.total_matches,
                    head_limit: run.head_limit,
                    per_file_limit: run.per_file_match_limit,
                    hit_head_limit: &mut run.hit_head_limit,
                    show_line_numbers,
                    needs_separator: false,
                    had_output: false,
                },
            );
        }
    }

    if file_matches == 0 {
        return;
    }

    run.files_matched += 1;
    if run.files_matched <= run.skip {
        run.files_skipped_by_page += 1;
        return;
    }

    if file_capped {
        let _ = writeln!(
            file_output,
            "[{} match(es) shown from this file; per-file cap {} reached — search this file directly for all matches]",
            file_matches, run.per_file_match_limit
        );
    }
    output.push_str(&file_output);

    if run.total_matches >= run.head_limit {
        run.hit_head_limit = true;
    }
    if output.len() > MAX_OUTPUT_CHARS {
        run.hit_output_cap = true;
    }
}

/// Collects matching lines with file paths, line numbers, and context.
/// Output format matches ripgrep's `--no-heading --color=never` style:
///   match:   `filepath:linenum:content`
///   context: `filepath-linenum-content`
///   group separator: `--`
///
/// Enforces both the run-wide `head_limit` and the per-file cap, stopping the
/// current file's search when either is hit (other files continue).
struct ContentSink<'a> {
    path: &'a str,
    output: &'a mut String,
    file_matches: &'a mut usize,
    file_capped: &'a mut bool,
    total_matches: &'a mut usize,
    head_limit: usize,
    per_file_limit: usize,
    hit_head_limit: &'a mut bool,
    show_line_numbers: bool,
    needs_separator: bool,
    had_output: bool,
}

impl Sink for ContentSink<'_> {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch<'_>) -> Result<bool, Self::Error> {
        if *self.total_matches >= self.head_limit {
            *self.hit_head_limit = true;
            return Ok(false);
        }
        if *self.file_matches >= self.per_file_limit {
            *self.file_capped = true;
            return Ok(false);
        }
        *self.total_matches += 1;
        *self.file_matches += 1;

        if self.needs_separator {
            self.output.push_str("--\n");
            self.needs_separator = false;
        }
        self.had_output = true;

        let line = String::from_utf8_lossy(mat.bytes());
        if self.show_line_numbers {
            if let Some(n) = mat.line_number() {
                let _ = write!(self.output, "{}:{}:{}", self.path, n, line);
            } else {
                let _ = write!(self.output, "{}:{}", self.path, line);
            }
        } else {
            let _ = write!(self.output, "{}:{}", self.path, line);
        }
        if !line.ends_with('\n') {
            self.output.push('\n');
        }
        Ok(true)
    }

    fn context(&mut self, _: &Searcher, ctx: &SinkContext<'_>) -> Result<bool, Self::Error> {
        if *self.total_matches >= self.head_limit || *self.file_matches >= self.per_file_limit {
            return Ok(false);
        }
        if self.needs_separator {
            self.output.push_str("--\n");
            self.needs_separator = false;
        }

        let line = String::from_utf8_lossy(ctx.bytes());
        if self.show_line_numbers {
            if let Some(n) = ctx.line_number() {
                let _ = write!(self.output, "{}-{}-{}", self.path, n, line);
            } else {
                let _ = write!(self.output, "{}-{}", self.path, line);
            }
        } else {
            let _ = write!(self.output, "{}-{}", self.path, line);
        }
        if !line.ends_with('\n') {
            self.output.push('\n');
        }
        Ok(true)
    }

    fn context_break(&mut self, _: &Searcher) -> Result<bool, Self::Error> {
        if self.had_output {
            self.needs_separator = true;
        }
        Ok(true)
    }
}

/// Short-circuits on first match — used by `files_with_matches` mode.
struct FileMatchSink<'a> {
    found: &'a mut bool,
}

impl Sink for FileMatchSink<'_> {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, _: &SinkMatch<'_>) -> Result<bool, Self::Error> {
        *self.found = true;
        Ok(false)
    }
}

/// Counts matches in a single file — used by `count` mode.
struct CountSink<'a> {
    count: &'a mut u64,
}

impl Sink for CountSink<'_> {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, _: &SinkMatch<'_>) -> Result<bool, Self::Error> {
        *self.count += 1;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(pattern: &str, extra: serde_json::Value) -> GrepArgs {
        let mut value = json!({ "pattern": pattern });
        if let (Some(obj), Some(extra_obj)) = (value.as_object_mut(), extra.as_object()) {
            for (k, v) in extra_obj {
                obj.insert(k.clone(), v.clone());
            }
        }
        serde_json::from_value(value).expect("valid args")
    }

    #[test]
    fn per_file_cap_in_directory_search() {
        let dir = tempfile::tempdir().unwrap();
        let hot = dir.path().join("hot.txt");
        let content: String = (0..30).map(|i| format!("match line {}\n", i)).collect();
        std::fs::write(&hot, content).unwrap();

        let out = run_search(
            default_fs(),
            &args("match", json!({ "path": dir.path().to_string_lossy() })),
            &dir.path().to_string_lossy(),
        )
        .unwrap();

        assert!(out.contains("match line 19"), "shows matches up to cap");
        assert!(!out.contains("match line 20"), "stops at per-file cap");
        assert!(
            out.contains("per-file cap 20 reached"),
            "footer explains the cap: {out}"
        );
    }

    #[test]
    fn single_file_gets_higher_cap() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("single.txt");
        let content: String = (0..30).map(|i| format!("match line {}\n", i)).collect();
        std::fs::write(&file, content).unwrap();

        let out = run_search(
            default_fs(),
            &args("match", json!({ "path": file.to_string_lossy() })),
            &file.to_string_lossy(),
        )
        .unwrap();

        assert!(
            out.contains("match line 29"),
            "single file not capped at 20"
        );
        assert!(!out.contains("per-file cap"), "no cap footer: {out}");
    }

    #[test]
    fn skip_paginates_by_matching_files() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a.txt", "b.txt", "c.txt"] {
            std::fs::write(dir.path().join(name), "needle here\n").unwrap();
        }
        let dir_str = dir.path().to_string_lossy().to_string();

        let out = run_search(
            default_fs(),
            &args("needle", json!({ "path": dir_str, "skip": 2 })),
            &dir_str,
        )
        .unwrap();

        assert_eq!(
            out.matches(":1:needle here").count(),
            1,
            "only one file inside the page window: {out}"
        );
        assert!(
            out.contains("2 matching file(s) skipped by skip=2"),
            "skip footer present: {out}"
        );
        assert!(out.contains("Use skip=3 for the next page"));
    }

    #[test]
    fn head_limit_footer_carries_suggestion() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("many.txt");
        let content: String = (0..5).map(|i| format!("hit {}\n", i)).collect();
        std::fs::write(&file, content).unwrap();

        let out = run_search(
            default_fs(),
            &args(
                "hit",
                json!({ "path": file.to_string_lossy(), "head_limit": 2 }),
            ),
            &file.to_string_lossy(),
        )
        .unwrap();

        assert!(out.contains("hit 1"));
        assert!(!out.contains("hit 2\n"));
        assert!(
            out.contains("match limit reached at 2 match(es)"),
            "limit footer: {out}"
        );
        assert!(
            out.contains("Use head_limit=4"),
            "doubled suggestion: {out}"
        );
    }

    #[test]
    fn oversized_files_are_skipped_and_reported() {
        let dir = tempfile::tempdir().unwrap();
        let big = dir.path().join("big.log");
        std::fs::write(&big, vec![b'x'; 5 * 1024 * 1024]).unwrap();
        let small = dir.path().join("small.txt");
        std::fs::write(&small, "needle\n").unwrap();
        let dir_str = dir.path().to_string_lossy().to_string();

        let out = run_search(
            default_fs(),
            &args("needle", json!({ "path": dir_str })),
            &dir_str,
        )
        .unwrap();

        assert!(out.contains("small.txt"), "small file still searched");
        assert!(
            out.contains("skipped 1 file(s) larger than 4 MiB"),
            "oversized footer: {out}"
        );
        assert!(out.contains("big.log"));
    }

    #[test]
    fn truncate_char_safe_respects_codepoints() {
        for (sample, head) in [
            ("你好世界", "你好"),        // Chinese
            ("こんにちは世界", "こん"),  // Japanese
            ("안녕하세요 세계", "안녕"), // Korean
        ] {
            let mut s = sample.repeat(100);
            truncate_char_safe(&mut s, 7); // mid-codepoint
            assert!(s.chars().all(|c| c != '\u{FFFD}'));
            assert!(s.starts_with(head));
            assert!(s.len() <= 7);
        }
    }
}
