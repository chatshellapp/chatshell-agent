//! Bash command security analysis.
//!
//! Implements a multi-layered security validator pipeline. Each validator is an
//! independent function that inspects the command and returns a SecurityVerdict.
//! The pipeline runs all validators and returns the most severe verdict.
//!
//! Architecture:
//! - AST-based analysis via tree-sitter for structural understanding
//! - Quote-aware content extraction producing multiple views of the command
//! - Independent validator functions for each security concern
//! - Compound command splitting for per-subcommand analysis

use std::collections::HashSet;
use std::sync::LazyLock;

use crate::bash_ast;

/// Result of a security check on a command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityVerdict {
    /// Command appears safe to execute.
    Allow,
    /// Command is blocked with the given reason.
    Block(String),
    /// Command requires a warning to be prepended to the output.
    Warn(String),
}

// ---------------------------------------------------------------------------
// Quote-aware content extraction
// ---------------------------------------------------------------------------

/// Extracted views of a command with different quoting levels stripped.
///
/// Produces multiple views so that validators can check patterns against the
/// appropriate quote context.
struct QuoteExtracted {
    /// Content with single-quoted sections removed but double-quoted sections preserved.
    /// Used for patterns that bash expands inside double quotes (e.g., `$()`, backticks).
    with_double_quotes: String,
    /// Content with both single and double quoted sections removed.
    /// Used for patterns that only apply in unquoted context (e.g., redirections, pipes).
    fully_unquoted: String,
}

fn extract_quoted_content(command: &str) -> QuoteExtracted {
    let mut with_double_quotes = String::new();
    let mut fully_unquoted = String::new();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    let bytes = command.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i];

        // Backslash escape outside single quotes
        if ch == b'\\' && !in_single_quote && i + 1 < bytes.len() {
            let next = bytes[i + 1] as char;
            if !in_single_quote {
                with_double_quotes.push('\\');
                with_double_quotes.push(next);
            }
            if !in_single_quote && !in_double_quote {
                fully_unquoted.push('\\');
                fully_unquoted.push(next);
            }
            i += 2;
            continue;
        }

        if ch == b'\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
            i += 1;
            continue;
        }

        if ch == b'"' && !in_single_quote {
            in_double_quote = !in_double_quote;
            i += 1;
            continue;
        }

        let c = ch as char;
        if !in_single_quote {
            with_double_quotes.push(c);
        }
        if !in_single_quote && !in_double_quote {
            fully_unquoted.push(c);
        }
        i += 1;
    }

    QuoteExtracted {
        with_double_quotes,
        fully_unquoted,
    }
}

// ---------------------------------------------------------------------------
// Safe wrapper stripping
// ---------------------------------------------------------------------------

/// Environment variable names safe to strip from commands for pattern matching.
/// These cannot execute code, load libraries, or alter program search paths.
///
/// SECURITY: Never add PATH, LD_PRELOAD, LD_LIBRARY_PATH, DYLD_*, PYTHONPATH,
/// NODE_PATH, CLASSPATH, RUBYLIB, GOFLAGS, RUSTFLAGS, NODE_OPTIONS, HOME,
/// TMPDIR, SHELL, or BASH_ENV — these can execute code or affect behavior.
static SAFE_ENV_VARS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        // Go build settings
        "GOEXPERIMENT",
        "GOOS",
        "GOARCH",
        "CGO_ENABLED",
        "GO111MODULE",
        // Rust logging/debugging
        "RUST_BACKTRACE",
        "RUST_LOG",
        // Node environment name (NOT NODE_OPTIONS!)
        "NODE_ENV",
        // Python behavior flags (NOT PYTHONPATH!)
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
        // Pytest configuration
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTEST_DEBUG",
        // Locale and encoding
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_TIME",
        "CHARSET",
        // Terminal and display
        "TERM",
        "COLORTERM",
        "NO_COLOR",
        "FORCE_COLOR",
        "TZ",
        // Color configuration
        "LS_COLORS",
        "LSCOLORS",
        "GREP_COLOR",
        "GREP_COLORS",
        "GCC_COLORS",
        // Display formatting
        "TIME_STYLE",
        "BLOCK_SIZE",
        "BLOCKSIZE",
        // CI indicator
        "CI",
    ]
    .into_iter()
    .collect()
});

/// Advance past horizontal whitespace (space and tab only, NOT newlines).
/// Newlines are command separators in bash; matching across them would be unsafe.
fn skip_hws(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t') {
        i += 1;
    }
    &s[i..]
}

/// Require at least one horizontal whitespace character, then skip all of them.
fn require_hws(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    if bytes.is_empty() || !matches!(bytes[0], b' ' | b'\t') {
        return None;
    }
    Some(skip_hws(s))
}

/// Safe characters for timeout/nice flag values: [A-Za-z0-9_.+-]
/// Rejects $, `, (, ), |, ;, & to prevent injection via flag values.
fn is_safe_flag_value_byte(b: u8) -> bool {
    matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'+' | b'-')
}

/// Safe characters for env var values: [A-Za-z0-9_./:-]
fn is_safe_env_value_byte(b: u8) -> bool {
    matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'/' | b':' | b'-')
}

/// Try to strip one `VAR=value ` env assignment from the start.
/// Only strips if VAR is in SAFE_ENV_VARS and value uses only safe characters.
fn try_strip_env_assignment(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    if bytes.is_empty() || !matches!(bytes[0], b'A'..=b'Z' | b'a'..=b'z' | b'_') {
        return None;
    }

    let mut i = 1;
    while i < bytes.len() && matches!(bytes[i], b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_') {
        i += 1;
    }

    if i >= bytes.len() || bytes[i] != b'=' {
        return None;
    }
    let var_name = &s[..i];
    i += 1;

    if !SAFE_ENV_VARS.contains(var_name) {
        return None;
    }

    let val_start = i;
    while i < bytes.len() && is_safe_env_value_byte(bytes[i]) {
        i += 1;
    }
    if i == val_start {
        return None;
    }

    require_hws(&s[i..])
}

/// Try to strip `timeout [flags] DURATION ` from the start.
///
/// Handles GNU timeout flags: --foreground, --preserve-status, --verbose (no-arg),
/// --kill-after, --signal (with value via = or space), -v (no-arg), -k/-s (with value).
fn try_strip_timeout(s: &str) -> Option<&str> {
    let rest = s.strip_prefix("timeout")?;
    let mut rest = require_hws(rest)?;

    loop {
        let bytes = rest.as_bytes();
        if bytes.is_empty() || bytes[0] != b'-' {
            break;
        }

        // `--` end-of-options marker
        if rest.starts_with("-- ") || rest.starts_with("--\t") {
            rest = skip_hws(&rest[2..]);
            break;
        }

        // No-value long flags
        if let Some(after) = rest
            .strip_prefix("--foreground")
            .or_else(|| rest.strip_prefix("--preserve-status"))
            .or_else(|| rest.strip_prefix("--verbose"))
        {
            rest = require_hws(after)?;
            continue;
        }

        // Value-taking long flags (=fused form)
        if let Some(after) = rest
            .strip_prefix("--kill-after=")
            .or_else(|| rest.strip_prefix("--signal="))
        {
            let bytes = after.as_bytes();
            let mut j = 0;
            while j < bytes.len() && is_safe_flag_value_byte(bytes[j]) {
                j += 1;
            }
            if j == 0 {
                return None;
            }
            rest = require_hws(&after[j..])?;
            continue;
        }

        // Value-taking long flags (space-separated form)
        if let Some(after) = rest
            .strip_prefix("--kill-after")
            .or_else(|| rest.strip_prefix("--signal"))
        {
            let val_s = require_hws(after)?;
            let bytes = val_s.as_bytes();
            let mut j = 0;
            while j < bytes.len() && is_safe_flag_value_byte(bytes[j]) {
                j += 1;
            }
            if j == 0 {
                return None;
            }
            rest = require_hws(&val_s[j..])?;
            continue;
        }

        // Short flag: -v (no value)
        if rest.starts_with("-v") && rest.len() > 2 && matches!(rest.as_bytes()[2], b' ' | b'\t') {
            rest = require_hws(&rest[2..])?;
            continue;
        }

        // Short flags: -k, -s (with value, fused or space-separated)
        if rest.len() >= 2
            && rest.as_bytes()[0] == b'-'
            && matches!(rest.as_bytes()[1], b'k' | b's')
        {
            let after_flag = &rest[2..];
            if !after_flag.is_empty() && !matches!(after_flag.as_bytes()[0], b' ' | b'\t') {
                let bytes = after_flag.as_bytes();
                let mut j = 0;
                while j < bytes.len() && is_safe_flag_value_byte(bytes[j]) {
                    j += 1;
                }
                if j == 0 {
                    return None;
                }
                rest = require_hws(&after_flag[j..])?;
            } else {
                let val_s = require_hws(after_flag)?;
                let bytes = val_s.as_bytes();
                let mut j = 0;
                while j < bytes.len() && is_safe_flag_value_byte(bytes[j]) {
                    j += 1;
                }
                if j == 0 {
                    return None;
                }
                rest = require_hws(&val_s[j..])?;
            }
            continue;
        }

        // Unknown flag — fail closed
        return None;
    }

    // Duration: \d+(\.\d+)?[smhd]?
    let bytes = rest.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 {
        return None;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i < bytes.len() && matches!(bytes[i], b's' | b'm' | b'h' | b'd') {
        i += 1;
    }

    require_hws(&rest[i..])
}

/// Try to strip `time [--] ` from the start.
fn try_strip_time(s: &str) -> Option<&str> {
    let rest = s.strip_prefix("time")?;
    let rest = require_hws(rest)?;
    if let Some(after) = rest.strip_prefix("--")
        && let Some(r) = require_hws(after)
    {
        return Some(r);
    }
    Some(rest)
}

/// Try to strip `nice [-n N | -N] [--] ` from the start.
fn try_strip_nice(s: &str) -> Option<&str> {
    let rest = s.strip_prefix("nice")?;
    let rest = require_hws(rest)?;

    let rest =
        if rest.starts_with("-n") && rest.len() > 2 && matches!(rest.as_bytes()[2], b' ' | b'\t') {
            // `-n N` form
            let after_n = require_hws(&rest[2..])?;
            let bytes = after_n.as_bytes();
            let mut i = 0;
            if i < bytes.len() && bytes[i] == b'-' {
                i += 1;
            }
            let digit_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i == digit_start {
                return None;
            }
            require_hws(&after_n[i..])?
        } else if rest.starts_with('-') && rest.len() > 1 && rest.as_bytes()[1].is_ascii_digit() {
            // Legacy `-N` form
            let bytes = rest.as_bytes();
            let mut i = 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            require_hws(&rest[i..])?
        } else {
            // Bare `nice cmd`
            rest
        };

    if let Some(after) = rest.strip_prefix("--")
        && let Some(r) = require_hws(after)
    {
        return Some(r);
    }
    Some(rest)
}

/// Try to strip `nohup [--] ` from the start.
fn try_strip_nohup(s: &str) -> Option<&str> {
    let rest = s.strip_prefix("nohup")?;
    let rest = require_hws(rest)?;
    if let Some(after) = rest.strip_prefix("--")
        && let Some(r) = require_hws(after)
    {
        return Some(r);
    }
    Some(rest)
}

/// Try to strip `stdbuf (-[ioe][LN0-9]+)+ [--] ` from the start.
fn try_strip_stdbuf(s: &str) -> Option<&str> {
    let rest = s.strip_prefix("stdbuf")?;
    let mut rest = require_hws(rest)?;

    let mut found_flag = false;
    loop {
        let bytes = rest.as_bytes();
        if bytes.len() < 3 || bytes[0] != b'-' || !matches!(bytes[1], b'i' | b'o' | b'e') {
            break;
        }
        let mut i = 2;
        while i < bytes.len() && matches!(bytes[i], b'L' | b'N' | b'0'..=b'9') {
            i += 1;
        }
        if i == 2 {
            break;
        }
        found_flag = true;
        rest = require_hws(&rest[i..])?;
    }

    if !found_flag {
        return None;
    }

    if let Some(after) = rest.strip_prefix("--")
        && let Some(r) = require_hws(after)
    {
        return Some(r);
    }
    Some(rest)
}

/// Strip safe wrapper commands and env var assignments to reveal the actual
/// command being executed.
///
/// Normalizes for pattern matching:
/// - `timeout 5 npm run build` -> `npm run build`
/// - `GOOS=linux go build` -> `go build`
/// - `nohup nice -n 10 cmd` -> `cmd`
/// - `time cmd` -> `cmd`
/// - `stdbuf -o0 cmd` -> `cmd`
///
/// Two-phase approach (matching real shell semantics):
/// - Phase 1: Strip env vars (bash treats `VAR=val cmd` as assignment)
/// - Phase 2: Strip wrappers (wrappers use execvp, so VAR=val after a wrapper
///   is a command to execute, NOT an env assignment)
pub fn strip_safe_wrappers(command: &str) -> &str {
    let mut s = command;

    // Phase 1: Strip leading safe env var assignments
    loop {
        let prev_len = s.len();
        s = skip_hws(s);
        if let Some(rest) = try_strip_env_assignment(s) {
            s = rest;
        }
        if s.len() == prev_len {
            break;
        }
    }

    // Phase 2: Strip wrapper commands (timeout, time, nice, nohup, stdbuf)
    loop {
        let prev_len = s.len();
        s = skip_hws(s);
        if let Some(rest) = try_strip_timeout(s) {
            s = rest;
        } else if let Some(rest) = try_strip_time(s) {
            s = rest;
        } else if let Some(rest) = try_strip_nice(s) {
            s = rest;
        } else if let Some(rest) = try_strip_nohup(s) {
            s = rest;
        } else if let Some(rest) = try_strip_stdbuf(s) {
            s = rest;
        }
        if s.len() == prev_len {
            break;
        }
    }

    s.trim()
}

// ---------------------------------------------------------------------------
// Validation context
// ---------------------------------------------------------------------------

/// Shared context passed to each validator, containing pre-computed views.
struct ValidationContext<'a> {
    /// The original command string.
    original: &'a str,
    /// Command after stripping safe wrappers and env vars.
    stripped: &'a str,
    /// Content with single-quoted sections removed (double-quoted preserved).
    unquoted_content: String,
    /// Content with both single and double quoted sections removed.
    fully_unquoted: String,
    /// Tree-sitter AST analysis (None if parser unavailable or command too long).
    ast: Option<bash_ast::AstAnalysis>,
}

impl<'a> ValidationContext<'a> {
    fn new(command: &'a str) -> Self {
        let extracted = extract_quoted_content(command);
        let ast = bash_ast::analyze(command);
        let stripped = strip_safe_wrappers(command);
        Self {
            original: command,
            stripped,
            unquoted_content: extracted.with_double_quotes,
            fully_unquoted: extracted.fully_unquoted,
            ast,
        }
    }
}

// ---------------------------------------------------------------------------
// Compound command splitting
// ---------------------------------------------------------------------------

/// Split a command string on shell operators (`&&`, `||`, `;`, `|`)
/// while respecting single quotes, double quotes, backticks, `$(...)`,
/// `${...}`, and backslash escapes.
///
/// Each returned element is a trimmed subcommand string.
pub fn split_compound_command(command: &str) -> Vec<&str> {
    let bytes = command.as_bytes();
    let len = bytes.len();
    let mut parts: Vec<&str> = Vec::new();
    let mut start = 0;
    let mut i = 0;

    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut in_backtick = false;
    let mut paren_depth: u32 = 0;
    let mut brace_depth: u32 = 0;

    while i < len {
        let ch = bytes[i];

        if ch == b'\\' && !in_single_quote && i + 1 < len {
            i += 2;
            continue;
        }

        if ch == b'\'' && !in_double_quote && !in_backtick {
            in_single_quote = !in_single_quote;
            i += 1;
            continue;
        }

        if in_single_quote {
            i += 1;
            continue;
        }

        if ch == b'"' && !in_backtick {
            in_double_quote = !in_double_quote;
            i += 1;
            continue;
        }

        if ch == b'`' {
            in_backtick = !in_backtick;
            i += 1;
            continue;
        }

        if ch == b'$' && i + 1 < len {
            let next = bytes[i + 1];
            if next == b'(' {
                paren_depth += 1;
                i += 2;
                continue;
            }
            if next == b'{' {
                brace_depth += 1;
                i += 2;
                continue;
            }
        }

        if ch == b')' && paren_depth > 0 {
            paren_depth -= 1;
            i += 1;
            continue;
        }
        if ch == b'}' && brace_depth > 0 {
            brace_depth -= 1;
            i += 1;
            continue;
        }

        if in_double_quote || in_backtick || paren_depth > 0 || brace_depth > 0 {
            i += 1;
            continue;
        }

        // Operator detection
        if ch == b'&' && i + 1 < len && bytes[i + 1] == b'&' {
            push_trimmed(&mut parts, &command[start..i]);
            i += 2;
            start = i;
            continue;
        }
        if ch == b'|' && i + 1 < len && bytes[i + 1] == b'|' {
            push_trimmed(&mut parts, &command[start..i]);
            i += 2;
            start = i;
            continue;
        }
        if ch == b'|' {
            push_trimmed(&mut parts, &command[start..i]);
            i += 1;
            start = i;
            continue;
        }
        if ch == b';' {
            push_trimmed(&mut parts, &command[start..i]);
            i += 1;
            start = i;
            continue;
        }

        i += 1;
    }

    push_trimmed(&mut parts, &command[start..]);
    parts
}

fn push_trimmed<'a>(parts: &mut Vec<&'a str>, s: &'a str) {
    let t = s.trim();
    if !t.is_empty() {
        parts.push(t);
    }
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

/// Block control characters that bash silently drops but that can hide
/// metacharacters from our validators.
fn validate_control_characters(ctx: &ValidationContext) -> SecurityVerdict {
    for &b in ctx.original.as_bytes() {
        if matches!(b, 0x00..=0x08 | 0x0B..=0x0C | 0x0E..=0x1F | 0x7F) {
            return SecurityVerdict::Block(
                "Command contains non-printable control characters that could bypass security checks"
                    .into(),
            );
        }
    }
    SecurityVerdict::Allow
}

/// Detect incomplete command fragments that look like continuations.
fn validate_incomplete_commands(ctx: &ValidationContext) -> SecurityVerdict {
    let trimmed = ctx.original.trim();
    if trimmed.starts_with('\t') {
        return SecurityVerdict::Warn(
            "Command appears to be an incomplete fragment (starts with tab)".into(),
        );
    }
    if trimmed.starts_with('-') {
        return SecurityVerdict::Warn(
            "Command appears to be an incomplete fragment (starts with flags)".into(),
        );
    }
    if trimmed.starts_with("&&")
        || trimmed.starts_with("||")
        || trimmed.starts_with(';')
        || trimmed.starts_with('>')
        || trimmed.starts_with('<')
    {
        return SecurityVerdict::Warn(
            "Command appears to be a continuation line (starts with operator)".into(),
        );
    }
    SecurityVerdict::Allow
}

/// Detect command substitution and process substitution patterns.
///
/// Checks `unquoted_content` (single-quoted sections stripped, double-quoted
/// preserved) because `$()`, backticks, `${}` expand inside double quotes.
fn validate_dangerous_patterns(ctx: &ValidationContext) -> SecurityVerdict {
    let content = &ctx.unquoted_content;

    // Check for unescaped backticks
    if has_unescaped_char(content, b'`') {
        return SecurityVerdict::Warn(
            "[security notice] Command contains backticks (`) for command substitution".into(),
        );
    }

    struct Pattern {
        needle: &'static str,
        message: &'static str,
    }

    const PATTERNS: &[Pattern] = &[
        Pattern {
            needle: "<(",
            message: "process substitution <()",
        },
        Pattern {
            needle: ">(",
            message: "process substitution >()",
        },
        Pattern {
            needle: "=(",
            message: "Zsh process substitution =()",
        },
        Pattern {
            needle: "$(",
            message: "$() command substitution",
        },
        Pattern {
            needle: "${",
            message: "${} parameter substitution",
        },
        Pattern {
            needle: "$[",
            message: "$[] legacy arithmetic expansion",
        },
    ];

    for p in PATTERNS {
        if content.contains(p.needle) {
            return SecurityVerdict::Warn(format!(
                "[security notice] Command contains {}",
                p.message
            ));
        }
    }

    SecurityVerdict::Allow
}

/// Detect variables in dangerous positions (adjacent to pipes or redirections).
fn validate_dangerous_variables(ctx: &ValidationContext) -> SecurityVerdict {
    let content = &ctx.fully_unquoted;

    let has_var_near_redirect = {
        let bytes = content.as_bytes();
        let len = bytes.len();
        let mut found = false;
        for i in 0..len {
            if matches!(bytes[i], b'<' | b'>' | b'|') {
                // Look ahead for $VAR
                let rest = &content[i..];
                if rest.len() > 2 {
                    let after_ws = rest
                        .trim_start_matches(|c: char| c.is_ascii_whitespace() || "<>|".contains(c));
                    if after_ws.starts_with('$')
                        && after_ws
                            .as_bytes()
                            .get(1)
                            .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
                    {
                        found = true;
                        break;
                    }
                }
                // Look behind for $VAR
                if i > 0 {
                    let before = &content[..i];
                    let before_ws = before.trim_end();
                    if !before_ws.is_empty() {
                        let last_word = before_ws
                            .rsplit_once(char::is_whitespace)
                            .map_or(before_ws, |t| t.1);
                        if last_word.starts_with('$')
                            && last_word
                                .as_bytes()
                                .get(1)
                                .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
                        {
                            found = true;
                            break;
                        }
                    }
                }
            }
        }
        found
    };

    if has_var_near_redirect {
        return SecurityVerdict::Warn(
            "[security notice] Command contains variables in dangerous contexts (redirections or pipes)"
                .into(),
        );
    }
    SecurityVerdict::Allow
}

/// Detect redirections in unquoted context.
fn validate_redirections(ctx: &ValidationContext) -> SecurityVerdict {
    let content = &ctx.fully_unquoted;

    if content.contains('<') {
        return SecurityVerdict::Warn(
            "[security notice] Command contains input redirection (<) which could read sensitive files"
                .into(),
        );
    }
    if content.contains('>') {
        return SecurityVerdict::Warn(
            "[security notice] Command contains output redirection (>) which could write to arbitrary files"
                .into(),
        );
    }
    SecurityVerdict::Allow
}

/// Detect IFS variable usage that could bypass validation.
fn validate_ifs_injection(ctx: &ValidationContext) -> SecurityVerdict {
    if ctx.original.contains("$IFS") || ctx.original.contains("${IFS") {
        return SecurityVerdict::Warn(
            "[security notice] Command contains IFS variable usage which could bypass security validation"
                .into(),
        );
    }
    SecurityVerdict::Allow
}

/// Detect /proc/*/environ access.
fn validate_proc_environ(ctx: &ValidationContext) -> SecurityVerdict {
    if contains_pattern(ctx.original, "/proc/", "/environ") {
        return SecurityVerdict::Warn(
            "[security notice] Command accesses /proc/*/environ which could expose sensitive environment variables"
                .into(),
        );
    }
    SecurityVerdict::Allow
}

/// Detect unquoted newlines that could separate multiple commands.
fn validate_newlines(ctx: &ValidationContext) -> SecurityVerdict {
    let content = &ctx.fully_unquoted;

    if !content.contains('\n') && !content.contains('\r') {
        return SecurityVerdict::Allow;
    }

    let bytes = content.as_bytes();
    for i in 0..bytes.len() {
        if matches!(bytes[i], b'\n' | b'\r') {
            // Check if followed by non-whitespace (looks like a command)
            if let Some(rest) = content.get(i + 1..) {
                let after_ws = rest.trim_start();
                if !after_ws.is_empty() {
                    // Allow backslash-newline continuation at word boundary
                    if i > 0 && bytes[i - 1] == b'\\' {
                        let before_bs = if i >= 2 { bytes[i - 2] } else { b' ' };
                        if before_bs.is_ascii_whitespace() {
                            continue;
                        }
                    }
                    return SecurityVerdict::Warn(
                        "[security notice] Command contains newlines that could separate multiple commands"
                            .into(),
                    );
                }
            }
        }
    }
    SecurityVerdict::Allow
}

/// Detect backslash-escaped whitespace outside quotes.
fn validate_backslash_escaped_whitespace(ctx: &ValidationContext) -> SecurityVerdict {
    let bytes = ctx.original.as_bytes();
    let len = bytes.len();
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut i = 0;

    while i < len {
        let ch = bytes[i];

        if ch == b'\\' && !in_single_quote {
            if !in_double_quote && i + 1 < len {
                let next = bytes[i + 1];
                if next == b' ' || next == b'\t' {
                    return SecurityVerdict::Warn(
                        "[security notice] Command contains backslash-escaped whitespace that could alter command parsing"
                            .into(),
                    );
                }
            }
            i += 2;
            continue;
        }

        if ch == b'"' && !in_single_quote {
            in_double_quote = !in_double_quote;
        } else if ch == b'\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
        }
        i += 1;
    }
    SecurityVerdict::Allow
}

/// Detect backslash-escaped shell operators outside quotes.
fn validate_backslash_escaped_operators(ctx: &ValidationContext) -> SecurityVerdict {
    let bytes = ctx.original.as_bytes();
    let len = bytes.len();
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut i = 0;

    while i < len {
        let ch = bytes[i];

        // Handle backslash first (before quote toggles)
        if ch == b'\\' && !in_single_quote {
            if !in_double_quote && i + 1 < len {
                let next = bytes[i + 1];
                if matches!(next, b';' | b'|' | b'&' | b'<' | b'>') {
                    return SecurityVerdict::Warn(
                        "[security notice] Command contains a backslash before a shell operator which can hide command structure"
                            .into(),
                    );
                }
            }
            i += 2;
            continue;
        }

        if ch == b'\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
        } else if ch == b'"' && !in_single_quote {
            in_double_quote = !in_double_quote;
        }
        i += 1;
    }
    SecurityVerdict::Allow
}

/// Detect brace expansion syntax in unquoted context.
fn validate_brace_expansion(ctx: &ValidationContext) -> SecurityVerdict {
    let content = &ctx.fully_unquoted;
    let bytes = content.as_bytes();

    for i in 0..bytes.len() {
        if bytes[i] != b'{' || is_escaped_at(bytes, i) {
            continue;
        }

        // Find matching unescaped `}` tracking nesting depth
        let mut depth: u32 = 1;
        let mut j = i + 1;
        let mut matching_close = None;
        while j < bytes.len() && depth > 0 {
            if bytes[j] == b'{' && !is_escaped_at(bytes, j) {
                depth += 1;
            } else if bytes[j] == b'}' && !is_escaped_at(bytes, j) {
                depth -= 1;
                if depth == 0 {
                    matching_close = Some(j);
                }
            }
            j += 1;
        }

        let Some(close) = matching_close else {
            continue;
        };

        // Check for `,` or `..` at outer nesting level between { and }
        let mut inner_depth: u32 = 0;
        for k in (i + 1)..close {
            if bytes[k] == b'{' && !is_escaped_at(bytes, k) {
                inner_depth += 1;
            } else if bytes[k] == b'}' && !is_escaped_at(bytes, k) {
                inner_depth = inner_depth.saturating_sub(1);
            } else if inner_depth == 0
                && (bytes[k] == b',' || (bytes[k] == b'.' && k + 1 < close && bytes[k + 1] == b'.'))
            {
                return SecurityVerdict::Warn(
                        "[security notice] Command contains brace expansion that could alter command parsing"
                            .into(),
                    );
            }
        }
    }
    SecurityVerdict::Allow
}

/// Detect Unicode whitespace characters that could cause parsing inconsistencies.
fn validate_unicode_whitespace(ctx: &ValidationContext) -> SecurityVerdict {
    for ch in ctx.original.chars() {
        if matches!(
            ch,
            '\u{00A0}' | '\u{1680}' | '\u{2000}'
                ..='\u{200A}'
                    | '\u{2028}'
                    | '\u{2029}'
                    | '\u{202F}'
                    | '\u{205F}'
                    | '\u{3000}'
                    | '\u{FEFF}'
        ) {
            return SecurityVerdict::Warn(
                "[security notice] Command contains Unicode whitespace characters that could cause parsing inconsistencies"
                    .into(),
            );
        }
    }
    SecurityVerdict::Allow
}

/// Detect ANSI-C quoting ($'...') and locale quoting ($"...") which can
/// encode arbitrary characters via escape sequences.
fn validate_ansi_c_quoting(ctx: &ValidationContext) -> SecurityVerdict {
    let cmd = ctx.original;
    let bytes = cmd.as_bytes();

    for i in 0..bytes.len().saturating_sub(2) {
        if bytes[i] == b'$' && (bytes[i + 1] == b'\'' || bytes[i + 1] == b'"') {
            let quote = bytes[i + 1];
            // Look for closing quote
            if bytes[i + 2..].contains(&quote) {
                let msg = if quote == b'\'' {
                    "[security notice] Command contains ANSI-C quoting ($'...') which can hide characters"
                } else {
                    "[security notice] Command contains locale quoting ($\"...\") which can hide characters"
                };
                return SecurityVerdict::Warn(msg.into());
            }
        }
    }
    SecurityVerdict::Allow
}

/// Detect Zsh-specific dangerous commands that can bypass security checks.
fn validate_zsh_dangerous_commands(ctx: &ValidationContext) -> SecurityVerdict {
    const ZSH_DANGEROUS: &[&str] = &[
        "zmodload", "emulate", "sysopen", "sysread", "syswrite", "sysseek", "zpty", "ztcp",
        "zsocket", "zf_rm", "zf_mv", "zf_ln", "zf_chmod", "zf_chown", "zf_mkdir", "zf_rmdir",
        "zf_chgrp",
    ];

    // Use the stripped command (env vars and safe wrappers already removed)
    // to find the real base command
    let base = ctx
        .stripped
        .split_whitespace()
        .find(|w| !matches!(*w, "command" | "builtin" | "noglob" | "nocorrect"))
        .unwrap_or("");

    if ZSH_DANGEROUS.contains(&base) {
        return SecurityVerdict::Warn(format!(
            "[security notice] Command uses Zsh-specific '{}' which can bypass security checks",
            base
        ));
    }

    // fc -e is an eval-equivalent
    if base == "fc" && ctx.stripped.contains("-e") {
        return SecurityVerdict::Warn(
            "[security notice] Command uses 'fc -e' which can execute arbitrary commands via editor"
                .into(),
        );
    }

    SecurityVerdict::Allow
}

/// Detect eval keyword at word boundary (outside single quotes).
/// Checked with quote awareness so `'eval ...'` is not flagged.
fn validate_eval(ctx: &ValidationContext) -> SecurityVerdict {
    let bytes = ctx.original.as_bytes();
    let len = bytes.len();
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut i = 0;

    while i < len {
        let ch = bytes[i];

        if ch == b'\\' && !in_single_quote && i + 1 < len {
            i += 2;
            continue;
        }
        if ch == b'\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
            i += 1;
            continue;
        }
        if in_single_quote {
            i += 1;
            continue;
        }
        if ch == b'"' {
            in_double_quote = !in_double_quote;
            i += 1;
            continue;
        }

        if !in_double_quote && ch == b'e' && is_eval_at(bytes, i) {
            return SecurityVerdict::Warn(
                "[security notice] eval keyword detected — commands may be constructed dynamically"
                    .into(),
            );
        }

        i += 1;
    }
    SecurityVerdict::Allow
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// True when `bytes[pos..]` starts with `eval` at a word boundary.
fn is_eval_at(bytes: &[u8], pos: usize) -> bool {
    if pos + 4 > bytes.len() {
        return false;
    }
    if &bytes[pos..pos + 4] != b"eval" {
        return false;
    }
    if pos > 0 && (bytes[pos - 1] as char).is_alphanumeric() {
        return false;
    }
    if pos + 4 < bytes.len() {
        let next = bytes[pos + 4];
        if (next as char).is_alphanumeric() || next == b'_' {
            return false;
        }
    }
    true
}

/// True when an unescaped occurrence of `char_byte` exists in `content`.
fn has_unescaped_char(content: &str, char_byte: u8) -> bool {
    let bytes = content.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if bytes[i] == char_byte {
            return true;
        }
        i += 1;
    }
    false
}

/// True when the character at `pos` is escaped by an odd number of preceding backslashes.
fn is_escaped_at(bytes: &[u8], pos: usize) -> bool {
    let mut count = 0u32;
    let mut i = pos;
    while i > 0 {
        i -= 1;
        if bytes[i] == b'\\' {
            count += 1;
        } else {
            break;
        }
    }
    count % 2 == 1
}

/// Check if `haystack` contains the pattern `prefix...suffix` (with anything between).
fn contains_pattern(haystack: &str, prefix: &str, suffix: &str) -> bool {
    if let Some(start) = haystack.find(prefix) {
        let rest = &haystack[start + prefix.len()..];
        return rest.contains(suffix);
    }
    false
}

fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

// ---------------------------------------------------------------------------
// AST-based validators (tree-sitter)
// ---------------------------------------------------------------------------

/// Use the tree-sitter AST to detect parse errors (malformed syntax).
/// Malformed commands could exploit parser differentials.
fn validate_ast_parse_errors(ctx: &ValidationContext) -> SecurityVerdict {
    if let Some(ref ast) = ctx.ast
        && ast.has_parse_errors
    {
        return SecurityVerdict::Warn(
            "[security notice] Command contains malformed syntax that could be misinterpreted"
                .into(),
        );
    }
    SecurityVerdict::Allow
}

/// Use the AST to detect command substitution, process substitution, and
/// parameter expansion with structural accuracy (no regex false positives).
fn validate_ast_dangerous_patterns(ctx: &ValidationContext) -> SecurityVerdict {
    let Some(ref ast) = ctx.ast else {
        return SecurityVerdict::Allow;
    };

    let mut warnings: Vec<&str> = Vec::new();

    if ast.dangerous.has_command_substitution {
        warnings.push("command substitution ($() or backticks)");
    }
    if ast.dangerous.has_process_substitution {
        warnings.push("process substitution (<() or >())");
    }
    if ast.dangerous.has_parameter_expansion {
        warnings.push("parameter expansion (${...})");
    }

    if warnings.is_empty() {
        SecurityVerdict::Allow
    } else {
        SecurityVerdict::Warn(format!(
            "[security notice] AST analysis detected: {}",
            warnings.join(", ")
        ))
    }
}

/// Use the AST to detect subshells and command groups, which can hide
/// dangerous operations inside structural constructs.
fn validate_ast_compound_structure(ctx: &ValidationContext) -> SecurityVerdict {
    let Some(ref ast) = ctx.ast else {
        return SecurityVerdict::Allow;
    };

    if ast.compound.has_subshell {
        return SecurityVerdict::Warn(
            "[security notice] Command contains subshell (...) which executes in a child process"
                .into(),
        );
    }
    if ast.compound.has_command_group {
        return SecurityVerdict::Warn(
            "[security notice] Command contains command group {...} which groups operations".into(),
        );
    }
    SecurityVerdict::Allow
}

/// When the AST confirms NO actual operator nodes exist, suppress the
/// backslash-escaped-operator warning (eliminates `find -exec \;` false positive).
fn validate_backslash_escaped_operators_with_ast(ctx: &ValidationContext) -> SecurityVerdict {
    if let Some(ref ast) = ctx.ast
        && !ast.has_actual_operators
    {
        return SecurityVerdict::Allow;
    }
    validate_backslash_escaped_operators(ctx)
}

// ---------------------------------------------------------------------------
// Comprehensive security check — the pipeline
// ---------------------------------------------------------------------------

/// Run all security checks on a command.
///
/// Pipeline:
/// 1. Early blockers (control chars)
/// 2. Dangerous pattern check via caller-provided function (full + per-subcommand)
/// 3. Validator pipeline collecting the most severe verdict
pub fn check_command(
    command: &str,
    check_dangerous: fn(&str) -> Option<&'static str>,
) -> SecurityVerdict {
    check_command_inner(command, check_dangerous).verdict
}

/// Full classifier output for one command: the verdict plus whether the
/// tree-sitter AST layer actually ran. When it did not (command longer
/// than the AST limit, parser failure) the AST-only validators were
/// silently skipped, so an `Allow` is weaker than usual; the approval
/// gate uses this to fail closed in prompting modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandClassification {
    pub verdict: SecurityVerdict,
    pub ast_available: bool,
    /// Every warning the validators emitted, attributed to its validator
    /// (ADR 09 attribution payload). Empty unless `verdict` is `Warn`.
    pub warnings: Vec<ClassifiedWarning>,
    /// The command's redirect structure from the AST (`None` without it).
    /// Context-free, so it is safe in the process-global memo; whether a
    /// warning is absorbed depends on the conversation and is never cached.
    pub redirects: Option<bash_ast::RedirectAnalysis>,
    /// Some command's name is not a plain literal word (AST only), so the
    /// command may run `cd` in a spelling no word scan finds.
    pub unreadable_command_name: bool,
}

/// Which validator emitted a warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningKind {
    /// `validate_redirections` (any `<` / `>` outside quotes).
    Redirection,
    /// `validate_newlines` (a newline that could separate commands).
    Newline,
    /// Every other validator: never absorbed.
    Other,
}

/// One validator warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifiedWarning {
    pub kind: WarningKind,
    pub message: String,
}

/// The pipeline's merged message for a set of warnings: the first, plus a
/// count of the rest.
fn merged_warning(messages: &[&str]) -> Option<String> {
    let first = messages.first()?;
    Some(if messages.len() > 1 {
        format!("{}; (+{} more warning(s))", first, messages.len() - 1)
    } else {
        first.to_string()
    })
}

/// Classify a command with the standard dangerous-pattern list, memoized.
///
/// The approval gate and the bash tool both need the verdict for the same
/// command within one tool call; the memo makes the 18-validator pipeline
/// run once. The classifier is a pure function of the command text, so
/// sharing results is sound.
pub fn classify_command(command: &str) -> CommandClassification {
    static CACHE: std::sync::LazyLock<ClassificationCache> =
        std::sync::LazyLock::new(ClassificationCache::default);
    CACHE.get_or_classify(command, |c| check_command_inner(c, check_dangerous))
}

/// Number of recent classifications kept by [`classify_command`].
const CLASSIFICATION_CACHE_CAPACITY: usize = 32;

/// Small FIFO memo of recent command classifications.
#[derive(Default)]
pub struct ClassificationCache {
    entries: std::sync::Mutex<std::collections::VecDeque<(String, CommandClassification)>>,
}

impl ClassificationCache {
    /// Return the cached classification for `command`, or run `classify`
    /// and remember the result.
    pub fn get_or_classify(
        &self,
        command: &str,
        classify: impl FnOnce(&str) -> CommandClassification,
    ) -> CommandClassification {
        if let Some((_, hit)) = self
            .entries
            .lock()
            .unwrap()
            .iter()
            .find(|(key, _)| key == command)
        {
            return hit.clone();
        }
        // Classify outside the lock: the pipeline is the slow part.
        let result = classify(command);
        let mut entries = self.entries.lock().unwrap();
        if !entries.iter().any(|(key, _)| key == command) {
            if entries.len() >= CLASSIFICATION_CACHE_CAPACITY {
                entries.pop_front();
            }
            entries.push_back((command.to_string(), result.clone()));
        }
        result
    }
}

fn check_command_inner(
    command: &str,
    check_dangerous: fn(&str) -> Option<&'static str>,
) -> CommandClassification {
    let blocked = |reason: String| CommandClassification {
        verdict: SecurityVerdict::Block(reason),
        ast_available: true,
        warnings: Vec::new(),
        redirects: None,
        unreadable_command_name: false,
    };
    // 1. Check the full command for dangerous patterns.
    if let Some(msg) = check_dangerous(command) {
        return blocked(msg.to_string());
    }

    // 1b. Also check after stripping safe wrappers (e.g. `timeout 5 rm -rf /`).
    let stripped = strip_safe_wrappers(command);
    if stripped != command
        && let Some(msg) = check_dangerous(stripped)
    {
        return blocked(format!(
            "{} (after normalizing wrappers: {})",
            msg,
            truncate_for_log(stripped, 80),
        ));
    }

    // 2. Split into subcommands and check each one.
    let subs = split_compound_command(command);
    for sub in &subs {
        if let Some(msg) = check_dangerous(sub) {
            return blocked(format!(
                "{} (in subcommand: {})",
                msg,
                truncate_for_log(sub, 80),
            ));
        }

        let stripped_sub = strip_safe_wrappers(sub);
        if stripped_sub != *sub
            && let Some(msg) = check_dangerous(stripped_sub)
        {
            return blocked(format!(
                "{} (in subcommand after normalizing wrappers: {})",
                msg,
                truncate_for_log(stripped_sub, 80),
            ));
        }
    }

    // 3. Build validation context and run the validator pipeline.
    let ctx = ValidationContext::new(command);
    let ast_available = ctx.ast.is_some();
    let redirects = ctx.ast.as_ref().map(|ast| ast.redirects.clone());
    let unreadable_command_name = ctx
        .ast
        .as_ref()
        .is_some_and(|ast| ast.unreadable_command_name);

    // Each validator returns Allow (continue), Warn (collect), or Block (stop).
    // AST-based validators run alongside regex validators for defense-in-depth.
    type Validator = fn(&ValidationContext) -> SecurityVerdict;
    let validators: &[(Validator, WarningKind)] = &[
        (validate_control_characters, WarningKind::Other),
        (validate_ast_parse_errors, WarningKind::Other),
        (validate_incomplete_commands, WarningKind::Other),
        (validate_ansi_c_quoting, WarningKind::Other),
        (validate_dangerous_patterns, WarningKind::Other),
        (validate_ast_dangerous_patterns, WarningKind::Other),
        (validate_ast_compound_structure, WarningKind::Other),
        (validate_dangerous_variables, WarningKind::Other),
        (validate_newlines, WarningKind::Newline),
        (validate_ifs_injection, WarningKind::Other),
        (validate_proc_environ, WarningKind::Other),
        (validate_redirections, WarningKind::Redirection),
        (validate_backslash_escaped_whitespace, WarningKind::Other),
        // Uses AST to skip false positive on `find -exec \;`
        (
            validate_backslash_escaped_operators_with_ast,
            WarningKind::Other,
        ),
        (validate_unicode_whitespace, WarningKind::Other),
        (validate_brace_expansion, WarningKind::Other),
        (validate_zsh_dangerous_commands, WarningKind::Other),
        (validate_eval, WarningKind::Other),
    ];

    let mut warnings: Vec<ClassifiedWarning> = Vec::new();
    for (validator, kind) in validators {
        match validator(&ctx) {
            SecurityVerdict::Block(msg) => {
                return CommandClassification {
                    verdict: SecurityVerdict::Block(msg),
                    ast_available,
                    warnings: Vec::new(),
                    redirects,
                    unreadable_command_name,
                };
            }
            SecurityVerdict::Warn(message) => warnings.push(ClassifiedWarning {
                kind: *kind,
                message,
            }),
            SecurityVerdict::Allow => {}
        }
    }

    let messages: Vec<&str> = warnings.iter().map(|w| w.message.as_str()).collect();
    let verdict = match merged_warning(&messages) {
        Some(message) => SecurityVerdict::Warn(message),
        None => SecurityVerdict::Allow,
    };
    CommandClassification {
        verdict,
        ast_available,
        warnings,
        redirects,
        unreadable_command_name,
    }
}

// ---------------------------------------------------------------------------
// Dangerous command patterns (shared with MCP argument scanning)
// ---------------------------------------------------------------------------

/// Patterns checked via simple `contains` (no boundary logic needed).
const SIMPLE_DANGEROUS_PATTERNS: &[(&str, &str)] = &[
    ("mkfs.", "Blocked: filesystem formatting command"),
    (":(){:|:&};:", "Blocked: fork bomb detected"),
    (":(){ :|:& };:", "Blocked: fork bomb detected"),
    ("> /dev/sda", "Blocked: direct write to block device"),
];

/// Patterns where we must verify the char after the trailing `/` to avoid
/// false positives like `rm -rf /tmp/mydir`.
const ROOT_PATH_PATTERNS: &[(&str, &str)] = &[
    ("rm -rf /", "Blocked: recursive deletion of root filesystem"),
    ("rm -fr /", "Blocked: recursive deletion of root filesystem"),
    (
        "chmod -r 777 /",
        "Blocked: recursive permission change on root",
    ),
];

/// Check a command string for known-dangerous patterns.
///
/// Returns `Some(message)` if the command matches a dangerous pattern, or
/// `None` if no match is found.  This function is used by both `BashTool`
/// and MCP argument scanning.
pub fn check_dangerous(command: &str) -> Option<&'static str> {
    let lower = command.to_lowercase();

    for (pattern, message) in ROOT_PATH_PATTERNS {
        if let Some(idx) = lower.find(pattern) {
            let after = idx + pattern.len();
            let is_root = if after >= lower.len() {
                true
            } else {
                let next = lower.as_bytes()[after];
                matches!(
                    next,
                    b'*' | b' ' | b';' | b'|' | b'&' | b')' | b'\n' | b'\t'
                )
            };
            if is_root {
                return Some(message);
            }
        }
    }

    for (pattern, message) in SIMPLE_DANGEROUS_PATTERNS {
        if lower.contains(pattern) {
            return Some(message);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Security context for graduated threshold
// ---------------------------------------------------------------------------

/// Controls the severity threshold when reusing `check_command` in different
/// contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityContext {
    /// The input is a known shell command (bash tool).  Both `Block` and
    /// `Warn` verdicts are propagated.
    Command,
    /// The input might be a command (e.g. an MCP STDIO argument string).
    /// Only `Block` verdicts are propagated; `Warn` is downgraded to `Allow`
    /// to avoid false positives on natural-language text.
    Untrusted,
}

/// Run `check_command` with a graduated severity threshold.
///
/// In `Command` context the full verdict is returned.  In `Untrusted`
/// context only `Block` verdicts survive — `Warn` is treated as `Allow`.
pub fn check_with_context(value: &str, context: SecurityContext) -> SecurityVerdict {
    let verdict = check_command(value, check_dangerous);
    match context {
        SecurityContext::Command => verdict,
        SecurityContext::Untrusted => match verdict {
            SecurityVerdict::Block(msg) => SecurityVerdict::Block(msg),
            _ => SecurityVerdict::Allow,
        },
    }
}

// ---------------------------------------------------------------------------
// Write-target extraction (blocklist, boundary analysis, absorption)
// ---------------------------------------------------------------------------

/// Check every write target of `command` (redirects and common write
/// commands) against `path_policy::check_write`.
///
/// Relative targets are checked against every directory in `cwds` — the
/// directories the command may actually run in (ADR 09 FA-5: the call's
/// effective cwd, the shell-reported session directory, and `$HOME` when
/// the command changes directory). With no `cwds`, relative targets are
/// checked as spelled.
///
/// Returns `Err(reason)` on the first blocklisted target. The workspace
/// boundary is not enforced here: the permission gate owns it, and the
/// OS jail confines sandboxed bash to its writable roots.
pub fn check_write_targets(command: &str, cwds: &[std::path::PathBuf]) -> Result<(), String> {
    for target in write_targets(command) {
        if is_relative_target(&target) && !cwds.is_empty() {
            for cwd in cwds {
                let resolved = resolve_target(&target, cwd.to_str());
                super::path_policy::check_write(&resolved, None)?;
            }
        } else {
            super::path_policy::check_write(&resolve_target(&target, None), None)?;
        }
    }
    Ok(())
}

/// The first write target of `command` that hits a write ask-tier group
/// (credential store, persistence location, agent settings, host data).
/// The bash tool allows these (the gate escalated them); the gate uses
/// this to escalate in prompting modes and to deny in `yolo`.
pub fn write_ask_target(
    command: &str,
    cwd: Option<&str>,
) -> Option<super::path_policy::WriteBlock> {
    let targets = write_targets(command);
    if targets.is_empty() {
        return None;
    }
    let classifier = super::path_policy::ReadClassifier::new();
    targets
        .iter()
        .find_map(|target| classifier.classify_write(&resolve_target(target, cwd)))
}

/// Commands whose operands are deleted irreversibly.
const DELETION_COMMANDS: &[&str] = &["rm", "unlink", "shred", "srm"];

/// Operands of deletion commands (`rm`, `unlink`, `shred`, `srm`) in every
/// segment of `command`, after stripping safe wrappers. Flags are skipped
/// up to `--`; quotes are trimmed.
pub fn extract_deletion_targets(command: &str) -> Vec<String> {
    let mut targets = Vec::new();
    for sub in command_segments(command) {
        let sub = strip_safe_wrappers(sub.trim());
        let mut parts = sub.split_whitespace();
        let Some(first) = parts.next() else {
            continue;
        };
        let base = first.rsplit('/').next().unwrap_or(first);
        if !DELETION_COMMANDS.contains(&base) {
            continue;
        }
        let mut flags_done = false;
        for arg in parts {
            if !flags_done && arg == "--" {
                flags_done = true;
                continue;
            }
            if !flags_done && arg.starts_with('-') {
                continue;
            }
            targets.push(arg.trim_matches(|c| c == '\'' || c == '"').to_string());
        }
    }
    targets
}

/// The existing path a deletion operand would remove, for the guard: the
/// operand itself, or for a glob/expansion operand the directory before
/// the first metacharacter (conservative: that directory's contents are
/// at stake).
fn deletion_subject(target: &str, cwd: Option<&str>) -> std::path::PathBuf {
    let meta = target.find(['*', '?', '[', '$', '{', '`']);
    let literal = match meta {
        Some(index) => match target[..index].rfind('/') {
            Some(slash) => &target[..=slash],
            None => ".",
        },
        None => target,
    };
    resolve_target(literal, cwd)
}

/// Irreversible-deletion guard: the first operand of a deletion command
/// that would remove a path that already exists outside the workspace.
/// Paths inside the workspace, inside `extra_roots` (the conversation
/// scratch directory), and inside the jail's temp channel (`/tmp`) are
/// not reported; paths that do not exist yet have nothing to lose.
/// Without a workspace there is no boundary, so nothing is reported (the
/// gate passes the scratch directory when no workspace is selected).
///
/// The permission gate escalates a hit (prompt in `ask`, judgment in
/// `auto`); `yolo` runs it. The bash tool itself does not refuse.
pub fn outside_deletion_target(
    command: &str,
    workspace: Option<&std::path::Path>,
    cwd: Option<&str>,
    extra_roots: &[std::path::PathBuf],
) -> Option<std::path::PathBuf> {
    let workspace = workspace?;
    let targets = extract_deletion_targets(command);
    if targets.is_empty() {
        return None;
    }
    let cwd = cwd.or_else(|| workspace.to_str());
    let scope = WriteScope::new(workspace, extra_roots);
    for target in &targets {
        let subject = deletion_subject(target, cwd);
        // Only pre-existing paths are at stake (a dangling symlink counts).
        if std::fs::symlink_metadata(&subject).is_err() {
            continue;
        }
        if !scope.contains(&subject, false) {
            return Some(subject);
        }
    }
    None
}

/// Write targets of `command` (redirects, `tee`, `cp`/`mv` destinations,
/// `touch`) that land outside the workspace, `extra_roots` (the scratch
/// directory), and the jail's temp channel. Device files (`/dev/null`)
/// are ignored, and so are targets that cannot be placed statically
/// (expansions, globs). In sandboxed modes these are what the jail would
/// refuse, so the gate escalates them before execution instead. Each is
/// returned where the write would physically land (symlinks resolved).
pub fn outside_write_targets(
    command: &str,
    workspace: Option<&std::path::Path>,
    cwd: Option<&str>,
    extra_roots: &[std::path::PathBuf],
) -> Vec<std::path::PathBuf> {
    let Some(workspace) = workspace else {
        return Vec::new();
    };
    let targets = write_targets(command);
    if targets.is_empty() {
        return Vec::new();
    }
    let cwd = cwd.or_else(|| workspace.to_str());
    let scope = WriteScope::new(workspace, extra_roots);
    let mut outside = Vec::new();
    for target in &targets {
        // `$VAR`, globs, and substitutions cannot be placed statically.
        if target.contains(['$', '*', '?', '`']) {
            continue;
        }
        let spelled = resolve_target(target, cwd);
        if crate::path_policy::normalize_path(&spelled).starts_with("/dev")
            || scope.contains(&spelled, true)
        {
            continue;
        }
        let landed = physical_path(&spelled, true)
            .unwrap_or_else(|| crate::path_policy::normalize_path(&spelled));
        if !outside.contains(&landed) {
            outside.push(landed);
        }
    }
    outside
}

/// The directories a command may write without leaving the user's scope:
/// the workspace and extra roots (themselves included) and the jail's
/// temp channel (`/tmp`, strictly below it) when it is on. Never the
/// analyzing process's own `env::temp_dir()`: the jail does not open it,
/// so counting it would make silent writes die at the jail (INV-7).
struct WriteScope {
    owned: Vec<std::path::PathBuf>,
    temp: Vec<std::path::PathBuf>,
}

impl WriteScope {
    fn new(workspace: &std::path::Path, extra_roots: &[std::path::PathBuf]) -> Self {
        let spellings = |roots: Vec<std::path::PathBuf>| -> Vec<std::path::PathBuf> {
            roots
                .iter()
                .flat_map(|root| {
                    std::iter::once(root.clone()).chain(std::fs::canonicalize(root).ok())
                })
                .collect()
        };
        let mut owned = vec![workspace.to_path_buf()];
        owned.extend(extra_roots.iter().cloned());
        let temp = if crate::sandbox::temp_channel_available(&owned) {
            spellings(
                crate::sandbox::TMP_SPELLINGS
                    .iter()
                    .map(std::path::PathBuf::from)
                    .collect(),
            )
        } else {
            Vec::new()
        };
        Self {
            owned: spellings(owned),
            temp,
        }
    }

    fn inside(&self, path: &std::path::Path) -> bool {
        self.owned.iter().any(|root| path.starts_with(root))
            || self
                .temp
                .iter()
                .any(|root| path.starts_with(root) && path != root.as_path())
    }

    /// Whether a write (or deletion) of `path` stays in scope, both as
    /// spelled and where it physically lands. `follow_final`: a redirect or
    /// write opens *through* a final symlink (a workspace link to `~/f`
    /// writes `~/f`), so writes pass `true`; deletion removes the link
    /// itself, so it passes `false`.
    fn contains(&self, path: &std::path::Path, follow_final: bool) -> bool {
        let lexical = crate::path_policy::normalize_path(path);
        if !self.inside(&lexical) {
            return false;
        }
        physical_path(path, follow_final).is_some_and(|landed| self.inside(&landed))
    }
}

/// Where `path` physically lands: its parent resolved through the
/// filesystem **as spelled** (lexical `..` popping before resolution
/// would move `<ws>/link/../x` with `link -> /outside/dir` back inside),
/// through the deepest existing ancestor when the parent does not exist
/// yet. With `follow_final`, a final symlink is followed as well; a
/// dangling one yields `None` (unplaceable).
fn physical_path(path: &std::path::Path, follow_final: bool) -> Option<std::path::PathBuf> {
    let Some(name) = path.file_name() else {
        return Some(resolve_deepest(path));
    };
    let parent = path.parent().unwrap_or(std::path::Path::new("/"));
    let joined = resolve_deepest(parent).join(name);
    if follow_final
        && std::fs::symlink_metadata(&joined).is_ok_and(|meta| meta.file_type().is_symlink())
    {
        return std::fs::canonicalize(&joined).ok();
    }
    Some(joined)
}

/// `path` canonicalized through its deepest existing ancestor; the
/// missing remainder (which cannot contain symlinks) is normalized
/// lexically.
fn resolve_deepest(path: &std::path::Path) -> std::path::PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }
    let mut existing = path.to_path_buf();
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    while let Some(name) = existing.file_name().map(|n| n.to_os_string()) {
        existing.pop();
        rest.push(name);
        if let Ok(canonical) = std::fs::canonicalize(&existing) {
            let mut out = canonical;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return crate::path_policy::normalize_path(&out);
        }
    }
    crate::path_policy::normalize_path(path)
}

/// Resolve a target path: expand `~`, make relative paths absolute using cwd.
fn resolve_target(target: &str, cwd: Option<&str>) -> std::path::PathBuf {
    let trimmed = target.trim().trim_matches(|c| c == '\'' || c == '"');

    if let Some(rest) = trimmed.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    if trimmed == "~"
        && let Some(home) = dirs::home_dir()
    {
        return home;
    }

    let path = std::path::Path::new(trimmed);
    if path.is_absolute() {
        path.to_path_buf()
    } else if let Some(dir) = cwd {
        std::path::Path::new(dir).join(trimmed)
    } else {
        path.to_path_buf()
    }
}

/// Whether a target resolves against the working directory.
fn is_relative_target(target: &str) -> bool {
    let trimmed = target.trim().trim_matches(|c| c == '\'' || c == '"');
    !(trimmed.starts_with('/') || trimmed == "~" || trimmed.starts_with("~/"))
}

/// Write targets of a command: redirect destinations (from the AST when
/// it is available, which bounds quotes and heredoc bodies exactly; a
/// byte scan otherwise) plus `tee`/`cp`/`mv`/`touch` destinations.
/// Unresolvable redirect destinations are left out.
fn write_targets(command: &str) -> Vec<String> {
    let mut targets = Vec::new();
    match classify_command(command).redirects {
        Some(analysis) => {
            for redirect in &analysis.redirects {
                if redirect.shape.writes_file()
                    && let Some(bash_ast::RedirectTarget::Literal(target)) = &redirect.target
                    && !target.starts_with("/dev/")
                {
                    targets.push(target.clone());
                }
            }
        }
        None => extract_redirect_targets(command, &mut targets),
    }
    extract_command_write_targets(command, &mut targets);
    targets
}

/// Extract write targets from a command string by a byte scan (tests and
/// the non-AST fallback of [`write_targets`]).
///
/// Handles:
/// - Output redirections: `>`, `>>`, `>|`, `2>`, `&>`, `&>>`, `>&file`
/// - `tee` targets
/// - `cp`/`mv` destinations
/// - `touch` arguments
#[cfg(test)]
fn extract_write_targets(command: &str) -> Vec<String> {
    let mut targets = Vec::new();

    extract_redirect_targets(command, &mut targets);
    extract_command_write_targets(command, &mut targets);

    targets
}

/// Byte-scan redirect targets: `> file`, `>> file`, `>| file`, `2> file`,
/// `&> file`, `>&file`. Descriptor duplications (`>&2`, `2>&1`, `>&-`)
/// carry no target; a purely numeric destination of a plain `>` does (`> 2`
/// writes a file named `2`).
fn extract_redirect_targets(command: &str, targets: &mut Vec<String>) {
    let bytes = command.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    // After the operator at `j`: skip whitespace, read the destination.
    let take = |mut j: usize, dup: bool, targets: &mut Vec<String>| -> usize {
        while j < len && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        if let Some(path) = extract_token(command, j)
            && !path.is_empty()
            && !path.starts_with('&')
            && !path.starts_with("/dev/")
            && !(dup && (path == "-" || path.bytes().all(|b| b.is_ascii_digit())))
        {
            targets.push(path);
        }
        j
    };

    while i < len {
        match bytes[i] {
            b'\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
                i += 1;
            }
            b'"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
                i += 1;
            }
            b'>' if !in_single_quote && !in_double_quote => {
                let mut j = i + 1;
                let mut dup = false;
                match bytes.get(j) {
                    Some(b'>') | Some(b'|') => j += 1,
                    Some(b'&') => {
                        j += 1;
                        dup = true;
                    }
                    _ => {}
                }
                i = take(j, dup, targets);
            }
            _ => {
                // `2>`, `&>`, `2>>`, `&>>`, `2>&1`
                if !in_single_quote
                    && !in_double_quote
                    && i + 1 < len
                    && bytes[i + 1] == b'>'
                    && (bytes[i] == b'&' || bytes[i].is_ascii_digit())
                {
                    let mut j = i + 2;
                    let mut dup = false;
                    match bytes.get(j) {
                        Some(b'>') | Some(b'|') => j += 1,
                        Some(b'&') => {
                            j += 1;
                            dup = true;
                        }
                        _ => {}
                    }
                    i = take(j, dup, targets);
                } else {
                    i += 1;
                }
            }
        }
    }
}

/// Whether a command argument is (or starts) a redirection, which is not
/// an operand of the command.
fn redirect_token(arg: &str) -> Option<bool> {
    let rest = arg.trim_start_matches(|c: char| c.is_ascii_digit() || c == '&');
    if rest.starts_with('>') || rest.starts_with('<') {
        // `>` alone consumes the next word; `>file` carries its target.
        let op_only = rest.trim_start_matches(['>', '<', '|', '&']).is_empty();
        return Some(op_only);
    }
    None
}

/// Operands of one segment, without redirections.
fn operands(parts: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip_next = false;
    for part in parts {
        if skip_next {
            skip_next = false;
            continue;
        }
        if let Some(op_only) = redirect_token(part) {
            skip_next = op_only;
            continue;
        }
        out.push(part.trim_matches(|c| c == '\'' || c == '"').to_string());
    }
    out
}

/// Extract write targets from common write commands like `tee`, `cp`,
/// `mv`, `touch`, in every segment, after stripping safe wrappers
/// (`timeout`, `env X=1`, `nice`, ...).
fn extract_command_write_targets(command: &str, targets: &mut Vec<String>) {
    for sub in command_segments(command) {
        let sub = strip_safe_wrappers(sub.trim());
        let parts: Vec<&str> = sub.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let base_cmd = parts[0].rsplit('/').next().unwrap_or(parts[0]);
        let args = operands(&parts[1..]);

        match base_cmd {
            "tee" | "touch" => {
                for arg in &args {
                    if !arg.starts_with('-') {
                        targets.push(arg.clone());
                    }
                }
            }
            "cp" | "mv" => {
                // `-t DIR` / `--target-directory=DIR` names the destination;
                // otherwise the last non-flag operand does.
                let mut explicit = None;
                let mut iter = args.iter();
                while let Some(arg) = iter.next() {
                    if arg == "-t" {
                        explicit = iter.next().cloned();
                    } else if let Some(dir) = arg.strip_prefix("--target-directory=") {
                        explicit = Some(dir.to_string());
                    }
                }
                if let Some(dir) = explicit {
                    targets.push(dir);
                    continue;
                }
                let non_flag: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
                if non_flag.len() >= 2 {
                    targets.push(non_flag.last().unwrap().to_string());
                }
            }
            _ => {}
        }
    }
}

/// `command` with heredoc bodies blanked (they are data, not commands),
/// split into segments.
fn command_segments(command: &str) -> Vec<String> {
    split_on_operators(&mask_heredoc_bodies(command))
}

/// `command` with every heredoc body replaced by spaces (byte positions
/// preserved).
fn mask_heredoc_bodies(command: &str) -> String {
    let Some(analysis) = classify_command(command).redirects else {
        return command.to_string();
    };
    if analysis.heredoc_bodies.is_empty() {
        return command.to_string();
    }
    let mut bytes = command.as_bytes().to_vec();
    for &(start, end) in &analysis.heredoc_bodies {
        for byte in &mut bytes[start..end.min(command.len())] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    }
    String::from_utf8(bytes).unwrap_or_else(|_| command.to_string())
}

/// Split a command on `&&`, `||`, `|`, `;`, a background `&`, and
/// newlines (respecting quotes). A `&` that belongs to a redirection
/// (`2>&1`, `&>`) does not split.
fn split_on_operators(command: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let bytes = command.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    while i < len {
        let unquoted = !in_single_quote && !in_double_quote;
        match bytes[i] {
            b'\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
                current.push('\'');
                i += 1;
            }
            b'"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
                current.push('"');
                i += 1;
            }
            b'&' if unquoted && i + 1 < len && bytes[i + 1] == b'&' => {
                parts.push(std::mem::take(&mut current));
                i += 2;
            }
            b'&' if unquoted
                && !(i > 0 && matches!(bytes[i - 1], b'>' | b'<'))
                && bytes.get(i + 1) != Some(&b'>') =>
            {
                parts.push(std::mem::take(&mut current));
                i += 1;
            }
            b'|' if unquoted => {
                if i + 1 < len && bytes[i + 1] == b'|' {
                    i += 2;
                } else {
                    i += 1;
                }
                parts.push(std::mem::take(&mut current));
            }
            b';' | b'\n' if unquoted => {
                parts.push(std::mem::take(&mut current));
                i += 1;
            }
            _ => {
                let ch = command[i..].chars().next().unwrap_or(' ');
                current.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    if !current.trim().is_empty() {
        parts.push(current);
    }
    parts
}

/// Extract a single shell token starting at position `start`.
fn extract_token(command: &str, start: usize) -> Option<String> {
    let bytes = command.as_bytes();
    let len = bytes.len();
    if start >= len {
        return None;
    }

    let mut token = String::new();
    let mut i = start;

    if bytes[i] == b'\'' {
        // Single-quoted token
        i += 1;
        while i < len && bytes[i] != b'\'' {
            token.push(bytes[i] as char);
            i += 1;
        }
        return Some(token);
    }
    if bytes[i] == b'"' {
        // Double-quoted token
        i += 1;
        while i < len && bytes[i] != b'"' {
            token.push(bytes[i] as char);
            i += 1;
        }
        return Some(token);
    }

    // Unquoted token
    while i < len && !matches!(bytes[i], b' ' | b'\t' | b';' | b'|' | b'&' | b'\n' | b'>') {
        token.push(bytes[i] as char);
        i += 1;
    }
    Some(token)
}

// ---------------------------------------------------------------------------
// Contained-redirect absorption (ADR 09 §1)
// ---------------------------------------------------------------------------

/// Words that change the shell's directory, or run code in the current
/// shell that may (`trap` handlers, `fc` history re-execution): their
/// presence anywhere makes the command's own cwd unprovable.
const CWD_WORDS: &[&str] = &[
    "cd", "pushd", "popd", "source", "eval", "exec", "trap", "fc",
];

/// Whether `command` may change its working directory: a cd-family or
/// cwd-executing word at any position (a conservative word scan, so a
/// compound like `while true; do cd ~; break; done` is caught), `.` as a
/// command word, or a command name the AST cannot read literally
/// (`"c""d" ~`). Without the AST the answer is yes (unprovable). Heredoc
/// bodies are data and are not scanned.
pub fn changes_directory(command: &str) -> bool {
    let classification = classify_command(command);
    if !classification.ast_available || classification.unreadable_command_name {
        return true;
    }
    let text = mask_heredoc_bodies(command);
    let separator = |c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ';' | '&' | '|' | '(' | ')' | '{' | '}' | '`' | '$' | '"' | '\'' | '<' | '>' | '='
            )
    };
    if text.split(separator).any(|word| CWD_WORDS.contains(&word)) {
        return true;
    }
    // `.` (source) in command position.
    let bytes = text.as_bytes();
    bytes.iter().enumerate().any(|(i, &b)| {
        b == b'.'
            && bytes
                .get(i + 1)
                .is_none_or(|next| next.is_ascii_whitespace())
            && text[..i]
                .trim_end_matches([' ', '\t'])
                .chars()
                .last()
                .is_none_or(|prev| matches!(prev, ';' | '&' | '|' | '(' | '{' | '\n' | '`'))
    })
}

/// The working directory of a bash call, and whether it is provable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BashCwd {
    /// The directory the call runs in, when known.
    pub path: Option<std::path::PathBuf>,
    /// Whether relative targets may be placed against `path`.
    pub trusted: bool,
}

impl BashCwd {
    /// A call with `workdir` in a session launched in `launch_dir`, whose
    /// shell has (`drifted`) or has not left that directory. An absolute
    /// `workdir` is provable on its own (its subshell starts there); a
    /// relative one joins the session directory, so it needs no drift.
    pub fn new(workdir: Option<&str>, launch_dir: Option<&std::path::Path>, drifted: bool) -> Self {
        match workdir {
            Some(dir) if std::path::Path::new(dir).is_absolute() => BashCwd {
                path: Some(std::path::PathBuf::from(dir)),
                trusted: true,
            },
            Some(dir) => BashCwd {
                path: launch_dir.map(|base| base.join(dir)),
                trusted: !drifted && launch_dir.is_some(),
            },
            None => BashCwd {
                path: launch_dir.map(std::path::Path::to_path_buf),
                trusted: !drifted && launch_dir.is_some(),
            },
        }
    }

    /// The path as the analysis functions take it.
    pub fn as_str(&self) -> Option<&str> {
        self.path.as_deref().and_then(std::path::Path::to_str)
    }
}

/// What redirect absorption needs to know about the call's environment.
#[derive(Debug, Clone)]
pub struct AbsorbScope {
    /// A jail backend confines bash this turn.
    pub sandboxed: bool,
    /// The jail's writable roots: the analysis workspace (the workspace,
    /// else the scratch directory) first, then the scratch directory.
    pub roots: Vec<std::path::PathBuf>,
    /// The call's working directory.
    pub cwd: BashCwd,
}

/// Device nodes both jail backends let bash write.
fn contained_device(target: &str) -> bool {
    matches!(target, "/dev/null" | "/dev/zero" | "/dev/tty")
        || target
            .strip_prefix("/dev/fd/")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Whether every redirect occurrence is containment-bounded: an fd-dup,
/// a heredoc written to a file by a data-only command, or an output
/// redirect to a resolvable, blocklist-clean target (inside the roots, or
/// outside them — the out-of-roots check then escalates it).
fn redirects_absorbable(
    analysis: &bash_ast::RedirectAnalysis,
    roots: &[std::path::PathBuf],
    cwd: Option<&str>,
) -> bool {
    if analysis.stray_operator || roots.is_empty() {
        return false;
    }
    for redirect in &analysis.redirects {
        match redirect.shape {
            bash_ast::RedirectShape::FdDup | bash_ast::RedirectShape::HeredocWrite => {}
            shape if shape.writes_file() => {
                let Some(bash_ast::RedirectTarget::Literal(target)) = &redirect.target else {
                    return false;
                };
                if target.starts_with("/dev/") {
                    if contained_device(target) {
                        continue;
                    }
                    return false;
                }
                if is_relative_target(target) && cwd.is_none() {
                    return false;
                }
                let spelled = resolve_target(target, cwd);
                if super::path_policy::check_write(&spelled, None).is_err() {
                    return false;
                }
                match physical_path(&spelled, true) {
                    Some(landed) if super::path_policy::check_write(&landed, None).is_ok() => {}
                    _ => return false,
                }
            }
            _ => return false,
        }
    }
    true
}

/// The classifier warning that still applies to `command` under `scope`
/// (ADR 09 §1): contained redirect and heredoc-newline warnings are
/// absorbed, every other warning re-emits. `None` when nothing remains.
/// The approval gate and the bash tool's output annotation both use this,
/// so a silent gate never meets a loud result.
pub fn effective_warning(
    command: &str,
    classification: &CommandClassification,
    scope: &AbsorbScope,
) -> Option<String> {
    let SecurityVerdict::Warn(original) = &classification.verdict else {
        return None;
    };
    let analysis = match &classification.redirects {
        Some(analysis) if scope.sandboxed && classification.ast_available => analysis,
        _ => return Some(original.clone()),
    };
    let cwd = (scope.cwd.trusted && !changes_directory(command))
        .then(|| scope.cwd.as_str())
        .flatten();
    let redirects_ok = redirects_absorbable(analysis, &scope.roots, cwd);
    let newline_ok = redirects_ok && !analysis.unattributed_newline;
    let remaining: Vec<&str> = classification
        .warnings
        .iter()
        .filter(|warning| match warning.kind {
            WarningKind::Other => true,
            WarningKind::Redirection => !redirects_ok,
            WarningKind::Newline => !newline_ok,
        })
        .map(|warning| warning.message.as_str())
        .collect();
    merged_warning(&remaining)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- split_compound_command -------------------------------------------

    #[test]
    fn split_simple() {
        assert_eq!(split_compound_command("echo hello"), vec!["echo hello"]);
    }

    #[test]
    fn split_and() {
        assert_eq!(
            split_compound_command("cd /tmp && ls -la"),
            vec!["cd /tmp", "ls -la"]
        );
    }

    #[test]
    fn split_or() {
        assert_eq!(
            split_compound_command("false || echo fallback"),
            vec!["false", "echo fallback"]
        );
    }

    #[test]
    fn split_pipe() {
        assert_eq!(
            split_compound_command("cat file | grep foo | wc -l"),
            vec!["cat file", "grep foo", "wc -l"]
        );
    }

    #[test]
    fn split_semicolon() {
        assert_eq!(
            split_compound_command("echo a; echo b; echo c"),
            vec!["echo a", "echo b", "echo c"]
        );
    }

    #[test]
    fn split_mixed() {
        assert_eq!(
            split_compound_command("cd /tmp && echo ok; ls | head"),
            vec!["cd /tmp", "echo ok", "ls", "head"]
        );
    }

    #[test]
    fn split_respects_single_quotes() {
        assert_eq!(
            split_compound_command("echo 'a && b' && echo done"),
            vec!["echo 'a && b'", "echo done"]
        );
    }

    #[test]
    fn split_respects_double_quotes() {
        assert_eq!(
            split_compound_command(r#"echo "a || b" || echo fail"#),
            vec![r#"echo "a || b""#, "echo fail"]
        );
    }

    #[test]
    fn split_respects_backtick_quoting() {
        assert_eq!(
            split_compound_command("echo `echo a && echo b` && echo done"),
            vec!["echo `echo a && echo b`", "echo done"]
        );
    }

    #[test]
    fn split_respects_dollar_paren() {
        assert_eq!(
            split_compound_command("echo $(echo a && echo b) && echo done"),
            vec!["echo $(echo a && echo b)", "echo done"]
        );
    }

    #[test]
    fn split_respects_backslash_escape() {
        assert_eq!(
            split_compound_command(r"echo a \&\& b && echo done"),
            vec![r"echo a \&\& b", "echo done"]
        );
    }

    #[test]
    fn split_nested_dollar_paren() {
        assert_eq!(
            split_compound_command("echo $(echo $(date)) && pwd"),
            vec!["echo $(echo $(date))", "pwd"]
        );
    }

    #[test]
    fn split_empty() {
        assert!(split_compound_command("").is_empty());
        assert!(split_compound_command("   ").is_empty());
    }

    // ---- extract_quoted_content -------------------------------------------

    #[test]
    fn extract_strips_single_quotes() {
        let e = extract_quoted_content("echo 'hello world' && ls");
        assert!(!e.with_double_quotes.contains("hello world"));
        assert!(!e.fully_unquoted.contains("hello world"));
    }

    #[test]
    fn extract_preserves_double_quoted_for_with_dq() {
        let e = extract_quoted_content(r#"echo "$(whoami)" foo"#);
        assert!(e.with_double_quotes.contains("$(whoami)"));
        assert!(!e.fully_unquoted.contains("$(whoami)"));
    }

    #[test]
    fn extract_backslash_escape() {
        let e = extract_quoted_content(r"echo \$(safe)");
        assert!(e.with_double_quotes.contains(r"\$"));
        assert!(e.fully_unquoted.contains(r"\$"));
    }

    // ---- individual validators -------------------------------------------

    #[test]
    fn control_chars_blocked() {
        let ctx = ValidationContext::new("echo safe\x00; rm -rf /");
        assert!(matches!(
            validate_control_characters(&ctx),
            SecurityVerdict::Block(_)
        ));
    }

    #[test]
    fn incomplete_starts_with_dash() {
        let ctx = ValidationContext::new("-la /tmp");
        assert!(matches!(
            validate_incomplete_commands(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn dangerous_patterns_dollar_paren() {
        let ctx = ValidationContext::new("echo $(whoami)");
        assert!(matches!(
            validate_dangerous_patterns(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn dangerous_patterns_backtick() {
        let ctx = ValidationContext::new("echo `id`");
        assert!(matches!(
            validate_dangerous_patterns(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn dangerous_patterns_safe_in_single_quotes() {
        let ctx = ValidationContext::new("echo '$(whoami)'");
        assert_eq!(validate_dangerous_patterns(&ctx), SecurityVerdict::Allow);
    }

    #[test]
    fn dangerous_patterns_dollar_paren_in_double_quotes() {
        let ctx = ValidationContext::new(r#"echo "$(whoami)""#);
        assert!(matches!(
            validate_dangerous_patterns(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn ifs_injection() {
        let ctx = ValidationContext::new("echo${IFS}hello");
        assert!(matches!(
            validate_ifs_injection(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn proc_environ() {
        let ctx = ValidationContext::new("cat /proc/self/environ");
        assert!(matches!(
            validate_proc_environ(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn newline_command_separation() {
        let ctx = ValidationContext::new("echo safe\nrm -rf /tmp");
        assert!(matches!(validate_newlines(&ctx), SecurityVerdict::Warn(_)));
    }

    #[test]
    fn backslash_escaped_whitespace() {
        let ctx = ValidationContext::new(r"echo\ test");
        assert!(matches!(
            validate_backslash_escaped_whitespace(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn backslash_escaped_operator() {
        let ctx = ValidationContext::new(r"cat file \; echo secret");
        assert!(matches!(
            validate_backslash_escaped_operators(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn brace_expansion_comma() {
        let ctx = ValidationContext::new("echo {a,b,c}");
        assert!(matches!(
            validate_brace_expansion(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn brace_expansion_range() {
        let ctx = ValidationContext::new("echo {1..10}");
        assert!(matches!(
            validate_brace_expansion(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn unicode_whitespace() {
        let ctx = ValidationContext::new("echo\u{00A0}test");
        assert!(matches!(
            validate_unicode_whitespace(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn ansi_c_quoting() {
        let ctx = ValidationContext::new("find . $'\\x2d'exec id");
        assert!(matches!(
            validate_ansi_c_quoting(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn zsh_zmodload() {
        let ctx = ValidationContext::new("zmodload zsh/system");
        assert!(matches!(
            validate_zsh_dangerous_commands(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn eval_keyword() {
        let ctx = ValidationContext::new("eval rm -rf /tmp/x");
        assert!(matches!(validate_eval(&ctx), SecurityVerdict::Warn(_)));
    }

    #[test]
    fn eval_not_substring() {
        let ctx = ValidationContext::new("echo evaluate");
        assert_eq!(validate_eval(&ctx), SecurityVerdict::Allow);
    }

    #[test]
    fn eval_safe_in_single_quotes() {
        let ctx = ValidationContext::new("echo 'eval rm -rf'");
        assert_eq!(validate_eval(&ctx), SecurityVerdict::Allow);
    }

    #[test]
    fn redirections_in_unquoted() {
        let ctx = ValidationContext::new("echo hello > /tmp/out");
        assert!(matches!(
            validate_redirections(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn redirections_safe_in_quotes() {
        let ctx = ValidationContext::new("echo 'hello > world'");
        assert_eq!(validate_redirections(&ctx), SecurityVerdict::Allow);
    }

    // ---- check_command (integration) --------------------------------------

    fn mock_dangerous(cmd: &str) -> Option<&'static str> {
        let trimmed = cmd.trim();
        if trimmed == "rm -rf /" || trimmed == "rm -fr /" {
            Some("Blocked: recursive deletion of root filesystem")
        } else {
            None
        }
    }

    #[test]
    fn compound_catches_hidden_rm() {
        let cmd = "echo safe && rm -rf / && echo done";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Block(msg) => {
                assert!(msg.contains("recursive deletion"));
                assert!(msg.contains("subcommand"));
            }
            other => panic!("expected Block, got {:?}", other),
        }
    }

    #[test]
    fn compound_safe_subcommands() {
        let cmd = "echo a && echo b";
        assert_eq!(check_command(cmd, mock_dangerous), SecurityVerdict::Allow);
    }

    #[test]
    fn compound_with_injection() {
        let cmd = "cd /tmp && echo $(whoami)";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Warn(msg) => assert!(msg.contains("command substitution")),
            other => panic!("expected Warn, got {:?}", other),
        }
    }

    #[test]
    fn compound_dangerous_in_pipe() {
        let cmd = "echo hello | rm -rf /";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Block(msg) => assert!(msg.contains("recursive deletion")),
            other => panic!("expected Block, got {:?}", other),
        }
    }

    #[test]
    fn pipeline_multiple_warnings() {
        let cmd = "eval $(whoami)";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Warn(msg) => {
                assert!(msg.contains("security notice"));
            }
            other => panic!("expected Warn, got {:?}", other),
        }
    }

    #[test]
    fn clean_command_allowed() {
        assert_eq!(
            check_command("ls -la", mock_dangerous),
            SecurityVerdict::Allow
        );
        assert_eq!(
            check_command("echo hello world", mock_dangerous),
            SecurityVerdict::Allow
        );
        assert_eq!(
            check_command("git status", mock_dangerous),
            SecurityVerdict::Allow
        );
        assert_eq!(
            check_command("cargo build --release", mock_dangerous),
            SecurityVerdict::Allow
        );
    }

    #[test]
    fn safe_pipe_allowed() {
        assert_eq!(
            check_command("cat file.txt | grep pattern | wc -l", mock_dangerous),
            SecurityVerdict::Allow
        );
    }

    // ---- AST integration tests -------------------------------------------

    #[test]
    fn ast_find_exec_no_false_positive() {
        // `find -exec \;` should NOT trigger the backslash-escaped-operator
        // warning, because tree-sitter confirms `\;` is a word argument
        // (no actual `;` operator node in the AST).
        let cmd = r"find . -type f -name '*.rs' -exec grep 'TODO' {} \;";
        assert_eq!(check_command(cmd, mock_dangerous), SecurityVerdict::Allow);
    }

    #[test]
    fn ast_real_escaped_semicolon_with_actual_operator_still_warns() {
        // When a REAL `;` operator coexists, the warning should still fire
        // on any `\;` in the command.
        let cmd = r"find . -exec cmd {} \; ; echo done";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Warn(msg) => {
                assert!(msg.contains("security notice"));
            }
            other => panic!("expected Warn, got {:?}", other),
        }
    }

    #[test]
    fn ast_subshell_detected() {
        let cmd = "(cd /tmp && ls)";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Warn(msg) => assert!(msg.contains("subshell")),
            other => panic!("expected Warn about subshell, got {:?}", other),
        }
    }

    #[test]
    fn ast_command_substitution_detected() {
        let cmd = "echo $(whoami)";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Warn(msg) => assert!(msg.contains("security notice")),
            other => panic!("expected Warn, got {:?}", other),
        }
    }

    #[test]
    fn ast_malformed_syntax_detected() {
        let cmd = "echo 'unterminated";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Warn(msg) => assert!(msg.contains("security notice")),
            other => panic!("expected Warn, got {:?}", other),
        }
    }

    // ---- strip_safe_wrappers -----------------------------------------------

    #[test]
    fn strip_timeout_basic() {
        assert_eq!(
            strip_safe_wrappers("timeout 5 npm run build"),
            "npm run build"
        );
    }

    #[test]
    fn strip_timeout_with_suffix() {
        assert_eq!(strip_safe_wrappers("timeout 5s npm start"), "npm start");
        assert_eq!(strip_safe_wrappers("timeout 10m cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("timeout 1h cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("timeout 2d cmd"), "cmd");
    }

    #[test]
    fn strip_timeout_with_decimal() {
        assert_eq!(strip_safe_wrappers("timeout 1.5 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("timeout 0.5s cmd"), "cmd");
    }

    #[test]
    fn strip_timeout_with_flags() {
        assert_eq!(strip_safe_wrappers("timeout --foreground 5 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("timeout --kill-after=5 10 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("timeout --signal TERM 10 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("timeout --signal=KILL 10 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("timeout -k 5 10 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("timeout -k5 10 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("timeout -sTERM 10 cmd"), "cmd");
        assert_eq!(
            strip_safe_wrappers("timeout -v --preserve-status 10 cmd"),
            "cmd"
        );
    }

    #[test]
    fn strip_timeout_with_dashdash() {
        assert_eq!(strip_safe_wrappers("timeout -- 5 cmd"), "cmd");
    }

    #[test]
    fn strip_timeout_rejects_unsafe_flag_value() {
        assert_eq!(
            strip_safe_wrappers("timeout -k$(id) 10 cmd"),
            "timeout -k$(id) 10 cmd"
        );
    }

    #[test]
    fn strip_time() {
        assert_eq!(strip_safe_wrappers("time cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("time -- cmd"), "cmd");
    }

    #[test]
    fn strip_time_does_not_match_timestamp() {
        assert_eq!(strip_safe_wrappers("timestamp cmd"), "timestamp cmd");
    }

    #[test]
    fn strip_nice_bare() {
        assert_eq!(strip_safe_wrappers("nice cmd"), "cmd");
    }

    #[test]
    fn strip_nice_with_priority() {
        assert_eq!(strip_safe_wrappers("nice -n 10 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("nice -n -5 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("nice -10 cmd"), "cmd");
    }

    #[test]
    fn strip_nice_with_dashdash() {
        assert_eq!(strip_safe_wrappers("nice -n 10 -- cmd"), "cmd");
    }

    #[test]
    fn strip_nohup() {
        assert_eq!(strip_safe_wrappers("nohup cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("nohup -- cmd"), "cmd");
    }

    #[test]
    fn strip_stdbuf() {
        assert_eq!(strip_safe_wrappers("stdbuf -o0 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("stdbuf -oL -eL cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("stdbuf -i0 -o0 -e0 cmd"), "cmd");
    }

    #[test]
    fn strip_stdbuf_with_dashdash() {
        assert_eq!(strip_safe_wrappers("stdbuf -o0 -- cmd"), "cmd");
    }

    #[test]
    fn strip_stdbuf_requires_flag() {
        assert_eq!(strip_safe_wrappers("stdbuf cmd"), "stdbuf cmd");
    }

    #[test]
    fn strip_env_var() {
        assert_eq!(strip_safe_wrappers("GOOS=linux go build"), "go build");
        assert_eq!(
            strip_safe_wrappers("NODE_ENV=production npm start"),
            "npm start"
        );
    }

    #[test]
    fn strip_multiple_env_vars() {
        assert_eq!(
            strip_safe_wrappers("GOOS=linux GOARCH=amd64 go build"),
            "go build"
        );
    }

    #[test]
    fn strip_env_plus_wrapper() {
        assert_eq!(
            strip_safe_wrappers("NODE_ENV=production timeout 10 npm start"),
            "npm start"
        );
    }

    #[test]
    fn strip_chained_wrappers() {
        assert_eq!(strip_safe_wrappers("nohup nice -n 10 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("nohup timeout 5 cmd"), "cmd");
        assert_eq!(strip_safe_wrappers("time nohup cmd"), "cmd");
    }

    #[test]
    fn strip_no_op_for_plain_commands() {
        assert_eq!(strip_safe_wrappers("npm run build"), "npm run build");
        assert_eq!(strip_safe_wrappers("ls -la"), "ls -la");
        assert_eq!(strip_safe_wrappers("git status"), "git status");
    }

    #[test]
    fn strip_unsafe_env_var_not_stripped() {
        assert_eq!(
            strip_safe_wrappers("PATH=/evil go build"),
            "PATH=/evil go build"
        );
        assert_eq!(
            strip_safe_wrappers("LD_PRELOAD=evil.so cmd"),
            "LD_PRELOAD=evil.so cmd"
        );
    }

    #[test]
    fn strip_env_with_unsafe_value_not_stripped() {
        assert_eq!(
            strip_safe_wrappers("GOOS=$(whoami) go build"),
            "GOOS=$(whoami) go build"
        );
        assert_eq!(strip_safe_wrappers("NODE_ENV=a;b cmd"), "NODE_ENV=a;b cmd");
    }

    #[test]
    fn strip_does_not_match_across_newlines() {
        assert_eq!(
            strip_safe_wrappers("timeout 5 echo safe\nrm -rf /"),
            "echo safe\nrm -rf /"
        );
    }

    // ---- wrapper stripping + dangerous pattern integration -----------------

    #[test]
    fn wrapper_stripped_catches_dangerous() {
        let cmd = "timeout 5 rm -rf /";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Block(msg) => {
                assert!(msg.contains("recursive deletion"));
                assert!(msg.contains("normalizing wrappers"));
            }
            other => panic!("expected Block, got {:?}", other),
        }
    }

    #[test]
    fn env_stripped_catches_dangerous() {
        let cmd = "NODE_ENV=prod rm -rf /";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Block(msg) => assert!(msg.contains("recursive deletion")),
            other => panic!("expected Block, got {:?}", other),
        }
    }

    #[test]
    fn wrapper_stripped_in_subcommand_catches_dangerous() {
        let cmd = "echo ok && timeout 5 rm -rf /";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Block(msg) => {
                assert!(msg.contains("recursive deletion"));
                assert!(msg.contains("subcommand"));
            }
            other => panic!("expected Block, got {:?}", other),
        }
    }

    #[test]
    fn nohup_wrapped_dangerous_caught() {
        let cmd = "nohup rm -rf /";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Block(msg) => assert!(msg.contains("recursive deletion")),
            other => panic!("expected Block, got {:?}", other),
        }
    }

    #[test]
    fn chained_wrapper_dangerous_caught() {
        let cmd = "nohup nice -n 10 rm -rf /";
        match check_command(cmd, mock_dangerous) {
            SecurityVerdict::Block(msg) => assert!(msg.contains("recursive deletion")),
            other => panic!("expected Block, got {:?}", other),
        }
    }

    #[test]
    fn zsh_dangerous_after_wrapper() {
        let ctx = ValidationContext::new("nohup zmodload zsh/system");
        assert!(matches!(
            validate_zsh_dangerous_commands(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    #[test]
    fn zsh_dangerous_after_env_var() {
        let ctx = ValidationContext::new("NODE_ENV=prod zmodload zsh/system");
        assert!(matches!(
            validate_zsh_dangerous_commands(&ctx),
            SecurityVerdict::Warn(_)
        ));
    }

    // ---- check_with_context ------------------------------------------------

    #[test]
    fn context_command_propagates_block() {
        let verdict = check_with_context("rm -rf /", SecurityContext::Command);
        assert!(matches!(verdict, SecurityVerdict::Block(_)));
    }

    #[test]
    fn context_untrusted_propagates_block() {
        let verdict = check_with_context("rm -rf /", SecurityContext::Untrusted);
        assert!(matches!(verdict, SecurityVerdict::Block(_)));
    }

    #[test]
    fn context_untrusted_downgrades_warn_to_allow() {
        // Output redirection produces a Warn in Command context
        let verdict = check_with_context("echo hello > /tmp/out.txt", SecurityContext::Command);
        assert!(
            matches!(verdict, SecurityVerdict::Warn(_)),
            "Expected Warn for redirection in Command context, got {:?}",
            verdict
        );

        let verdict_untrusted =
            check_with_context("echo hello > /tmp/out.txt", SecurityContext::Untrusted);
        assert!(
            matches!(verdict_untrusted, SecurityVerdict::Allow),
            "Untrusted context should downgrade Warn to Allow, got {:?}",
            verdict_untrusted
        );
    }

    #[test]
    fn context_allows_safe_command_in_both() {
        assert!(matches!(
            check_with_context("echo hello", SecurityContext::Command),
            SecurityVerdict::Allow
        ));
        assert!(matches!(
            check_with_context("echo hello", SecurityContext::Untrusted),
            SecurityVerdict::Allow
        ));
    }

    // ---- extract_write_targets / check_write_targets -----------------------

    #[test]
    fn extract_redirect_simple() {
        let targets = extract_write_targets("echo hello > /tmp/out.txt");
        assert!(
            targets.contains(&"/tmp/out.txt".to_string()),
            "targets: {:?}",
            targets
        );
    }

    #[test]
    fn extract_redirect_append() {
        let targets = extract_write_targets("echo hello >> /tmp/out.txt");
        assert!(
            targets.contains(&"/tmp/out.txt".to_string()),
            "targets: {:?}",
            targets
        );
    }

    #[test]
    fn extract_redirect_stderr() {
        let targets = extract_write_targets("cmd 2> /tmp/err.log");
        assert!(
            targets.contains(&"/tmp/err.log".to_string()),
            "targets: {:?}",
            targets
        );
    }

    #[test]
    fn extract_redirect_ampersand() {
        let targets = extract_write_targets("cmd &> /tmp/all.log");
        assert!(
            targets.contains(&"/tmp/all.log".to_string()),
            "targets: {:?}",
            targets
        );
    }

    #[test]
    fn extract_redirect_skips_dev_null() {
        let targets = extract_write_targets("cmd > /dev/null 2>&1");
        assert!(
            targets.is_empty(),
            "should skip /dev/null, got: {:?}",
            targets
        );
    }

    #[test]
    fn extract_tee_targets() {
        let targets = extract_write_targets("echo hello | tee /tmp/a.txt /tmp/b.txt");
        assert!(targets.contains(&"/tmp/a.txt".to_string()));
        assert!(targets.contains(&"/tmp/b.txt".to_string()));
    }

    #[test]
    fn extract_cp_destination() {
        let targets = extract_write_targets("cp src.txt /tmp/dest.txt");
        assert!(targets.contains(&"/tmp/dest.txt".to_string()));
    }

    #[test]
    fn extract_mv_destination() {
        let targets = extract_write_targets("mv old.txt /tmp/new.txt");
        assert!(targets.contains(&"/tmp/new.txt".to_string()));
    }

    #[test]
    fn extract_touch_targets() {
        let targets = extract_write_targets("touch /tmp/a.txt /tmp/b.txt");
        assert!(targets.contains(&"/tmp/a.txt".to_string()));
        assert!(targets.contains(&"/tmp/b.txt".to_string()));
    }

    #[test]
    fn extract_compound_redirect() {
        let targets = extract_write_targets("echo a > /tmp/a.txt && echo b > /tmp/b.txt");
        assert!(targets.contains(&"/tmp/a.txt".to_string()));
        assert!(targets.contains(&"/tmp/b.txt".to_string()));
    }

    #[test]
    fn write_targets_leave_the_workspace_boundary_to_the_gate() {
        assert!(
            check_write_targets(
                "echo hello > /tmp/out.txt",
                &[std::path::PathBuf::from("/home/user/project")]
            )
            .is_ok()
        );
        assert!(
            check_write_targets(
                "echo hello > ./out.txt",
                &[std::path::PathBuf::from("/home/user/project")]
            )
            .is_ok()
        );
        assert!(
            check_write_targets(
                "echo x > ~/.zshrc",
                &[std::path::PathBuf::from("/home/user/project")]
            )
            .is_err()
        );
    }

    #[test]
    fn deletion_targets_skip_flags_and_strip_wrappers() {
        assert_eq!(
            extract_deletion_targets("timeout 5 rm -rf -- a 'b c' && unlink d"),
            vec!["a", "b", "c", "d"]
        );
        assert!(extract_deletion_targets("echo rm -rf x").is_empty());
        assert_eq!(extract_deletion_targets("/bin/rm x"), vec!["x"]);
    }

    #[test]
    fn deletion_guard_reports_existing_paths_outside_the_workspace() {
        let home = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let workspace = home.path().join("ws");
        let outside = home.path().join("keep");
        std::fs::create_dir_all(workspace.join("build")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("notes.txt"), "x").unwrap();
        let ws = Some(workspace.as_path());
        let check = |cmd: &str| outside_deletion_target(cmd, ws, None, &[]);

        // Inside the workspace, the workspace itself, and missing paths: allowed.
        assert!(check("rm -rf build").is_none());
        assert!(check(&format!("rm -rf {}", workspace.display())).is_none());
        assert!(check(&format!("rm {}/missing.txt", outside.display())).is_none());
        // Existing paths outside: blocked, including traversal and globs.
        assert!(check(&format!("rm {}/notes.txt", outside.display())).is_some());
        assert!(check("rm -rf ../keep").is_some());
        assert!(check(&format!("rm -rf {}/*", outside.display())).is_some());
        assert!(check(&format!("shred -u {}/notes.txt", outside.display())).is_some());
        // Files under the jail's temp channel (`/tmp`) and extra roots
        // (scratch) stay deletable.
        if crate::sandbox::temp_channel_available(std::slice::from_ref(&workspace)) {
            let tmp = tempfile::NamedTempFile::new_in("/tmp").unwrap();
            assert!(check(&format!("rm {}", tmp.path().display())).is_none());
        }
        assert!(
            outside_deletion_target(
                &format!("rm {}/notes.txt", outside.display()),
                ws,
                None,
                std::slice::from_ref(&outside)
            )
            .is_none()
        );
        // No workspace: no boundary.
        assert!(
            outside_deletion_target(
                &format!("rm {}/notes.txt", outside.display()),
                None,
                None,
                &[]
            )
            .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn deletion_guard_treats_a_final_symlink_as_the_link() {
        let root = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let workspace = root.path().join("ws");
        let outside = root.path().join("outside");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, workspace.join("link")).unwrap();
        let ws = Some(workspace.as_path());
        assert!(outside_deletion_target("rm link", ws, None, &[]).is_none());
        // Through the link, the deleted entries live outside.
        std::fs::write(outside.join("f"), "x").unwrap();
        assert!(outside_deletion_target("rm link/f", ws, None, &[]).is_some());
    }

    #[test]
    fn outside_write_targets_are_found() {
        let root = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let ws = root.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let outside = root.path().join("out.txt");
        let find = |cmd: &str| outside_write_targets(cmd, Some(&ws), None, &[]);
        assert_eq!(
            find(&format!("echo hi > {}", outside.display())),
            vec![outside.clone()]
        );
        assert_eq!(find("echo hi > ../out.txt"), vec![outside.clone()]);
        assert!(find("echo hi > build/log.txt && cat x > /dev/null").is_empty());
        assert!(find("echo hi > \"$HOME/x\"").is_empty());
        if crate::sandbox::temp_channel_available(std::slice::from_ref(&ws)) {
            assert!(find("touch /tmp/chatshell-t.txt").is_empty());
        }
        // The analyzing process's own TMPDIR is not a jail root (ADR 09
        // §3): on macOS `/var/folders/.../T` is outside.
        let host_tmp = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        if !crate::sandbox::TMP_SPELLINGS
            .iter()
            .any(|tmp| host_tmp.starts_with(tmp))
        {
            let target = host_tmp.join("chatshell-t.txt");
            assert_eq!(
                find(&format!("touch {}", target.display())),
                vec![target.clone()]
            );
        }
        assert!(outside_write_targets("echo > /etc/x", None, None, &[]).is_empty());
    }

    #[test]
    fn write_ask_targets_are_reported() {
        let home = dirs::home_dir().unwrap();
        let cmd = format!(
            "cp evil.json {}",
            home.join(".claude/settings.json").display()
        );
        assert!(write_ask_target(&cmd, Some("/tmp")).is_some());
        assert!(write_ask_target("echo hi > out.txt", Some("/tmp")).is_none());
        assert!(write_ask_target("ls -la", None).is_none());
    }

    #[test]
    fn write_target_no_root_only_blocklist() {
        let result = check_write_targets(
            "echo hello > /tmp/out.txt",
            &[std::path::PathBuf::from("/tmp")],
        );
        assert!(
            result.is_ok(),
            "Without project root, /tmp should be allowed"
        );
    }

    #[test]
    fn write_target_blocks_system_dir() {
        let result = check_write_targets("echo evil > /etc/crontab", &[]);
        assert!(
            result.is_err(),
            "Should block writes to /etc/ regardless of project root"
        );
    }
}

#[cfg(test)]
mod classification_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn cache_runs_pipeline_once_per_command() {
        let cache = ClassificationCache::default();
        let runs = AtomicUsize::new(0);
        let classify = |command: &str| {
            runs.fetch_add(1, Ordering::SeqCst);
            check_command_inner(command, check_dangerous)
        };
        let first = cache.get_or_classify("cargo test", classify);
        let second = cache.get_or_classify("cargo test", |command: &str| {
            runs.fetch_add(1, Ordering::SeqCst);
            check_command_inner(command, check_dangerous)
        });
        assert_eq!(first, second);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cache_evicts_oldest_beyond_capacity() {
        let cache = ClassificationCache::default();
        let runs = AtomicUsize::new(0);
        let count = |command: &str| {
            runs.fetch_add(1, Ordering::SeqCst);
            check_command_inner(command, check_dangerous)
        };
        for i in 0..=CLASSIFICATION_CACHE_CAPACITY {
            cache.get_or_classify(&format!("echo {i}"), count);
        }
        let before = runs.load(Ordering::SeqCst);
        // "echo 0" was evicted; "echo 1" is still cached.
        cache.get_or_classify("echo 1", count);
        assert_eq!(runs.load(Ordering::SeqCst), before);
        cache.get_or_classify("echo 0", count);
        assert_eq!(runs.load(Ordering::SeqCst), before + 1);
    }

    #[test]
    fn classification_reports_ast_availability() {
        assert!(classify_command("ls -la").ast_available);
        let long = format!("echo {}", "a".repeat(12_000));
        let classified = classify_command(&long);
        assert!(!classified.ast_available);
        assert_eq!(classified.verdict, SecurityVerdict::Allow);
    }

    #[test]
    fn classify_matches_check_command() {
        for command in ["ls", "echo $(whoami)", "rm -rf /"] {
            assert_eq!(
                classify_command(command).verdict,
                check_command(command, check_dangerous)
            );
        }
    }

    // ---- ADR 09: analysis scope ⊆ jail writable set ---------------------

    #[cfg(unix)]
    #[test]
    fn write_scope_resolves_symlinks_as_the_kernel_does() {
        let root = tempfile::tempdir_in(dirs::home_dir().unwrap()).unwrap();
        let base = root.path().canonicalize().unwrap();
        let ws = base.join("ws");
        let outside = base.join("outside");
        std::fs::create_dir_all(outside.join("dir")).unwrap();
        std::fs::create_dir_all(&ws).unwrap();
        std::os::unix::fs::symlink(outside.join("dir"), ws.join("link")).unwrap();
        std::fs::write(outside.join("f"), "x").unwrap();
        std::os::unix::fs::symlink(outside.join("f"), ws.join("file-link")).unwrap();
        let scope = WriteScope::new(&ws, &[]);
        // `<ws>/link/../x` lands in `outside/x`: `..` follows the link.
        assert!(!scope.contains(&ws.join("link/../x"), true));
        // A write opens through a final symlink; a deletion removes the link.
        assert!(!scope.contains(&ws.join("file-link"), true));
        assert!(scope.contains(&ws.join("file-link"), false));
        // Plain in-workspace paths, existing or not.
        assert!(scope.contains(&ws.join("src/new/file.rs"), true));
        let find = |cmd: &str| outside_write_targets(cmd, Some(&ws), None, &[]);
        assert_eq!(find("echo hi > link/../x"), vec![outside.join("x")]);
        assert_eq!(find("echo hi > file-link"), vec![outside.join("f")]);
    }

    #[test]
    fn write_scope_temp_roots_are_a_subset_of_the_jail() {
        let home = dirs::home_dir().unwrap();
        let ws = home.join("ws-property");
        let scope = WriteScope::new(&ws, &[]);
        let policy = crate::sandbox::SandboxPolicy::for_bash(Some(&ws), &home.join("s"), false);
        for temp in &scope.temp {
            // Every temp spelling the analysis counts is opened by the jail:
            // the channel is on and the spelling is (a resolution of) /tmp.
            assert!(policy.temp_scratch.is_some(), "{temp:?}");
            assert!(
                crate::sandbox::TMP_SPELLINGS
                    .iter()
                    .any(|tmp| temp.as_path() == std::path::Path::new(tmp)),
                "{temp:?}"
            );
        }
        // The analyzing process's TMPDIR is never a temp root.
        let host = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        if !crate::sandbox::TMP_SPELLINGS
            .iter()
            .any(|t| host.starts_with(t))
        {
            assert!(!scope.inside(&host.join("x")));
        }
    }

    #[test]
    fn lexical_extractor_covers_the_output_family() {
        assert_eq!(extract_write_targets("echo hi >& f"), vec!["f"]);
        assert_eq!(extract_write_targets("echo hi >| f"), vec!["f"]);
        assert_eq!(extract_write_targets("echo hi > 2"), vec!["2"]);
        assert!(extract_write_targets("cmd >&2").is_empty());
        assert!(extract_write_targets("cmd 2>&1").is_empty());
        assert!(extract_write_targets("cmd >&-").is_empty());
        assert_eq!(extract_write_targets("cmd &>> all.log"), vec!["all.log"]);
    }

    #[test]
    fn command_targets_survive_wrappers_and_separators() {
        let targets = |cmd: &str| write_targets(cmd);
        assert_eq!(targets("timeout 5 cp a /x/b"), vec!["/x/b"]);
        assert_eq!(targets("echo hi & cp a /x/b"), vec!["/x/b"]);
        assert_eq!(targets("echo hi\ncp a /x/b"), vec!["/x/b"]);
        assert_eq!(targets("cp -t /x/dir a b"), vec!["/x/dir"]);
        assert_eq!(targets("cp a b 2>&1"), vec!["b"]);
        assert_eq!(targets("cp a b > log"), vec!["log", "b"]);
        // Heredoc body lines are data, not commands or targets.
        assert_eq!(
            targets("cat > f <<'EOF'\ncp a /x/b\nx > /y\nEOF"),
            vec!["f"]
        );
        // An apostrophe in a body does not hide what follows it.
        assert_eq!(
            targets("cat > f <<'EOF'\nit's\nEOF\necho x > /z"),
            vec!["f", "/z"]
        );
    }

    #[test]
    fn blocklist_checks_every_directory_the_command_may_run_in() {
        let home = dirs::home_dir().unwrap();
        let ws = home.join("some-project");
        // Relative `.zshrc` from the workspace is harmless...
        assert!(check_write_targets("echo x >> .zshrc", std::slice::from_ref(&ws)).is_ok());
        // ...but not when the command may run in $HOME (FA-5).
        assert!(check_write_targets("echo x >> .zshrc", &[ws.clone(), home.clone()]).is_err());
    }

    #[test]
    fn cwd_changing_words_are_found_anywhere() {
        assert!(changes_directory("cd ~ && ls"));
        assert!(changes_directory("while true; do cd ~; break; done"));
        assert!(changes_directory("pushd /x"));
        assert!(changes_directory("source env.sh"));
        assert!(changes_directory(". env.sh"));
        assert!(changes_directory("ls; . ./env.sh"));
        assert!(changes_directory("eval \"$x\""));
        // Spellings no word scan reads: the AST flags the command name.
        assert!(changes_directory("\"c\"\"d\" ~ && echo hi > out.txt"));
        assert!(changes_directory("c\\d ~"));
        assert!(changes_directory("builtin 'cd' ~"));
        assert!(changes_directory("trap 'cd ~' DEBUG"));
        assert!(!changes_directory("cp a ."));
        assert!(!changes_directory("echo abcd > cdrom.txt"));
        assert!(!changes_directory("cat > f <<'EOF'\ncd /\nEOF"));
    }

    #[test]
    fn warnings_are_attributed_to_their_validator() {
        let c = classify_command("eval \"$cmd\" > src/f");
        let kinds: Vec<WarningKind> = c.warnings.iter().map(|w| w.kind).collect();
        assert!(kinds.contains(&WarningKind::Redirection));
        assert!(kinds.contains(&WarningKind::Other));
        let c = classify_command("cat > f <<'EOF'\nx\nEOF");
        let kinds: Vec<WarningKind> = c.warnings.iter().map(|w| w.kind).collect();
        assert!(kinds.contains(&WarningKind::Newline));
        assert!(c.redirects.is_some());
        // The merged message format is unchanged.
        assert!(matches!(c.verdict, SecurityVerdict::Warn(ref m) if m.contains("more warning")));
    }
}
