//! Edit-tool anchor bookkeeping: content tags minted on read/write and
//! validated on edit, no-op loop guard, bounded observed-files store, and
//! the diagnostic / re-grounding renders.
//!
//! Pure state machines and string renders — no I/O. The host tool layer
//! owns the state and keys it by canonical path.

/// Upper bound on the content size that gets a minted tag (upstream
/// `SNAPSHOT_MAX_BYTES`): hashing unbounded outputs buys nothing for the
/// anchor check and can be abused by pointing read at huge files.
pub const SNAPSHOT_MAX_BYTES: usize = 4 * 1024 * 1024;

/// After this many consecutive byte-identical no-op edits on one path the
/// guard escalates from a soft hint to a hard error (upstream `NOOP_HARD_LIMIT`).
pub const NOOP_HARD_LIMIT: u32 = 3;

/// Default number of observed files retained (bounded, FIFO eviction).
pub const DEFAULT_OBSERVED_CAPACITY: usize = 256;

// ---------------------------------------------------------------------------
// Fingerprints and tags
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit over bytes. Not cryptographic: tags only need to detect
/// "the file changed since the anchor was minted", deterministically and
/// cheaply.
pub fn content_fingerprint(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// CRLF checkouts must mint the same tag as LF ones (upstream `normalizeToLF`).
pub fn normalize_lf(text: &str) -> String {
    text.replace("\r\n", "\n")
}

/// Fingerprint of the LF-normalized content — the identity the store keeps.
pub fn normalized_fingerprint(content: &str) -> u64 {
    content_fingerprint(&normalize_lf(content))
}

/// 4-hex content tag (low 16 bits of the normalized fingerprint). upstream-width;
/// collisions resolve last-mint-wins in the store.
pub fn content_tag(content: &str) -> String {
    format!("{:04x}", normalized_fingerprint(content) as u16)
}

/// The anchor header stamped into read/write/edit output: `[path#TAG]`.
pub fn tag_header(path_display: &str, tag: &str) -> String {
    format!("[{}#{}]", path_display, tag)
}

/// Stable fingerprint of the raw model-authored edit payload (old and new
/// string bytes). upstream hashes the raw patch input: a different payload — even
/// whitespace-only — is progress and earns a fresh soft hint; re-issuing the
/// same bytes after a warning is the loop we break.
pub fn edit_input_fingerprint(old_string: &str, new_string: &str) -> u64 {
    // Length-prefixed encoding: a bare separator would be ambiguous when the
    // strings themselves embed NUL bytes.
    let combined = format!("{}\0{}\0{}", old_string.len(), old_string, new_string);
    content_fingerprint(&combined)
}

// ---------------------------------------------------------------------------
// Observed files (stale-anchor detection)
// ---------------------------------------------------------------------------

/// Anchor state of one file relative to what this app last served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorStatus {
    /// Never observed: nothing to compare against.
    Unobserved,
    /// Content hashes to the last-served fingerprint.
    Fresh,
    /// The file changed since this app last read/wrote/edited it.
    Drifted,
}

#[derive(Debug, Clone)]
struct ObservedEntry {
    fingerprint: u64,
    tag: String,
}

/// Bounded map of canonical path to the last-served content identity
/// (fingerprint + tag). FIFO eviction approximates the reference implementation's bounded LRU.
#[derive(Debug)]
pub struct ObservedFiles {
    map: std::collections::HashMap<String, ObservedEntry>,
    order: std::collections::VecDeque<String>,
    capacity: usize,
}

impl ObservedFiles {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_OBSERVED_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Mint (or re-mint) the anchor for `path_key` from `content`. Returns
    /// the new 4-hex tag, or `None` when the content exceeds
    /// [`SNAPSHOT_MAX_BYTES`] (nothing is recorded then).
    pub fn record(&mut self, path_key: &str, content: &str) -> Option<String> {
        if content.len() > SNAPSHOT_MAX_BYTES {
            return None;
        }
        let fingerprint = normalized_fingerprint(content);
        let tag = format!("{:04x}", fingerprint as u16);
        if !self.map.contains_key(path_key) {
            if self.order.len() >= self.capacity
                && let Some(evicted) = self.order.pop_front()
            {
                self.map.remove(&evicted);
            }
            self.order.push_back(path_key.to_string());
        }
        self.map.insert(
            path_key.to_string(),
            ObservedEntry {
                fingerprint,
                tag: tag.clone(),
            },
        );
        Some(tag)
    }

    /// Compare live `content` against the last-served anchor.
    pub fn check(&self, path_key: &str, content: &str) -> AnchorStatus {
        match self.map.get(path_key) {
            None => AnchorStatus::Unobserved,
            Some(entry) if entry.fingerprint == normalized_fingerprint(content) => {
                AnchorStatus::Fresh
            }
            Some(_) => AnchorStatus::Drifted,
        }
    }

    /// Tag most recently minted for `path_key`, if any.
    pub fn tag_for(&self, path_key: &str) -> Option<&str> {
        self.map.get(path_key).map(|entry| entry.tag.as_str())
    }
}

impl Default for ObservedFiles {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// No-op loop guard
// ---------------------------------------------------------------------------

/// Outcome of recording one no-op edit against the guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoopRecordResult {
    /// Consecutive identical no-op count, including the current one.
    pub count: u32,
    /// True once `count >= NOOP_HARD_LIMIT`; the caller MUST escalate.
    pub escalate: bool,
}

/// Per-path guard against looping on byte-identical no-op edits, keyed by
/// the raw input fingerprint. Any real commit on the path resets it.
#[derive(Debug, Default)]
pub struct NoopLoopGuard {
    entries: std::collections::HashMap<String, (u64, u32)>,
}

impl NoopLoopGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a no-op attempt for `path_key` keyed by `input_hash`.
    pub fn record(&mut self, path_key: &str, input_hash: u64) -> NoopRecordResult {
        let count = match self.entries.get(path_key) {
            Some(&(hash, previous)) if hash == input_hash => previous + 1,
            _ => 1,
        };
        self.entries
            .insert(path_key.to_string(), (input_hash, count));
        NoopRecordResult {
            count,
            escalate: count >= NOOP_HARD_LIMIT,
        }
    }

    /// Clear the counter for `path_key` after a non-noop commit.
    pub fn reset(&mut self, path_key: &str) {
        self.entries.remove(path_key);
    }
}

// ---------------------------------------------------------------------------
// Renders
// ---------------------------------------------------------------------------

/// Clip `line` to `max_chars` (character-boundary safe), appending an
/// ellipsis when clipped. Unlike `context_pruning::ellipsize` this preserves
/// internal whitespace — code lines must not be reflowed.
pub fn clip_line(line: &str, max_chars: usize) -> String {
    if line.chars().count() <= max_chars {
        return line.to_string();
    }
    let mut clipped: String = line.chars().take(max_chars.saturating_sub(1)).collect();
    clipped.push('\u{2026}');
    clipped
}

/// Numbered preview windows around `match_lines` (1-based) for ambiguity
/// errors, ported from the reference implementation's EditMatchError formatting: `context` lines
/// before/after each match, `>` marking match starts, lines clipped.
/// Overlapping windows merge so near-duplicates render once.
pub fn render_preview_windows(
    content: &str,
    match_lines: &[usize],
    context: usize,
    max_line_chars: usize,
) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    if total == 0 || match_lines.is_empty() {
        return String::new();
    }

    let mut windows: Vec<(usize, usize)> = match_lines
        .iter()
        .filter(|&&line| line >= 1 && line <= total)
        .map(|&line| {
            let from = line.saturating_sub(context + 1);
            let to = (line + context).min(total); // exclusive
            (from, to)
        })
        .collect();
    windows.sort_unstable();

    let mut merged: Vec<(usize, usize)> = Vec::new();
    for window in windows {
        match merged.last_mut() {
            Some(last) if window.0 <= last.1 => last.1 = last.1.max(window.1),
            _ => merged.push(window),
        }
    }

    let mut blocks: Vec<String> = Vec::new();
    for (from, to) in merged {
        let mut block = String::new();
        for (offset, line) in lines[from..to].iter().enumerate() {
            let line_number = from + offset + 1;
            let marker = if match_lines.contains(&line_number) {
                '>'
            } else {
                ' '
            };
            block.push_str(&format!(
                "{}{:>5} | {}\n",
                marker,
                line_number,
                clip_line(line, max_line_chars)
            ));
        }
        blocks.push(block.trim_end_matches('\n').to_string());
    }
    blocks.join("\n\n")
}

/// Numbered context window around a freshly edited region (re-grounding):
/// `context` lines before `start_line` and after `end_line` (both 1-based,
/// inclusive), lines clipped. Mirrors the read tool's `{:>6}\t` numbering so
/// the model anchors the next edit without a fresh read.
pub fn render_context_window(
    content: &str,
    start_line: usize,
    end_line: usize,
    context: usize,
    max_line_chars: usize,
) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    if total == 0 {
        return String::new();
    }

    let start = start_line.clamp(1, total);
    let end = end_line.clamp(start, total);
    let from = start.saturating_sub(context + 1); // 0-based inclusive
    let to = (end + context).min(total); // 0-based exclusive

    let mut parts: Vec<String> = Vec::with_capacity(to - from + 1);
    for (offset, line) in lines[from..to].iter().enumerate() {
        parts.push(format!(
            "{:>6}\t{}",
            from + offset + 1,
            clip_line(line, max_line_chars)
        ));
    }
    if from > 0 || to < total {
        parts.push(format!(
            "... ({} more lines, {} total)",
            total - (to - from),
            total
        ));
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_stable_and_discriminating() {
        assert_eq!(content_fingerprint("abc"), content_fingerprint("abc"));
        assert_ne!(content_fingerprint("abc"), content_fingerprint("abd"));
        assert_eq!(content_fingerprint(""), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn tags_are_four_hex_and_crlf_insensitive() {
        let lf = "line1\nline2\n";
        let crlf = "line1\r\nline2\r\n";
        assert_eq!(content_tag(lf), content_tag(crlf));
        let tag = content_tag(lf);
        assert_eq!(tag.len(), 4);
        assert!(tag.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(tag, content_tag("line1\nline3\n"));
    }

    #[test]
    fn observed_files_status_transitions() {
        let mut observed = ObservedFiles::new();
        assert_eq!(observed.check("/f", "a"), AnchorStatus::Unobserved);

        let tag = observed.record("/f", "a").unwrap();
        assert_eq!(observed.check("/f", "a"), AnchorStatus::Fresh);
        assert_eq!(observed.tag_for("/f"), Some(tag.as_str()));

        assert_eq!(observed.check("/f", "b"), AnchorStatus::Drifted);
        // CRLF drift must not count as a change.
        observed.record("/f", "a\r\n").unwrap();
        assert_eq!(observed.check("/f", "a\n"), AnchorStatus::Fresh);
    }

    #[test]
    fn observed_files_skip_oversized_content() {
        let mut observed = ObservedFiles::new();
        let huge = "x".repeat(SNAPSHOT_MAX_BYTES + 1);
        assert!(observed.record("/f", &huge).is_none());
        assert_eq!(observed.check("/f", &huge), AnchorStatus::Unobserved);
    }

    #[test]
    fn observed_files_evict_fifo() {
        let mut observed = ObservedFiles::with_capacity(2);
        observed.record("/a", "a").unwrap();
        observed.record("/b", "b").unwrap();
        observed.record("/c", "c").unwrap();
        assert_eq!(observed.check("/a", "a"), AnchorStatus::Unobserved);
        assert_eq!(observed.check("/b", "b"), AnchorStatus::Fresh);
        assert_eq!(observed.check("/c", "c"), AnchorStatus::Fresh);
    }

    #[test]
    fn noop_guard_escalates_at_limit_and_resets() {
        let mut guard = NoopLoopGuard::new();
        let first = guard.record("/f", 7);
        assert_eq!(
            first,
            NoopRecordResult {
                count: 1,
                escalate: false
            }
        );
        let second = guard.record("/f", 7);
        assert_eq!(
            second,
            NoopRecordResult {
                count: 2,
                escalate: false
            }
        );
        let third = guard.record("/f", 7);
        assert_eq!(
            third,
            NoopRecordResult {
                count: 3,
                escalate: true
            }
        );

        // A different payload is progress: fresh soft hint.
        let changed = guard.record("/f", 8);
        assert_eq!(changed.count, 1);
        assert!(!changed.escalate);

        // A real commit resets entirely.
        guard.reset("/f");
        assert_eq!(guard.record("/f", 8).count, 1);
    }

    #[test]
    fn clip_line_is_char_boundary_safe() {
        assert_eq!(clip_line("short", 10), "short");
        assert_eq!(clip_line("1234567890", 10), "1234567890");
        assert_eq!(clip_line("1234567890A", 10), "123456789\u{2026}");
        // Multibyte characters must not panic or split.
        let cjk = "\u{4f60}\u{597d}\u{4e16}\u{754c}\u{4f60}\u{597d}\u{4e16}\u{754c}";
        let clipped = clip_line(cjk, 5);
        assert_eq!(clipped.chars().count(), 5);
        assert!(clipped.ends_with('\u{2026}'));
    }

    #[test]
    fn preview_windows_merge_overlaps_and_mark_matches() {
        let content = "one\ntwo\nthree\nfour\nfive\nsix\nseven";
        // Matches on lines 2 and 3 with context 2 merge into one window.
        let rendered = render_preview_windows(content, &[2, 3], 2, 80);
        assert!(rendered.contains(">    2 | two"), "{rendered}");
        assert!(rendered.contains(">    3 | three"), "{rendered}");
        assert!(rendered.contains("     1 | one"), "{rendered}");
        assert!(rendered.contains("     5 | five"), "{rendered}");
        assert!(!rendered.contains("\n\n"), "overlapping windows must merge");

        // Distant matches render as separate blocks.
        let split = render_preview_windows(content, &[1, 7], 1, 80);
        assert!(split.contains("\n\n"), "distant windows stay separate");
        assert!(split.contains("seven"));
    }

    #[test]
    fn preview_windows_clamp_to_file_bounds() {
        let content = "only\nline";
        let rendered = render_preview_windows(content, &[1], 5, 80);
        assert!(rendered.contains("only"));
        assert!(rendered.contains("line"));
        // Out-of-range match lines are dropped, not panicking.
        assert_eq!(render_preview_windows(content, &[9], 2, 80), "");
    }

    #[test]
    fn preview_windows_clip_long_lines() {
        let content = "a".repeat(200);
        let rendered = render_preview_windows(&content, &[1], 0, 40);
        assert!(rendered.contains('\u{2026}'));
        assert!(rendered.lines().next().unwrap().chars().count() <= 50);
    }

    #[test]
    fn context_window_numbers_and_footers() {
        let content: String = (1..=100)
            .map(|n| format!("line{n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let rendered = render_context_window(&content, 50, 51, 3, 200);
        assert!(rendered.contains("    47\tline47"), "{rendered}");
        assert!(rendered.contains("    50\tline50"));
        assert!(rendered.contains("    54\tline54"));
        assert!(!rendered.contains("    46\t"));
        assert!(!rendered.contains("    55\t"));
        assert!(rendered.contains("... (92 more lines, 100 total)"));
    }

    #[test]
    fn context_window_clamps_out_of_range_region() {
        let content = "a\nb\nc";
        // Region beyond EOF clamps instead of panicking.
        let rendered = render_context_window(content, 99, 120, 1, 80);
        assert!(rendered.contains("b"));
        assert!(rendered.contains("c"));
    }

    #[test]
    fn edit_input_fingerprint_separates_payloads() {
        assert_eq!(
            edit_input_fingerprint("a", "b"),
            edit_input_fingerprint("a", "b")
        );
        // Whitespace-only difference counts as a new payload.
        assert_ne!(
            edit_input_fingerprint("a", "b"),
            edit_input_fingerprint("a ", "b")
        );
        // Embedded NUL cannot forge collisions between the two fields.
        assert_ne!(
            edit_input_fingerprint("a\0b", "c"),
            edit_input_fingerprint("a", "b\0c")
        );
    }
}
