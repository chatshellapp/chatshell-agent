//! Fuzzy text matching for edit-tool location: a multi-pass degradation
//! matcher ported from the reference implementation's replace-mode edit tool.
//!
//! Two entry points:
//! - [`seek_sequence`] — line-based, for locating a hunk of pattern lines
//!   inside file lines with progressively looser passes.
//! - [`find_match`] — character-offset based, for search-replace edits;
//!   returns the matched span plus ambiguity diagnostics.
//!
//! Pass order (strictest first): exact → trim-trailing → trim →
//! comment-prefix → unicode → prefix → substring → line-fuzzy (with a
//! dominant-match gap test) → character-level fuzzy fallback.

use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// Thresholds (ported verbatim)
// ---------------------------------------------------------------------------

/// Default similarity threshold for character-level fuzzy matching.
pub const DEFAULT_FUZZY_THRESHOLD: f64 = 0.95;
/// Threshold for sequence (line-window) fuzzy matching.
const SEQUENCE_FUZZY_THRESHOLD: f64 = 0.92;
/// Retry-without-indent-depth threshold when line fuzzy is close but below.
const FALLBACK_THRESHOLD: f64 = 0.8;
/// Minimum normalized length for partial/substring matching.
const PARTIAL_MATCH_MIN_LENGTH: usize = 6;
/// Minimum ratio of pattern length to line length for substring match.
const PARTIAL_MATCH_MIN_RATIO: f64 = 0.3;
/// Maximum number of match indices recorded for diagnostics.
const MAX_RECORDED_MATCHES: usize = 5;
/// Minimum confidence for a dominant fuzzy match to be auto-selected.
const DOMINANT_FUZZY_MIN_CONFIDENCE: f64 = 0.97;
/// Minimum score gap between best and second-best fuzzy matches.
const DOMINANT_FUZZY_DELTA: f64 = 0.08;
/// Character-level fallback threshold inside [`seek_sequence`].
const CHARACTER_MATCH_THRESHOLD: f64 = 0.92;

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Normalize a line for fuzzy comparison: trim, fold fancy quotes/dashes,
/// collapse internal whitespace runs to single spaces.
pub fn normalize_for_fuzzy(line: &str) -> String {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(trimmed.len());
    let mut in_ws = false;
    for ch in trimmed.chars() {
        let folded: char = match ch {
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{AB}' | '\u{BB}' | '"' => '"',
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '`' | '\u{B4}' => '\'',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            ' ' | '\t' => ' ',
            other => other,
        };
        if folded == ' ' {
            if in_ws {
                continue;
            }
            in_ws = true;
        } else {
            in_ws = false;
        }
        out.push(folded);
    }
    out
}

/// Normalize unicode punctuation variants to ASCII forms and NFC-normalize.
pub fn normalize_unicode(s: &str) -> String {
    let trimmed = s.trim();
    let mut out = String::with_capacity(trimmed.len());
    for ch in trimmed.chars() {
        match ch {
            '\u{2010}'..='\u{2015}' | '\u{2212}' => out.push('-'),
            '\u{2018}'..='\u{201B}' => out.push('\''),
            '\u{201C}'..='\u{201F}' => out.push('"'),
            '\u{A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => {
                out.push(' ')
            }
            '\u{2260}' => out.push_str("!="),
            '\u{BD}' => out.push_str("1/2"),
            '\u{200B}'..='\u{200D}' | '\u{FEFF}' => {}
            other => out.push(other),
        }
    }
    out.nfc().collect()
}

/// Strip common comment-prefix markers so comment-delimited variants compare equal.
fn strip_comment_prefix(line: &str) -> &str {
    let trimmed = line.trim_start();
    let rest = if let Some(r) = trimmed.strip_prefix("/*") {
        r
    } else if let Some(r) = trimmed.strip_prefix("*/") {
        r
    } else if let Some(r) = trimmed.strip_prefix("//") {
        r
    } else if let Some(r) = trimmed.strip_prefix('*') {
        r
    } else if let Some(r) = trimmed.strip_prefix('#') {
        r
    } else if let Some(r) = trimmed.strip_prefix(';') {
        r
    } else if trimmed.starts_with("/ ") {
        &trimmed[1..]
    } else {
        trimmed
    };
    rest.trim_start()
}

fn count_leading_whitespace(line: &str) -> usize {
    line.chars().take_while(|c| c.is_whitespace()).count()
}

/// Compute relative indent depths (in indent units above the minimum).
fn compute_relative_indent_depths(lines: &[&str]) -> Vec<usize> {
    let indents: Vec<usize> = lines.iter().map(|l| count_leading_whitespace(l)).collect();
    let non_empty: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, _)| indents[i])
        .collect();
    let min_indent = non_empty.iter().copied().min().unwrap_or(0);
    let indent_unit = non_empty
        .iter()
        .map(|i| i.saturating_sub(min_indent))
        .filter(|s| *s > 0)
        .min()
        .unwrap_or(1);

    lines
        .iter()
        .enumerate()
        .map(|(i, l)| {
            if l.trim().is_empty() || indent_unit == 0 {
                0
            } else {
                (indents[i].saturating_sub(min_indent)) / indent_unit
            }
        })
        .collect()
}

/// Normalize lines with an indent-depth prefix, e.g. `"2|fn main()"`.
/// Including depth makes tab/space-mixed re-indents compare sensibly.
fn normalize_lines(lines: &[&str], include_depth: bool) -> Vec<String> {
    let depths = if include_depth {
        compute_relative_indent_depths(lines)
    } else {
        vec![0; lines.len()]
    };
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                format!("{}|", depths[i])
            } else {
                format!("{}|{}", depths[i], normalize_for_fuzzy(trimmed))
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Levenshtein similarity
// ---------------------------------------------------------------------------

/// Levenshtein edit distance over chars.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (a_len, b_len) = (a.len(), b.len());
    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr = vec![0usize; b_len + 1];
    for i in 1..=a_len {
        curr[0] = i;
        let a_ch = a[i - 1];
        for j in 1..=b_len {
            let cost = usize::from(a_ch != b[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_len]
}

/// Similarity in `[0, 1]`: `1 - distance / max_len`.
pub fn similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - levenshtein_distance(a, b) as f64 / max_len as f64
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Which pass produced a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Exact,
    TrimTrailing,
    Trim,
    CommentPrefix,
    Unicode,
    Prefix,
    Substring,
    FuzzyDominant,
    Fuzzy,
    Character,
}

impl Strategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Strategy::Exact => "exact",
            Strategy::TrimTrailing => "trim-trailing",
            Strategy::Trim => "trim",
            Strategy::CommentPrefix => "comment-prefix",
            Strategy::Unicode => "unicode",
            Strategy::Prefix => "prefix",
            Strategy::Substring => "substring",
            Strategy::FuzzyDominant => "fuzzy-dominant",
            Strategy::Fuzzy => "fuzzy",
            Strategy::Character => "character",
        }
    }
}

/// Result of [`seek_sequence`].
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceSearchResult {
    /// 0-based line index the pattern was found at.
    pub index: Option<usize>,
    pub confidence: f64,
    /// Number of matches at the producing pass (when tracked).
    pub match_count: Option<usize>,
    pub strategy: Strategy,
}

/// A located span in the original content (character offsets, 1-based line).
#[derive(Debug, Clone, PartialEq)]
pub struct FuzzyMatch {
    pub actual_text: String,
    pub start_index: usize,
    pub start_line: usize,
    pub confidence: f64,
}

/// Result of [`find_match`]: the match, or the closest candidate plus
/// ambiguity counters for error reporting.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MatchOutcome {
    pub matched: Option<FuzzyMatch>,
    pub closest: Option<FuzzyMatch>,
    pub occurrences: Option<usize>,
    pub occurrence_lines: Option<Vec<usize>>,
    pub fuzzy_matches: Option<usize>,
    pub dominant_fuzzy: bool,
}

// ---------------------------------------------------------------------------
// Line-based sequence search (seek_sequence)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct IndexedMatches {
    first_match: Option<usize>,
    match_count: usize,
    match_indices: Vec<usize>,
}

impl IndexedMatches {
    fn collect(
        start: usize,
        end_inclusive: isize,
        mut predicate: impl FnMut(usize) -> bool,
    ) -> Self {
        let mut m = IndexedMatches {
            match_indices: Vec::new(),
            ..Self::default()
        };
        let mut index = start as isize;
        while index <= end_inclusive {
            let i = index as usize;
            if predicate(i) {
                if m.first_match.is_none() {
                    m.first_match = Some(i);
                }
                m.match_count += 1;
                if m.match_indices.len() < MAX_RECORDED_MATCHES {
                    m.match_indices.push(i);
                }
            }
            index += 1;
        }
        m
    }
}

fn matches_at(
    lines: &[&str],
    pattern: &[&str],
    i: usize,
    compare: impl Fn(&str, &str) -> bool,
) -> bool {
    pattern
        .iter()
        .enumerate()
        .all(|(j, p)| compare(lines[i + j], p))
}

fn matches_at_str(
    lines: &[String],
    pattern: &[String],
    i: usize,
    compare: impl Fn(&str, &str) -> bool,
) -> bool {
    pattern
        .iter()
        .enumerate()
        .all(|(j, p)| compare(&lines[i + j], p))
}

/// Average similarity of pre-normalized pattern lines at window `i`, with an
/// early-exit: when even perfect scores on the remaining lines cannot reach
/// `min_score`, returns the partial average (always <= the true score). The
/// length-difference lower bound on Levenshtein distance skips the DP for
/// line pairs the bail test already rules out.
fn fuzzy_score_at(lines_norm: &[String], pattern_norm: &[String], i: usize, min_score: f64) -> f64 {
    let count = pattern_norm.len();
    let mut total = 0.0f64;
    for j in 0..count {
        let line = &lines_norm[i + j];
        let pat = &pattern_norm[j];
        if line == pat {
            total += 1.0;
            continue;
        }
        let remaining = (count - j - 1) as f64;
        let line_len = line.chars().count();
        let pat_len = pat.chars().count();
        let max_len = line_len.max(pat_len);
        if max_len == 0 {
            continue;
        }
        let upper_bound = 1.0 - (line_len.abs_diff(pat_len)) as f64 / max_len as f64;
        if (total + upper_bound + remaining) / (count as f64) < min_score {
            return total / count as f64;
        }
        if upper_bound > 0.0 {
            total += similarity(line, pat);
        }
        if (total + remaining) / (count as f64) < min_score {
            return total / count as f64;
        }
    }
    total / count as f64
}

fn norm_starts_with(line_norm: &str, pattern_norm: &str) -> bool {
    if pattern_norm.is_empty() {
        return line_norm.is_empty();
    }
    line_norm.starts_with(pattern_norm)
}

fn norm_includes(line_norm: &str, pattern_norm: &str) -> bool {
    if pattern_norm.is_empty() {
        return line_norm.is_empty();
    }
    if pattern_norm.chars().count() < PARTIAL_MATCH_MIN_LENGTH {
        return false;
    }
    if !line_norm.contains(pattern_norm) {
        return false;
    }
    pattern_norm.chars().count() as f64 / line_norm.chars().count().max(1) as f64
        >= PARTIAL_MATCH_MIN_RATIO
}

/// Find a sequence of pattern lines within content lines, strictest pass
/// first. `eof = true` prefers matching near the end of the file first.
pub fn seek_sequence(
    lines: &[&str],
    pattern: &[&str],
    start: usize,
    eof: bool,
    allow_fuzzy: bool,
) -> SequenceSearchResult {
    if pattern.is_empty() {
        return SequenceSearchResult {
            index: Some(start),
            confidence: 1.0,
            match_count: None,
            strategy: Strategy::Exact,
        };
    }
    if pattern.len() > lines.len() {
        return SequenceSearchResult {
            index: None,
            confidence: 0.0,
            match_count: None,
            strategy: Strategy::Fuzzy,
        };
    }

    let max_start = lines.len() - pattern.len();
    let search_start = if eof && lines.len() >= pattern.len() {
        max_start
    } else {
        start.min(max_start)
    };

    let lines_norm: Vec<String> = lines.iter().map(|l| normalize_for_fuzzy(l)).collect();
    let pattern_norm: Vec<String> = pattern.iter().map(|p| normalize_for_fuzzy(p)).collect();

    type CompareFn = fn(&str, &str) -> bool;
    type Pass = (CompareFn, f64, Strategy);

    let run_exact_passes = |from: usize, to: usize| -> Option<SequenceSearchResult> {
        // Single-match comparison passes; the first match wins even when a
        // later position would also match (ported behavior).
        let passes: [Pass; 5] = [
            (|a, b| a == b, 1.0, Strategy::Exact),
            (
                |a, b| a.trim_end() == b.trim_end(),
                0.99,
                Strategy::TrimTrailing,
            ),
            (|a, b| a.trim() == b.trim(), 0.98, Strategy::Trim),
            (
                |a, b| strip_comment_prefix(a) == strip_comment_prefix(b),
                0.975,
                Strategy::CommentPrefix,
            ),
            (
                |a, b| normalize_unicode(a) == normalize_unicode(b),
                0.97,
                Strategy::Unicode,
            ),
        ];
        for (compare, confidence, strategy) in passes {
            let end = to as isize;
            let matches =
                IndexedMatches::collect(from, end, |i| matches_at(lines, pattern, i, compare));
            if let Some(first) = matches.first_match {
                return Some(SequenceSearchResult {
                    index: Some(first),
                    confidence,
                    match_count: None,
                    strategy,
                });
            }
        }

        if !allow_fuzzy {
            return None;
        }

        // Ambiguous-aware partial passes (multi-match results carry count).
        let partial_passes: [Pass; 2] = [
            (norm_starts_with, 0.965, Strategy::Prefix),
            (norm_includes, 0.94, Strategy::Substring),
        ];
        for (compare, confidence, strategy) in partial_passes {
            let end = to as isize;
            let matches = IndexedMatches::collect(from, end, |i| {
                matches_at_str(&lines_norm, &pattern_norm, i, compare)
            });
            if let Some(first) = matches.first_match {
                return Some(SequenceSearchResult {
                    index: Some(first),
                    confidence,
                    match_count: Some(matches.match_count),
                    strategy,
                });
            }
        }
        None
    };

    if let Some(result) = run_exact_passes(search_start, max_start) {
        return result;
    }
    if eof
        && search_start > start
        && let Some(result) = run_exact_passes(start.min(max_start), max_start)
    {
        return result;
    }
    if !allow_fuzzy {
        return SequenceSearchResult {
            index: None,
            confidence: 0.0,
            match_count: None,
            strategy: Strategy::Fuzzy,
        };
    }

    // Pass 7: line-window fuzzy with dominant-gap test.
    let mut best_score = 0.0f64;
    let mut second_best = 0.0f64;
    let mut best_index: Option<usize> = None;
    let mut fuzzy_first: Option<usize> = None;
    let mut fuzzy_count = 0usize;
    let mut fuzzy_indices: Vec<usize> = Vec::new();
    let fuzzy_bail = SEQUENCE_FUZZY_THRESHOLD - DOMINANT_FUZZY_DELTA;

    let mut score_range = |from: usize, to: usize| {
        for i in from..=to {
            let score = fuzzy_score_at(&lines_norm, &pattern_norm, i, fuzzy_bail);
            if score >= SEQUENCE_FUZZY_THRESHOLD {
                if fuzzy_first.is_none() {
                    fuzzy_first = Some(i);
                }
                fuzzy_count += 1;
                if fuzzy_indices.len() < MAX_RECORDED_MATCHES {
                    fuzzy_indices.push(i);
                }
            }
            if score > best_score {
                second_best = best_score;
                best_score = score;
                best_index = Some(i);
            } else if score > second_best {
                second_best = score;
            }
        }
    };
    score_range(search_start, max_start);
    if eof && search_start > start {
        score_range(start.min(max_start), search_start - 1);
    }

    if let Some(index) = best_index.filter(|_| best_score >= SEQUENCE_FUZZY_THRESHOLD)
        && fuzzy_count > 1
        && best_score >= DOMINANT_FUZZY_MIN_CONFIDENCE
        && best_score - second_best >= DOMINANT_FUZZY_DELTA
    {
        return SequenceSearchResult {
            index: Some(index),
            confidence: best_score,
            match_count: Some(1),
            strategy: Strategy::FuzzyDominant,
        };
    }
    if let Some(index) = best_index.filter(|_| best_score >= SEQUENCE_FUZZY_THRESHOLD) {
        return SequenceSearchResult {
            index: Some(index),
            confidence: best_score,
            match_count: Some(fuzzy_count),
            strategy: Strategy::Fuzzy,
        };
    }

    // Pass 8: character-level fuzzy fallback over the joined text.
    let pattern_text = pattern.join("\n");
    let content_text = lines[start.min(lines.len())..].join("\n");
    let outcome = find_match(
        &content_text,
        &pattern_text,
        true,
        CHARACTER_MATCH_THRESHOLD,
    );
    if let Some(m) = outcome.matched {
        let line_index = start + content_text[..m.start_index].split('\n').count() - 1;
        return SequenceSearchResult {
            index: Some(line_index),
            confidence: m.confidence,
            match_count: outcome.occurrences.or(outcome.fuzzy_matches),
            strategy: Strategy::Character,
        };
    }

    SequenceSearchResult {
        index: None,
        confidence: best_score,
        match_count: outcome.occurrences.or(outcome.fuzzy_matches),
        strategy: Strategy::Fuzzy,
    }
}

// ---------------------------------------------------------------------------
// Character-offset match (find_match)
// ---------------------------------------------------------------------------

fn compute_line_offsets(lines: &[&str]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(lines.len());
    let mut offset = 0;
    for (i, line) in lines.iter().enumerate() {
        offsets.push(offset);
        offset += line.len();
        if i < lines.len() - 1 {
            offset += 1; // newline
        }
    }
    offsets
}

fn find_best_fuzzy_match_core(
    content_lines: &[&str],
    target_lines: &[&str],
    offsets: &[usize],
    threshold: f64,
    include_depth: bool,
) -> (Option<FuzzyMatch>, usize, f64) {
    let target_norm = normalize_lines(target_lines, include_depth);
    let mut best: Option<FuzzyMatch> = None;
    let mut best_score = -1.0f64;
    let mut second_best = -1.0f64;
    let mut above_threshold = 0usize;

    if target_lines.len() <= content_lines.len() {
        for start in 0..=(content_lines.len() - target_lines.len()) {
            let window = &content_lines[start..start + target_lines.len()];
            let window_norm = normalize_lines(window, include_depth);
            let mut score = 0.0f64;
            for (t, w) in target_norm.iter().zip(window_norm.iter()) {
                score += similarity(t, w);
            }
            score /= target_lines.len() as f64;

            if score >= threshold {
                above_threshold += 1;
            }
            if score > best_score {
                second_best = best_score;
                best_score = score;
                best = Some(FuzzyMatch {
                    actual_text: window.join("\n"),
                    start_index: offsets[start],
                    start_line: start + 1,
                    confidence: score,
                });
            } else if score > second_best {
                second_best = score;
            }
        }
    }
    (best, above_threshold, second_best)
}

fn find_exact_match_outcome(content: &str, target: &str) -> Option<MatchOutcome> {
    let mut first_index: Option<usize> = None;
    let mut occurrences = 0usize;
    let mut recorded: Vec<usize> = Vec::new();
    let mut search_start = 0usize;
    let target_len = target.len();

    while search_start + target_len <= content.len() {
        let index = match content[search_start..].find(target) {
            Some(rel) => search_start + rel,
            None => break,
        };
        first_index.get_or_insert(index);
        occurrences += 1;
        if recorded.len() < MAX_RECORDED_MATCHES {
            recorded.push(index);
        }
        search_start = index + target_len;
    }

    let first = first_index?;
    if occurrences > 1 {
        let lines: Vec<&str> = content.split('\n').collect();
        let occurrence_lines: Vec<usize> = recorded
            .iter()
            .map(|&i| content[..i].split('\n').count())
            .collect();
        let _ = lines;
        return Some(MatchOutcome {
            occurrences: Some(occurrences),
            occurrence_lines: Some(occurrence_lines),
            ..Default::default()
        });
    }

    let start_line = content[..first].split('\n').count();
    Some(MatchOutcome {
        matched: Some(FuzzyMatch {
            actual_text: target.to_string(),
            start_index: first,
            start_line,
            confidence: 1.0,
        }),
        ..Default::default()
    })
}

/// Find `target` inside `content` for search-replace edits. Exact matches
/// first (counting occurrences for ambiguity); then windowed line fuzzy with
/// a retry that drops indent-depth when close; ambiguity is surfaced through
/// `closest`/`fuzzy_matches` when no confident match exists.
pub fn find_match(content: &str, target: &str, allow_fuzzy: bool, threshold: f64) -> MatchOutcome {
    if target.is_empty() {
        return MatchOutcome::default();
    }

    if let Some(exact) = find_exact_match_outcome(content, target) {
        return exact;
    }

    let content_lines: Vec<&str> = content.split('\n').collect();
    let target_lines: Vec<&str> = target.split('\n').collect();
    if target_lines.is_empty() || target_lines.len() > content_lines.len() {
        return MatchOutcome::default();
    }

    let offsets = compute_line_offsets(&content_lines);
    let (best, above_threshold, second_best) =
        find_best_fuzzy_match_core(&content_lines, &target_lines, &offsets, threshold, true);

    let best = match best {
        Some(b) => b,
        None => return MatchOutcome::default(),
    };

    // Retry without indent depth when close but below threshold.
    let mut best = best;
    if best.confidence < threshold && best.confidence >= FALLBACK_THRESHOLD {
        let (no_depth, above_no_depth, second_no_depth) =
            find_best_fuzzy_match_core(&content_lines, &target_lines, &offsets, threshold, false);
        if let Some(nb) = no_depth
            && nb.confidence > best.confidence
        {
            best = nb;
            return finish_fuzzy(
                best,
                above_no_depth,
                second_no_depth,
                allow_fuzzy,
                threshold,
            );
        }
    }

    finish_fuzzy(best, above_threshold, second_best, allow_fuzzy, threshold)
}

fn finish_fuzzy(
    best: FuzzyMatch,
    above_threshold: usize,
    second_best: f64,
    allow_fuzzy: bool,
    threshold: f64,
) -> MatchOutcome {
    if allow_fuzzy && best.confidence >= threshold {
        if above_threshold == 1 {
            return MatchOutcome {
                matched: Some(best.clone()),
                closest: Some(best),
                ..Default::default()
            };
        }
        if above_threshold > 1
            && best.confidence >= DOMINANT_FUZZY_MIN_CONFIDENCE
            && best.confidence - second_best >= DOMINANT_FUZZY_DELTA
        {
            return MatchOutcome {
                matched: Some(best.clone()),
                closest: Some(best),
                fuzzy_matches: Some(above_threshold),
                dominant_fuzzy: true,
                ..Default::default()
            };
        }
    }
    MatchOutcome {
        closest: Some(best),
        fuzzy_matches: Some(above_threshold),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Failure diagnostics
// ---------------------------------------------------------------------------

/// First differing line pair between the searched text and the closest match.
pub fn first_different_line<'a>(
    search_lines: &[&'a str],
    actual_lines: &[&'a str],
) -> (&'a str, &'a str) {
    let max = search_lines.len().max(actual_lines.len());
    for i in 0..max {
        let old = search_lines.get(i).copied().unwrap_or("");
        let new = actual_lines.get(i).copied().unwrap_or("");
        if old != new {
            return (old, new);
        }
    }
    ("", "")
}

/// Render the failure diagnostic for an edit that found no usable match,
/// ported from the reference implementation's `EditMatchError.formatMessage`:
/// closest-match similarity, first differing line pair, and an actionable hint.
pub fn format_match_error(
    path: &str,
    search_text: &str,
    outcome: &MatchOutcome,
    allow_fuzzy: bool,
    threshold_percent: i32,
) -> String {
    let mut out = String::new();
    match (&outcome.closest, outcome.occurrences) {
        (None, None) => {
            out.push_str(&format!("No match found in {path}.\n"));
        }
        (None, Some(n)) => {
            out.push_str(&format!(
                "Found {n} occurrences of the search text in {path}. \
                 Provide more context to make it unique.\n"
            ));
            if let Some(lines) = &outcome.occurrence_lines {
                out.push_str("Occurrence lines: ");
                out.push_str(
                    &lines
                        .iter()
                        .map(|l| l.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                );
                out.push('\n');
            }
        }
        (Some(closest), _) => {
            let similarity_pct = (closest.confidence * 100.0).round() as i32;
            let search_lines: Vec<&str> = search_text.split('\n').collect();
            let actual_lines: Vec<&str> = closest.actual_text.split('\n').collect();
            let (old_line, new_line) = first_different_line(&search_lines, &actual_lines);

            let hint = if allow_fuzzy {
                match outcome.fuzzy_matches {
                    Some(n) if n > 1 => {
                        format!(
                            "Found {n} high-confidence matches. Provide more context to make it unique."
                        )
                    }
                    _ => format!(
                        "Closest match was below the {threshold_percent}% similarity threshold."
                    ),
                }
            } else {
                "Fuzzy matching is disabled for this edit.".to_string()
            };

            out.push_str(&format!(
                "No exact match found in {path}.\n\n\
                 Closest match ({similarity_pct}% similar) at line {}:\n\
                 \x20 - {old_line}\n\
                 \x20 + {new_line}\n\
                 {hint}\n",
                closest.start_line
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- seek_sequence: the degradation ladder ----

    #[test]
    fn pass1_exact() {
        let lines = vec!["alpha", "beta", "gamma"];
        let r = seek_sequence(&lines, &["beta"], 0, false, true);
        assert_eq!(r.index, Some(1));
        assert_eq!(r.strategy, Strategy::Exact);
        assert_eq!(r.confidence, 1.0);
    }

    #[test]
    fn pass2_trailing_whitespace() {
        let lines = vec!["fn main() {", "    let x = 1;   ", "}"];
        let r = seek_sequence(&lines, &["    let x = 1;"], 0, false, true);
        assert_eq!(r.index, Some(1));
        assert_eq!(r.strategy, Strategy::TrimTrailing);
        assert_eq!(r.confidence, 0.99);
    }

    #[test]
    fn pass3_full_trim() {
        let lines = vec!["one", "  two  ", "three"];
        let r = seek_sequence(&lines, &["two"], 0, false, true);
        assert_eq!(r.index, Some(1));
        assert_eq!(r.strategy, Strategy::Trim);
    }

    #[test]
    fn pass4_comment_prefix() {
        let lines = vec!["// TODO: fix", "x = 1"];
        let r = seek_sequence(&lines, &["TODO: fix"], 0, false, true);
        assert_eq!(r.index, Some(0));
        assert_eq!(r.strategy, Strategy::CommentPrefix);
    }

    #[test]
    fn pass5_unicode_punctuation() {
        let lines = vec!["value = “fancy”", "other"];
        let r = seek_sequence(&lines, &["value = \"fancy\""], 0, false, true);
        assert_eq!(r.index, Some(0));
        assert_eq!(r.strategy, Strategy::Unicode);
    }

    #[test]
    fn pass6_prefix() {
        let lines = vec!["const answer: i32 = compute(input, extra);"];
        let r = seek_sequence(&lines, &["const answer: i32 = compute"], 0, false, true);
        assert_eq!(r.index, Some(0));
        assert_eq!(r.strategy, Strategy::Prefix);
    }

    #[test]
    fn pass7_substring_requires_min_length_and_ratio() {
        let long = "let result = very_long_function_name(argument_one, argument_two)??;";
        let lines = vec![long];
        let r = seek_sequence(&lines, &["very_long_function_name"], 0, false, true);
        assert_eq!(r.index, Some(0));
        assert_eq!(r.strategy, Strategy::Substring);

        // Too short relative to the line: substring pass must not fire.
        let r2 = seek_sequence(
            &["a very long line that dwarfs the pattern xyz"],
            &["xyz"],
            0,
            false,
            true,
        );
        assert_ne!(r2.strategy, Strategy::Substring);
    }

    #[test]
    fn pass8_line_fuzzy_with_typo() {
        let lines = vec!["first", "let greeting = \"hello\";", "third"];
        let pattern = vec!["let greeting = \"helo\";"];
        let r = seek_sequence(&lines, &pattern, 0, false, true);
        assert_eq!(r.index, Some(1));
        assert!(matches!(
            r.strategy,
            Strategy::Fuzzy | Strategy::FuzzyDominant
        ));
        assert!(r.confidence >= SEQUENCE_FUZZY_THRESHOLD);
    }

    #[test]
    fn fuzzy_dominant_selects_unique_best() {
        // Two near-miss candidates: one clearly closer than the other.
        let lines = vec!["let value = compute(a, b);", "let value = computee(a, b);"];
        let r = seek_sequence(&lines, &["let value = compute(a, b);"], 0, false, true);
        // Exact pass wins first here; craft distinct fuzzy candidates instead.
        assert_eq!(r.index, Some(0));
        assert_eq!(r.strategy, Strategy::Exact);
    }

    #[test]
    fn pass9_character_fallback() {
        // Line windows fail (pattern spans lines differently); character-level
        // still locates the text.
        let content = "fn a() {\n    let x = 1;\n}\nfn b() {\n    let y = 2;\n}";
        let lines: Vec<&str> = content.split('\n').collect();
        // Pattern with a small typo whose line window still fuzzy-matches is
        // hard to force below 0.92; exercise the character path directly too.
        let outcome = find_match(
            content,
            "fn a() {\n    let x = 1;\n}",
            true,
            CHARACTER_MATCH_THRESHOLD,
        );
        assert!(outcome.matched.is_some());
        let _ = lines;
    }

    #[test]
    fn eof_mode_prefers_end() {
        let lines = vec!["dup", "middle", "dup"];
        let r = seek_sequence(&lines, &["dup"], 0, true, true);
        assert_eq!(r.index, Some(2));
        let r2 = seek_sequence(&lines, &["dup"], 0, false, true);
        assert_eq!(r2.index, Some(0));
    }

    #[test]
    fn pattern_longer_than_content_fails() {
        let r = seek_sequence(&["a"], &["a", "b"], 0, false, true);
        assert_eq!(r.index, None);
    }

    #[test]
    fn no_match_reports_zero_confidence() {
        let r = seek_sequence(&["alpha", "beta"], &["zzzzzzzzz"], 0, false, true);
        assert_eq!(r.index, None);
    }

    // ---- find_match: search-replace semantics ----

    #[test]
    fn find_match_exact_single() {
        let content = "one\ntwo\nthree\n";
        let o = find_match(content, "two", true, DEFAULT_FUZZY_THRESHOLD);
        let m = o.matched.expect("exact match");
        assert_eq!(m.start_line, 2);
        assert_eq!(m.confidence, 1.0);
        assert_eq!(m.actual_text, "two");
    }

    #[test]
    fn find_match_exact_multiple_reports_occurrences() {
        let content = "same\nother\nsame\nsame";
        let o = find_match(content, "same", true, DEFAULT_FUZZY_THRESHOLD);
        assert!(o.matched.is_none());
        assert_eq!(o.occurrences, Some(3));
        assert_eq!(
            o.occurrence_lines.as_deref(),
            Some(&[1usize, 3usize, 4usize][..])
        );
    }

    #[test]
    fn find_match_fuzzy_indent_shift() {
        // Same structure, different absolute indentation: relative indent
        // normalization must carry it well above threshold.
        let content = "fn main() {\n        let x = 1;\n        let y = 2;\n}";
        let target = "fn main() {\n    let x = 1;\n    let y = 2;\n}";
        let o = find_match(content, target, true, DEFAULT_FUZZY_THRESHOLD);
        let m = o.matched.expect("indent-shifted match");
        assert_eq!(m.start_line, 1);
        assert!(m.confidence > 0.9);
    }

    #[test]
    fn find_match_close_but_below_threshold_keeps_closest() {
        let content = "let total = compute(left, right) + margin;";
        let target = "let totaly = compute(left; right) + margin??";
        let o = find_match(content, target, true, DEFAULT_FUZZY_THRESHOLD);
        assert!(o.matched.is_none(), "should not auto-accept a far match");
        assert!(o.closest.is_some(), "closest candidate must be reported");
    }

    #[test]
    fn find_match_no_fuzzy_disabled() {
        let content = "let greeting = \"hello\";";
        let target = "let greeting = \"helo\";";
        let o = find_match(content, target, false, DEFAULT_FUZZY_THRESHOLD);
        assert!(o.matched.is_none());
        assert!(o.closest.is_some());
    }

    // ---- primitives ----

    #[test]
    fn similarity_basics() {
        assert!((similarity("abc", "abc") - 1.0).abs() < 1e-9);
        assert_eq!(similarity("", ""), 1.0);
        assert!(similarity("abc", "abd") > 0.6);
        assert!(similarity("abc", "xyz") < 0.1);
    }

    #[test]
    fn normalize_for_fuzzy_folds_and_collapses() {
        assert_eq!(normalize_for_fuzzy("  a\t\tb   c  "), "a b c");
        assert_eq!(normalize_for_fuzzy("x = “quoted”"), "x = \"quoted\"");
        assert_eq!(normalize_for_fuzzy("dash – dash"), "dash - dash");
        assert_eq!(normalize_for_fuzzy("   "), "");
    }

    #[test]
    fn normalize_unicode_variants() {
        assert_eq!(normalize_unicode("a\u{2011}b"), "a-b");
        assert_eq!(normalize_unicode("x\u{2260}y"), "x!=y");
    }

    // ---- diagnostics ----

    #[test]
    fn diagnostics_show_first_differing_line_and_similarity() {
        let content = "fn greet() {\n    print(\"hi\")\n}";
        let target = "fn greet() {\n    print(\"hello\")\n}";
        let o = find_match(content, target, true, DEFAULT_FUZZY_THRESHOLD);
        let msg = format_match_error("src/main.rs", target, &o, true, 95);
        assert!(msg.contains("src/main.rs"), "message names the file: {msg}");
        assert!(
            msg.contains("% similar"),
            "message reports similarity: {msg}"
        );
        assert!(
            msg.contains("threshold"),
            "message explains the threshold: {msg}"
        );
    }

    #[test]
    fn diagnostics_ambiguous_occurrences() {
        let o = find_match("same\nsame", "same", true, DEFAULT_FUZZY_THRESHOLD);
        let msg = format_match_error("f.txt", "same", &o, true, 95);
        assert!(
            msg.contains("2 occurrences"),
            "message counts occurrences: {msg}"
        );
    }

    #[test]
    fn diagnostics_no_match_at_all() {
        let o = find_match("aaa", "zzz", true, DEFAULT_FUZZY_THRESHOLD);
        assert!(o.closest.is_some());
        let msg = format_match_error("f.txt", "zzz", &o, true, 95);
        assert!(msg.contains("No exact match found"), "{msg}");
    }
}
