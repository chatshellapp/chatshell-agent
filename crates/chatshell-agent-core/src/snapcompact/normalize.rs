//! Text normalization and word wrapping for the snapcompact renderer,
//! ported from the reference implementation `the upstream snapcompact package`:
//! strip ANSI escapes, fold Unicode toward ASCII where the bitmap font
//! covers it, collapse whitespace runs, and mark newline runs with the
//! full-block cell glyph so line structure survives at one-cell cost.

use unicode_normalization::UnicodeNormalization;

/// Zero-width ink toggles understood by the renderer (shift-out/shift-in):
/// text between them prints in dim gray ink without occupying a cell.
pub const DIM_ON: char = '\u{000e}';
pub const DIM_OFF: char = '\u{000f}';

/// Printed in place of newline runs: fills its entire cell with pitch-black
/// ink, so line structure survives whitespace collapsing.
pub const NEWLINE_GLYPH: char = '\u{2588}';

/// Punctuation folds applied before the NFKD fallback for characters with no
/// useful compatibility decomposition (subset of the reference implementation's CHAR_FOLD).
const CHAR_FOLD: &[(char, &str)] = &[
    ('\u{2018}', "'"),
    ('\u{2019}', "'"),
    ('\u{201a}', ","),
    ('\u{201c}', "\""),
    ('\u{201d}', "\""),
    ('\u{2013}', "-"),
    ('\u{2014}', "-"),
    ('\u{2026}', "..."),
    ('\u{2192}', "->"),
    ('\u{2190}', "<-"),
    ('\u{2022}', "*"),
    ('\u{00b7}', "."),
    ('\u{00a0}', " "),
];

fn fold_char(ch: char) -> Option<&'static str> {
    CHAR_FOLD
        .iter()
        .find(|(from, _)| *from == ch)
        .map(|(_, to)| *to)
}

/// Strip ANSI escape sequences (CSI and OSC) from tool output.
pub fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            out.push(ch);
            continue;
        }
        match chars.peek() {
            Some('[') => {
                chars.next();
                for inner in chars.by_ref() {
                    if inner.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            Some(']') => {
                chars.next();
                let mut escaped = false;
                for inner in chars.by_ref() {
                    if escaped {
                        escaped = false;
                        if inner == '\\' {
                            break;
                        }
                        continue;
                    }
                    if inner == '\u{7}' {
                        break;
                    }
                    if inner == '\u{1b}' {
                        escaped = true;
                    }
                }
            }
            _ => {
                // Lone ESC or a two-char escape: drop the following char.
                chars.next();
            }
        }
    }
    out
}

/// Prepare text for rendering: strip ANSI, fold punctuation and Unicode
/// toward ASCII (keeping characters the caller's font can render — pass a
/// coverage predicate; uncovered non-CJK chars drop, uncovered CJK stays for
/// the TrueType fallback), collapse whitespace, and mark newline runs.
///
/// Control characters other than the dim toggles are dropped.
pub fn normalize<F>(text: &str, renderable: F) -> String
where
    F: Fn(char) -> bool,
{
    let stripped = strip_ansi(text);
    let mut out = String::with_capacity(stripped.len());
    let mut last_was_space = false;
    let mut last_was_newline_glyph = false;

    fn push_keeping_space_runs_collapsed(out: &mut String, ch: char, last_space: &mut bool) {
        if ch == ' ' {
            if !*last_space {
                out.push(' ');
                *last_space = true;
            }
        } else {
            out.push(ch);
            *last_space = false;
        }
    }

    for ch in stripped.chars() {
        match ch {
            DIM_ON | DIM_OFF | NEWLINE_GLYPH => {
                out.push(ch);
                last_was_space = false;
                last_was_newline_glyph = false;
            }
            '\n' | '\r' | '\u{2028}' | '\u{2029}' => {
                if !last_was_newline_glyph {
                    out.push(NEWLINE_GLYPH);
                    last_was_newline_glyph = true;
                    last_was_space = false;
                }
            }
            '\t' => push_keeping_space_runs_collapsed(&mut out, ' ', &mut last_was_space),
            c if c.is_control() => {}
            ' ' => push_keeping_space_runs_collapsed(&mut out, ' ', &mut last_was_space),
            c if c.is_ascii() => {
                out.push(c);
                last_was_space = false;
                last_was_newline_glyph = false;
            }
            c => {
                if let Some(folded) = fold_char(c) {
                    for f in folded.chars() {
                        push_keeping_space_runs_collapsed(&mut out, f, &mut last_was_space);
                    }
                    last_was_newline_glyph = false;
                    continue;
                }
                if renderable(c) {
                    out.push(c);
                    last_was_space = false;
                    last_was_newline_glyph = false;
                    continue;
                }
                // NFKD fold: keep the ASCII decomposition when there is one.
                let mut kept = false;
                for decomposed in c.nfkd() {
                    if decomposed.is_ascii() && !decomposed.is_control() {
                        push_keeping_space_runs_collapsed(
                            &mut out,
                            decomposed,
                            &mut last_was_space,
                        );
                        kept = true;
                    }
                }
                if !kept {
                    // Not renderable by either font and no ASCII fold:
                    // keep wide-range and alphabetic characters (they may
                    // still draw through the Silver fallback); drop the
                    // rest (decorative emoji, combining marks).
                    if is_wide(c) || c.is_alphabetic() {
                        out.push(c);
                        last_was_space = false;
                        last_was_newline_glyph = false;
                    }
                } else {
                    last_was_newline_glyph = false;
                }
            }
        }
    }
    out.trim_matches(|c: char| c == ' ' || c == NEWLINE_GLYPH)
        .to_string()
}

/// East Asian Wide / Fullwidth code points (port of the reference
/// implementation's `is_wide`): these occupy two grid cells in narrow
/// bitmap shapes and one full-width Silver cell.
pub fn is_wide(ch: char) -> bool {
    let cp = ch as u32;
    (0x1100..=0x115F).contains(&cp)
        || (0x2E80..=0x2EFF).contains(&cp)
        || (0x2F00..=0x2FDF).contains(&cp)
        || (0x3000..=0x303E).contains(&cp)
        || (0x3041..=0x33FF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x4E00..=0x9FFF).contains(&cp)
        || (0xA000..=0xA4CF).contains(&cp)
        || (0xAC00..=0xD7A3).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0xFE30..=0xFE4F).contains(&cp)
        || (0xFF00..=0xFF60).contains(&cp)
        || (0xFFE0..=0xFFE6).contains(&cp)
        || (0x20000..=0x2FFFD).contains(&cp)
        || (0x30000..=0x3FFFD).contains(&cp)
}

/// Grid cells one character occupies in a narrow bitmap shape: 0 for the
/// zero-width dim toggles, 2 for wide code points, 1 otherwise. The Silver
/// shape sizes every cell full-width, so it always counts 1.
pub fn cell_units(ch: char) -> usize {
    if is_zero_width(ch) {
        0
    } else if is_wide(ch) {
        2
    } else {
        1
    }
}

/// Greedy word wrap at `width` cells. With `wide_cells` (narrow bitmap
/// shapes) wide characters count two cells; with the full-width Silver
/// shape every character counts one. No mid-word breaks except for words
/// longer than the width — and a wide character never straddles the right
/// edge. Explicit [`NEWLINE_GLYPH`] separators force line starts;
/// zero-width dim toggles ride along without consuming cells.
pub fn wrap(text: &str, width: usize, wide_cells: bool) -> Vec<String> {
    let units = |ch: char| -> usize {
        if !wide_cells {
            usize::from(!is_zero_width(ch))
        } else {
            cell_units(ch)
        }
    };
    if width == 0 {
        return vec![String::new()];
    }
    let mut lines = Vec::new();
    for raw_line in text.split(NEWLINE_GLYPH) {
        let mut current = String::new();
        let mut cells = 0usize;
        for word in raw_line.split(' ') {
            let word_cells: usize = word.chars().map(units).sum();
            let gap = if current.is_empty() { 0 } else { 1 };
            if cells + gap + word_cells <= width {
                if gap == 1 {
                    current.push(' ');
                    cells += 1;
                }
                current.push_str(word);
                cells += word_cells;
            } else if word_cells <= width {
                lines.push(current);
                current = word.to_string();
                cells = word_cells;
            } else {
                // Hard-split overlong words.
                if !current.is_empty() {
                    lines.push(std::mem::take(&mut current));
                }
                cells = 0;
                for ch in word.chars() {
                    let ch_units = units(ch);
                    if cells + ch_units > width {
                        lines.push(std::mem::take(&mut current));
                        cells = 0;
                    }
                    cells += ch_units;
                    current.push(ch);
                }
            }
        }
        lines.push(current);
    }
    lines.into_iter().filter(|line| !line.is_empty()).collect()
}

/// Zero-width renderer control characters.
pub fn is_zero_width(ch: char) -> bool {
    matches!(ch, DIM_ON | DIM_OFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ascii_only(ch: char) -> bool {
        ch.is_ascii()
    }

    #[test]
    fn ansi_is_stripped() {
        assert_eq!(strip_ansi("\u{1b}[31mred\u{1b}[0m"), "red");
        assert_eq!(strip_ansi("\u{1b}]0;title\u{7}rest"), "rest");
        assert_eq!(strip_ansi("plain"), "plain");
    }

    #[test]
    fn whitespace_collapses_and_newlines_become_blocks() {
        let out = normalize("hello   world\n\n\nnext\ttab", ascii_only);
        assert_eq!(out, "hello world\u{2588}next tab");
    }

    #[test]
    fn unicode_folds_toward_ascii() {
        let out = normalize("caf\u{e9} \u{201c}quoted\u{201d} \u{2014} dash", ascii_only);
        assert_eq!(out, "cafe \"quoted\" - dash");
        assert_eq!(normalize("\u{ff01}", ascii_only), "!");
    }

    #[test]
    fn dim_toggles_survive_at_zero_width() {
        let out = normalize("\u{000e}tool noise\u{000f} plain", ascii_only);
        assert_eq!(out, "\u{000e}tool noise\u{000f} plain");
        assert_eq!(wrap(&out, 80, true).len(), 1);
    }

    #[test]
    fn cjk_survives_for_the_ttf_fallback() {
        let out = normalize("\u{4f60}\u{597d} world", ascii_only);
        assert!(out.contains('\u{4f60}'));
        assert!(out.contains("world"));
    }

    #[test]
    fn decorative_emoji_drop_rather_than_placeholder() {
        assert_eq!(normalize("a\u{7}b", ascii_only), "ab");
        // Decorative emoji: not foldable, not wide-range, not alphabetic.
        assert_eq!(normalize("ok \u{1f600} done", ascii_only), "ok done");
    }

    #[test]
    fn cell_units_classify_widths() {
        assert_eq!(cell_units('a'), 1);
        assert_eq!(cell_units('\u{4f60}'), 2); // CJK
        assert_eq!(cell_units('\u{ac00}'), 2); // Hangul
        assert_eq!(cell_units('\u{ff21}'), 2); // fullwidth A
        assert_eq!(cell_units(DIM_ON), 0);
        assert_eq!(cell_units(DIM_OFF), 0);
    }

    #[test]
    fn wrap_counts_wide_chars_as_two_cells() {
        // 4 ASCII + 1 CJK (2 cells) = 6 cells: fits exactly, no straddle.
        assert_eq!(wrap("AAAA\u{4f60}", 6, true), vec!["AAAA\u{4f60}"]);
        // The second CJK glyph would straddle the width: new line first.
        assert_eq!(
            wrap("AAAA\u{4f60}\u{597d}", 6, true),
            vec!["AAAA\u{4f60}", "\u{597d}"]
        );
        // The Silver shape counts every glyph one cell.
        assert_eq!(
            wrap("AAAA\u{4f60}\u{597d}", 6, false),
            vec!["AAAA\u{4f60}\u{597d}"]
        );
    }

    #[test]
    fn wrap_greedy_no_mid_word() {
        let lines = wrap("alpha beta gamma delta", 11, true);
        assert_eq!(lines, vec!["alpha beta", "gamma delta"]);
        // Hard split for overlong words.
        let lines = wrap("abcdefgh", 3, true);
        assert_eq!(lines, vec!["abc", "def", "gh"]);
        // Explicit separators force new lines.
        let lines = wrap("one\u{2588}\u{2588}two", 10, true);
        assert_eq!(lines, vec!["one", "two"]);
    }
}
