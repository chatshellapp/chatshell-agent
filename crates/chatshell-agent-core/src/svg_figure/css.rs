//! CSS allowlisting for figure `style` attributes and `<style>` sheets.
//!
//! Conservative by construction: a declaration, rule or at-rule that might
//! load something, run something, or hide either behind escapes is dropped.
//! Kept: qualified rules, `@keyframes`, `@media`, `@supports`.

/// What sanitizing CSS removed and found.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct CssReport {
    pub removed: u32,
    pub animated: bool,
}

/// Sanitize a declaration list (`style="…"` or a rule body).
pub(super) fn sanitize_declarations(
    css: &str,
    strip_animation: bool,
    report: &mut CssReport,
) -> String {
    let css = strip_comments(css);
    let mut out = String::with_capacity(css.len());
    for declaration in split_top_level(&css, b';') {
        let declaration = declaration.trim();
        if declaration.is_empty() {
            continue;
        }
        let Some((property, value)) = declaration.split_once(':') else {
            report.removed += 1;
            continue;
        };
        let property = property.trim();
        let value = value.trim();
        let lower = property.to_ascii_lowercase();
        if !is_safe_property(&lower) || !is_safe_value(value) {
            report.removed += 1;
            continue;
        }
        if is_animation_property(&lower) {
            report.animated = true;
            if strip_animation {
                continue;
            }
        }
        if !out.is_empty() {
            out.push(';');
        }
        out.push_str(property);
        out.push(':');
        out.push_str(value);
    }
    out
}

/// Sanitize a full stylesheet (`<style>` text).
pub(super) fn sanitize_stylesheet(
    css: &str,
    strip_animation: bool,
    report: &mut CssReport,
) -> String {
    let css = strip_comments(css);
    let mut out = String::with_capacity(css.len());
    sheet_into(&css, strip_animation, report, &mut out, 0);
    out
}

const MAX_NESTING: usize = 8;

fn sheet_into(
    css: &str,
    strip_animation: bool,
    report: &mut CssReport,
    out: &mut String,
    depth: usize,
) {
    let mut rest = css;
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            return;
        }
        let Some(boundary) = find_top_level(rest, b"{;") else {
            // Trailing garbage without a block.
            report.removed += 1;
            return;
        };
        let prelude = rest[..boundary].trim();
        if rest.as_bytes()[boundary] == b';' {
            // Statement at-rules (`@import`, `@charset`, `@namespace`) and stray text.
            report.removed += 1;
            rest = &rest[boundary + 1..];
            continue;
        }
        let Some(close) = matching_brace(rest, boundary) else {
            report.removed += 1;
            return;
        };
        let body = &rest[boundary + 1..close];
        rest = &rest[close + 1..];

        if !is_safe_value(prelude) {
            report.removed += 1;
            continue;
        }
        if let Some(at_rule) = prelude.strip_prefix('@') {
            let name_end = at_rule
                .find(|c: char| c.is_whitespace() || c == '(')
                .unwrap_or(at_rule.len());
            let name = at_rule[..name_end].to_ascii_lowercase();
            match name.as_str() {
                "keyframes" | "-webkit-keyframes" => {
                    report.animated = true;
                    if strip_animation {
                        continue;
                    }
                    out.push_str(prelude);
                    out.push('{');
                    keyframes_into(body, strip_animation, report, out);
                    out.push('}');
                }
                "media" | "supports" if depth < MAX_NESTING => {
                    out.push_str(prelude);
                    out.push('{');
                    sheet_into(body, strip_animation, report, out, depth + 1);
                    out.push('}');
                }
                _ => report.removed += 1,
            }
            continue;
        }
        let declarations = sanitize_declarations(body, strip_animation, report);
        out.push_str(prelude);
        out.push('{');
        out.push_str(&declarations);
        out.push('}');
    }
}

/// `@keyframes` body: `<selector> { declarations }` blocks.
fn keyframes_into(css: &str, strip_animation: bool, report: &mut CssReport, out: &mut String) {
    let mut rest = css;
    loop {
        rest = rest.trim_start();
        let Some(open) = find_top_level(rest, b"{") else {
            return;
        };
        let Some(close) = matching_brace(rest, open) else {
            report.removed += 1;
            return;
        };
        let selector = rest[..open].trim();
        let body = &rest[open + 1..close];
        rest = &rest[close + 1..];
        if !is_safe_value(selector) {
            report.removed += 1;
            continue;
        }
        out.push_str(selector);
        out.push('{');
        out.push_str(&sanitize_declarations(body, strip_animation, report));
        out.push('}');
    }
}

/// Whether a value (or prelude) loads or runs nothing. `url(#id)` is the
/// only `url()` allowed; escapes are refused outright since they can spell
/// anything.
pub(super) fn is_safe_value(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    if lower.contains('\\') {
        return false;
    }
    const BLOCKED: [&str; 9] = [
        "expression(",
        "image(",
        "image-set(",
        "cross-fade(",
        "element(",
        "src(",
        "javascript:",
        "vbscript:",
        "@import",
    ];
    if BLOCKED.iter().any(|pattern| lower.contains(pattern)) {
        return false;
    }
    let mut search = lower.as_str();
    while let Some(at) = search.find("url(") {
        let after = search[at + 4..].trim_start_matches([' ', '\t', '\n', '"', '\'']);
        if !after.starts_with('#') {
            return false;
        }
        search = &search[at + 4..];
    }
    true
}

fn is_safe_property(property: &str) -> bool {
    !property.is_empty()
        && property
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        && !matches!(property, "behavior" | "-moz-binding" | "cursor")
}

pub(super) fn is_animation_property(property: &str) -> bool {
    let property = property.strip_prefix("-webkit-").unwrap_or(property);
    property.starts_with("animation") || property.starts_with("transition")
}

fn strip_comments(css: &str) -> String {
    let mut out = String::with_capacity(css.len());
    let mut rest = css;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        match rest[start + 2..].find("*/") {
            Some(end) => rest = &rest[start + 2 + end + 2..],
            None => return out,
        }
        out.push(' ');
    }
    out.push_str(rest);
    out
}

/// Split on `separator` outside quotes, parentheses and braces.
fn split_top_level(css: &str, separator: u8) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut scanner = Scanner::default();
    for (at, &byte) in css.as_bytes().iter().enumerate() {
        if scanner.top_level(byte) && byte == separator {
            parts.push(&css[start..at]);
            start = at + 1;
        }
    }
    parts.push(&css[start..]);
    parts
}

/// First index of any `targets` byte outside quotes and parentheses.
fn find_top_level(css: &str, targets: &[u8]) -> Option<usize> {
    let mut scanner = Scanner::default();
    css.as_bytes()
        .iter()
        .enumerate()
        .find(|(_, byte)| scanner.top_level(**byte) && targets.contains(byte))
        .map(|(at, _)| at)
}

/// Index of the `}` matching the `{` at `open`.
fn matching_brace(css: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote: Option<u8> = None;
    for (at, &byte) in css.as_bytes().iter().enumerate().skip(open) {
        match quote {
            Some(q) if byte == q => quote = None,
            Some(_) => {}
            None => match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(at);
                    }
                }
                _ => {}
            },
        }
    }
    None
}

/// Tracks quotes, parentheses and braces while scanning CSS byte by byte.
#[derive(Default)]
struct Scanner {
    quote: Option<u8>,
    parens: usize,
    braces: usize,
}

impl Scanner {
    /// Feed `byte`; returns whether it sits at the top level (outside any
    /// quote, parenthesis or brace). Delimiters themselves count as top level
    /// when they open from or close to it.
    fn top_level(&mut self, byte: u8) -> bool {
        if let Some(q) = self.quote {
            if byte == q {
                self.quote = None;
            }
            return false;
        }
        let was_top = self.parens == 0 && self.braces == 0;
        match byte {
            b'"' | b'\'' => {
                self.quote = Some(byte);
                false
            }
            b'(' => {
                self.parens += 1;
                false
            }
            b')' => {
                self.parens = self.parens.saturating_sub(1);
                false
            }
            b'{' => {
                self.braces += 1;
                was_top
            }
            b'}' => {
                self.braces = self.braces.saturating_sub(1);
                false
            }
            _ => was_top,
        }
    }
}
