//! Theme tokens (`var(--name)`) and root sizing.
//!
//! Figures render as isolated images, where the host page's CSS variables do
//! not reach, so every token is replaced with a concrete color here.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use super::color::{is_achromatic, series_colors};
use super::tree::{Element, Node};
use super::{MAX_VIEWBOX_EDGE, SvgFigureError, SvgPalette};

/// `var(--name)` / `var(--name, fallback)`; the fallback may hold one level
/// of parentheses (`rgb(…)`).
static VAR_REFERENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"var\(\s*--([\w-]+)\s*(?:,\s*((?:[^()]|\([^()]*\))*))?\)")
        .expect("var() pattern compiles")
});

const DEFAULT_FONT: &str = "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif";

pub(super) struct Tokens {
    colors: HashMap<&'static str, String>,
    fg: String,
}

impl Tokens {
    pub fn new(palette: &SvgPalette) -> Self {
        let mut colors = HashMap::with_capacity(14);
        colors.insert("fg", palette.fg.clone());
        colors.insert("muted", palette.muted.clone());
        colors.insert("border", palette.border.clone());
        colors.insert("surface", palette.surface.clone());
        colors.insert("success", palette.success.clone());
        colors.insert("warning", palette.warning.clone());
        colors.insert("error", palette.error.clone());
        const SERIES: [&str; 6] = ["c1", "c2", "c3", "c4", "c5", "c6"];
        let series = series_colors(&palette.accent, palette.is_dark);
        // A gray accent (monochrome themes) would read as plain text; the
        // first series color stands in so highlights stay visible.
        let accent = if is_achromatic(&palette.accent) {
            series[0].clone()
        } else {
            palette.accent.clone()
        };
        colors.insert("accent", accent);
        for (name, color) in SERIES.iter().zip(series) {
            colors.insert(name, color);
        }
        Self {
            colors,
            fg: palette.fg.clone(),
        }
    }

    fn resolve(&self, text: &str) -> String {
        if !text.contains("var(") {
            return text.to_string();
        }
        VAR_REFERENCE
            .replace_all(text, |captures: &regex::Captures<'_>| {
                let name = &captures[1];
                if let Some(color) = self.colors.get(name) {
                    return color.clone();
                }
                match captures.get(2).map(|m| m.as_str().trim()) {
                    Some(fallback) if !fallback.is_empty() => fallback.to_string(),
                    _ => self.fg.clone(),
                }
            })
            .into_owned()
    }
}

/// Replace every `var(--name)` in attribute values and `<style>` text.
pub(super) fn resolve_tokens(element: &mut Element, tokens: &Tokens) {
    for (_, value) in &mut element.attributes {
        if value.contains("var(") {
            *value = tokens.resolve(value);
        }
    }
    let is_style = super::tree::local_name(&element.name) == "style";
    for child in &mut element.children {
        match child {
            Node::Element(child) => resolve_tokens(child, tokens),
            Node::Text(text) if is_style && text.contains("var(") => {
                *text = tokens.resolve(text);
            }
            Node::Text(_) => {}
        }
    }
}

/// Settle the root: size from the viewBox (or numeric width/height when it
/// is missing), drop width/height so the renderer scales it, and default the
/// text color and font so `currentColor` and text follow the theme.
pub(super) fn finish_root(
    root: &mut Element,
    tokens: &Tokens,
) -> Result<(f64, f64), SvgFigureError> {
    root.name = "svg".to_string();
    let width_attribute = root.remove_attribute("width");
    let height_attribute = root.remove_attribute("height");
    let (width, height) = match root.attribute("viewBox").map(parse_view_box) {
        Some(Some((_, _, width, height))) => (width, height),
        Some(None) => return Err(SvgFigureError::Parse("invalid viewBox".into())),
        None => {
            let width = width_attribute.as_deref().and_then(parse_length);
            let height = height_attribute.as_deref().and_then(parse_length);
            match (width, height) {
                (Some(width), Some(height)) => {
                    root.set_attribute("viewBox", format!("0 0 {width} {height}"));
                    (width, height)
                }
                _ => return Err(SvgFigureError::Parse("missing viewBox".into())),
            }
        }
    };
    let valid = |edge: f64| edge.is_finite() && (1.0..=MAX_VIEWBOX_EDGE).contains(&edge);
    if !valid(width) || !valid(height) {
        return Err(SvgFigureError::Parse("viewBox size out of range".into()));
    }
    if root.attribute("color").is_none() {
        root.set_attribute("color", tokens.fg.clone());
    }
    if root.attribute("font-family").is_none() {
        root.set_attribute("font-family", DEFAULT_FONT.to_string());
    }
    // Namespace first, as serializers conventionally emit it.
    root.attributes.insert(
        0,
        (
            "xmlns".to_string(),
            "http://www.w3.org/2000/svg".to_string(),
        ),
    );
    Ok((width, height))
}

fn parse_view_box(value: &str) -> Option<(f64, f64, f64, f64)> {
    let numbers: Vec<f64> = value
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|part| !part.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    match numbers.as_slice() {
        [x, y, width, height] => Some((*x, *y, *width, *height)),
        _ => None,
    }
}

/// A plain or `px` length; percentages and other units have no fixed size.
fn parse_length(value: &str) -> Option<f64> {
    let value = value.trim();
    let number = value.strip_suffix("px").unwrap_or(value).trim();
    number.parse().ok()
}
