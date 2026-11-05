//! Inline SVG figures in assistant Markdown.
//!
//! The model answers with a fenced `svg` block; every client renders it as
//! an isolated image (`<img src="data:image/svg+xml;...">`). This module is
//! the shared, platform-independent part of that pipeline:
//!
//! - [`svg_figure_rules`]: the system-prompt section on how to draw a figure
//!   (when to draw one is routed by `crate::visual_output`);
//! - [`close_partial_svg`]: makes a still-streaming fence parseable;
//! - [`prepare_svg_figure`]: repair → theme → sanitize → size, producing a
//!   self-contained document safe to hand to a web engine.
//!
//! Design: `internal-docs/features/svg-figures.md`. The prompt contract,
//! streaming repair, token substitution and series palette are adapted from
//! oh-my-pi (MIT, Copyright (c) 2025 Mario Zechner):
//! `packages/tui/src/chat/svg-source.ts`, `svg-figure.ts`,
//! `packages/utils/src/color.ts`.

mod color;
mod css;
mod prompt;
mod repair;
mod sanitize;
mod theme;
mod tree;

pub use color::series_colors;
pub use prompt::svg_figure_rules;
pub use repair::close_partial_svg;

use std::fmt;

/// Largest figure source accepted, in bytes.
pub const MAX_SOURCE_BYTES: usize = 256 * 1024;
/// Most elements a figure may hold.
pub const MAX_ELEMENTS: usize = 10_000;
/// Deepest element nesting accepted.
pub const MAX_DEPTH: usize = 64;
/// Most `<use>` references kept; extra ones are dropped.
pub const MAX_USE_REFERENCES: usize = 500;
/// Largest viewBox edge, in user units.
pub const MAX_VIEWBOX_EDGE: f64 = 10_000.0;

/// Theme colors a figure's `var(--name)` tokens resolve to. Values are CSS
/// colors (`#RRGGBB` recommended); `c1`…`c6` are derived from `accent`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SvgPalette {
    pub fg: String,
    pub muted: String,
    pub border: String,
    pub surface: String,
    pub accent: String,
    pub success: String,
    pub warning: String,
    pub error: String,
    pub is_dark: bool,
}

/// How [`prepare_svg_figure`] treats its input.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrepareOptions {
    /// The fence is still streaming: repair the truncated tail first.
    pub streaming: bool,
    /// Remove SMIL and CSS animation (reduced motion, or paused).
    pub strip_animation: bool,
}

/// A sanitized, themed, well-formed figure.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedSvg {
    /// The document to load as `image/svg+xml`.
    pub svg: String,
    /// viewBox width in user units (≈ CSS px).
    pub width: f64,
    /// viewBox height in user units.
    pub height: f64,
    /// The source animates (SMIL or CSS), whether or not it was stripped.
    pub animated: bool,
    /// Elements and attributes the sanitizer dropped.
    pub removed: u32,
}

/// Why a figure could not be prepared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SvgFigureError {
    /// The root `<svg …>` start tag has not fully arrived (streaming).
    Incomplete,
    /// Not well-formed, or no usable size.
    Parse(String),
    /// The root element is not `<svg>`.
    NoRoot,
    /// Over a size limit.
    TooLarge,
    /// Nothing drawable remains.
    Empty,
}

impl fmt::Display for SvgFigureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incomplete => f.write_str("SVG root tag is incomplete"),
            Self::Parse(message) => write!(f, "Invalid SVG: {message}"),
            Self::NoRoot => f.write_str("Root element is not <svg>"),
            Self::TooLarge => f.write_str("SVG exceeds size limits"),
            Self::Empty => f.write_str("SVG has nothing to draw"),
        }
    }
}

impl std::error::Error for SvgFigureError {}

/// Turn a figure source into a document safe to render as an image.
///
/// Pipeline: size check → repair (when `streaming`) → parse → resolve theme
/// tokens → sanitize (allowlist) → strip animation (when asked) → size from
/// the viewBox → serialize.
pub fn prepare_svg_figure(
    source: &str,
    palette: &SvgPalette,
    options: &PrepareOptions,
) -> Result<PreparedSvg, SvgFigureError> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err(SvgFigureError::TooLarge);
    }
    let repaired;
    let document = if options.streaming {
        repaired = close_partial_svg(source).ok_or(SvgFigureError::Incomplete)?;
        repaired.as_str()
    } else {
        source
    };
    let mut root = tree::parse(document)?;
    if !tree::is_svg_name(&root.name) {
        return Err(SvgFigureError::NoRoot);
    }
    let tokens = theme::Tokens::new(palette);
    theme::resolve_tokens(&mut root, &tokens);
    let report = sanitize::sanitize(&mut root, options.strip_animation);
    let (width, height) = theme::finish_root(&mut root, &tokens)?;
    if !root.children.iter().any(tree::Node::is_drawable) {
        return Err(SvgFigureError::Empty);
    }
    Ok(PreparedSvg {
        svg: tree::serialize(&root),
        width,
        height,
        animated: report.animated,
        removed: report.removed,
    })
}

#[cfg(test)]
mod tests;
