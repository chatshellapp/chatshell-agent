//! System-prompt section with the rules for inline SVG figures.
//!
//! When to draw an SVG at all is decided by the visual-output routing rules
//! (`crate::visual_output`); this section only says how to draw one.

const HEAD: &str = "\
SVG figures: a ```svg fenced block renders inline as an image. One visual per \
idea; never draw the same content twice, and never announce or restate the \
figure in prose; add only what it does not show.";

const RULES: &str = "\
- Size with viewBox only (no width/height attributes). 1 unit is about 1 CSS px. \
Keep the viewBox 320 to 640 units wide so it reads on a phone; font-size at least 14.
- Colors ONLY via currentColor and var(--fg), var(--muted), var(--border), \
var(--surface), var(--accent), var(--success), var(--warning), var(--error); \
series var(--c1) to var(--c6). They follow the reader's theme.
- Never paint a full background; the canvas is the reader's theme.
- You MAY animate with SMIL (<animate>, <animateTransform>, <animateMotion>, <set>) \
or CSS @keyframes in a <style> element, for prototypes, transitions, and processes. \
Keep loops calm (at least 1s per cycle) and make the first frame meaningful.
- Self-contained: NEVER <script>, event attributes (onclick and similar), \
<foreignObject>, <image>, external href or url(), or embedded fonts.";

/// How to draw a figure: appended after the routing rules when the client
/// renders SVG figures.
pub fn svg_figure_rules() -> String {
    format!("{HEAD}\n{RULES}")
}
