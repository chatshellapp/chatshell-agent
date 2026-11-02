//! System-prompt fragment inviting inline SVG figures.

const HEAD: &str = "\
Visuals: when a diagram, flow, state machine, UI mockup, layout, or geometry \
says more than prose, you SHOULD draw it in a ```svg fenced block; it renders \
inline as an image. One visual per idea; never draw the same content twice, \
and never announce or restate the figure in prose; add only what it does not show.";

const MERMAID: &str = "\
- Flow, sequence, or dependency graph that Mermaid draws well? Prefer a \
```mermaid block; use SVG for what Mermaid cannot draw.";

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
<foreignObject>, <image>, external href or url(), or embedded fonts.
- Numbers to compare: use a Markdown table instead of an SVG chart.";

/// The fragment appended to the system prompt when the client renders SVG
/// figures. `mermaid`: the client also renders ```mermaid fences.
pub fn svg_figure_prompt(mermaid: bool) -> String {
    let mut prompt = String::with_capacity(HEAD.len() + MERMAID.len() + RULES.len() + 2);
    prompt.push_str(HEAD);
    prompt.push('\n');
    if mermaid {
        prompt.push_str(MERMAID);
        prompt.push('\n');
    }
    prompt.push_str(RULES);
    prompt
}
