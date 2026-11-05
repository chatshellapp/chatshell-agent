//! Visual-output routing: one system-prompt fragment telling the model which
//! format fits which content, built from what the requesting client renders.
//!
//! Every format (Markdown tables, LaTeX math, Mermaid, SVG figures) used to
//! need its own fragment, and each one invited its own format. One routing
//! fragment keeps the instructions consistent and never offers a format the
//! client cannot draw. The fragment is static per capability set, so it does
//! not break prompt caching across turns.
//!
//! Design: `internal-docs/features/visual-output-formats.md`.

use crate::svg_figure::svg_figure_rules;

/// The visual formats a client renders inline in assistant Markdown.
/// Markdown tables are always available and need no flag.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VisualCapabilities {
    /// ```svg fences render as figures (the user setting is on).
    pub svg: bool,
    /// ```mermaid fences render as diagrams.
    pub mermaid: bool,
    /// `$...$` and `$$...$$` render as math.
    pub math: bool,
}

const INTRO: &str = "\
Visual output: choose the format by what the content is; the first rule that \
matches wins.";

const NUMBERS_WITH_SVG: &str = "\
- Numbers to compare: put exact values or attributes side by side in a Markdown table. \
Never draw a chart in SVG; when a trend or comparison matters, use the table \
and state the takeaway in one sentence.";

const NUMBERS: &str = "\
- Numbers to compare: put exact values or attributes side by side in a Markdown table; \
when a trend or comparison matters, state the takeaway in one sentence.";

const MATH: &str = "\
- Formulas: LaTeX between $...$ inline or $$...$$ for display; it renders. \
Never use \\(...\\) or \\[...\\] delimiters.";

const NO_MATH: &str = "\
- Formulas: this client does not render LaTeX. Write them in plain text with \
Unicode symbols, and put a long derivation in a code block.";

const MERMAID_WITH_SVG: &str = "\
- Boxes joined by arrows (flowchart, sequence, state, ER, class, or Gantt \
diagram): a ```mermaid block; it lays itself out. Use SVG only for what Mermaid \
cannot draw or when it must animate.";

const MERMAID: &str = "\
- Boxes joined by arrows (flowchart, sequence, state, ER, class, or Gantt \
diagram): a ```mermaid block; it lays itself out.";

const SVG_WITH_MERMAID: &str = "\
- Pictures: when a shape, layout, UI mockup, geometry, illustration, or \
animated process says more than prose, you SHOULD draw it in a ```svg block, \
following the SVG rules below.";

const SVG: &str = "\
- Pictures: when a diagram, flow, state machine, shape, layout, UI mockup, \
geometry, illustration, or animated process says more than prose, you SHOULD \
draw it in a ```svg block, following the SVG rules below.";

const PROSE: &str = "\
- Anything else stays prose. Leave out any visual that only restates the prose.";

/// The visual-output fragment for a client with `caps`.
///
/// Rules are ordered like the routing table in the design doc (numbers,
/// formulas, graphs, pictures, prose) and filtered to the enabled formats;
/// the SVG drawing rules follow when SVG is enabled.
pub fn visual_output_prompt(caps: &VisualCapabilities) -> String {
    let mut lines: Vec<&str> = vec![INTRO];
    lines.push(if caps.svg { NUMBERS_WITH_SVG } else { NUMBERS });
    lines.push(if caps.math { MATH } else { NO_MATH });
    if caps.mermaid {
        lines.push(if caps.svg { MERMAID_WITH_SVG } else { MERMAID });
    }
    if caps.svg {
        lines.push(if caps.mermaid { SVG_WITH_MERMAID } else { SVG });
    }
    lines.push(PROSE);

    let mut prompt = lines.join("\n");
    if caps.svg {
        prompt.push_str("\n\n");
        prompt.push_str(&svg_figure_rules());
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_caps() -> Vec<VisualCapabilities> {
        let mut out = Vec::new();
        for svg in [false, true] {
            for mermaid in [false, true] {
                for math in [false, true] {
                    out.push(VisualCapabilities { svg, mermaid, math });
                }
            }
        }
        out
    }

    #[test]
    fn never_offers_a_format_the_client_cannot_render() {
        for caps in all_caps() {
            let prompt = visual_output_prompt(&caps);
            assert_eq!(prompt.contains("```svg"), caps.svg, "{caps:?}");
            assert_eq!(prompt.contains("var(--accent)"), caps.svg, "{caps:?}");
            assert_eq!(prompt.contains("```mermaid"), caps.mermaid, "{caps:?}");
            assert_eq!(
                prompt.to_lowercase().contains("mermaid"),
                caps.mermaid,
                "{caps:?}"
            );
            assert_eq!(prompt.contains("$$...$$"), caps.math, "{caps:?}");
            assert_eq!(
                prompt.contains("does not render LaTeX"),
                !caps.math,
                "{caps:?}"
            );
            assert_eq!(prompt.contains("SVG"), caps.svg, "{caps:?}");
        }
    }

    #[test]
    fn tables_and_prose_rules_are_always_present() {
        for caps in all_caps() {
            let prompt = visual_output_prompt(&caps);
            assert!(prompt.starts_with("Visual output:"), "{caps:?}");
            assert!(prompt.contains("Markdown table"), "{caps:?}");
            assert!(prompt.contains("only restates the prose"), "{caps:?}");
        }
    }

    #[test]
    fn graphs_go_to_svg_only_without_mermaid() {
        let with = visual_output_prompt(&VisualCapabilities {
            svg: true,
            mermaid: true,
            math: true,
        });
        let without = visual_output_prompt(&VisualCapabilities {
            svg: true,
            mermaid: false,
            math: true,
        });
        assert!(!with.contains("state machine"));
        assert!(with.contains("Use SVG only for what Mermaid"));
        assert!(without.contains("flow, state machine"));
    }

    #[test]
    fn charts_are_never_drawn_in_svg() {
        let prompt = visual_output_prompt(&VisualCapabilities {
            svg: true,
            mermaid: false,
            math: false,
        });
        assert!(prompt.contains("Never draw a chart in SVG"));
    }

    #[test]
    fn rules_are_ordered_numbers_formulas_graphs_pictures_prose() {
        let prompt = visual_output_prompt(&VisualCapabilities {
            svg: true,
            mermaid: true,
            math: true,
        });
        let order = [
            "- Numbers to compare:",
            "- Formulas:",
            "- Boxes joined",
            "- Pictures:",
            "- Anything else",
            "SVG figures:",
        ];
        let positions: Vec<usize> = order.iter().map(|m| prompt.find(m).unwrap()).collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "{positions:?}");
    }

    #[test]
    fn prompt_is_ascii_and_deterministic() {
        for caps in all_caps() {
            let prompt = visual_output_prompt(&caps);
            assert!(prompt.is_ascii(), "prompt must stay ASCII: {caps:?}");
            assert_eq!(prompt, visual_output_prompt(&caps));
        }
    }
}
