//! Visual-output routing: one system-prompt fragment telling the model which
//! format fits which content, built from what the requesting client renders.
//!
//! Every format (Markdown tables, LaTeX math, Mermaid, SVG figures, OpenUI
//! blocks) used to
//! need its own fragment, and each one invited its own format. One routing
//! fragment keeps the instructions consistent and never offers a format the
//! client cannot draw. The fragment is static per capability set, so it does
//! not break prompt caching across turns.
//!
//! The OpenUI section (`openui_prompt.txt`) is written in the desktop app
//! next to the component library it renders (`src/lib/openui/prompt.ts`); a
//! desktop test fails when the two drift apart.
//!
//! Design: `internal-docs/features/visual-output-formats.md`,
//! `internal-docs/features/openui-blocks.md`.

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
    /// ```openui-lang fences render as OpenUI charts and KPI cards.
    pub openui: bool,
}

/// The OpenUI library section, generated from the desktop renderer's library.
const OPENUI_PROMPT: &str = include_str!("openui_prompt.txt");

const INTRO: &str = "\
Visual output: choose the format by what the content is; the first rule that \
matches wins.";

const NUMBERS_OPENUI: &str = "\
- Numbers to compare: a few exact values to read go in a Markdown table; when a \
trend, distribution, share, or set of metrics is worth seeing, use an \
```openui-lang block with a chart or KPI cards, following the OpenUI rules \
below.";

const NUMBERS: &str = "\
- Numbers to compare: put exact values or attributes side by side in a Markdown table; \
when a trend or comparison matters, state the takeaway in one sentence.";

const NO_SVG_CHARTS: &str = " Never draw a chart in SVG.";

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
/// formulas, graphs, pictures, prose) and filtered to the enabled
/// formats; the SVG drawing rules and the OpenUI library section follow when
/// those formats are enabled.
pub fn visual_output_prompt(caps: &VisualCapabilities) -> String {
    let mut numbers = String::from(if caps.openui { NUMBERS_OPENUI } else { NUMBERS });
    if caps.svg {
        numbers.push_str(NO_SVG_CHARTS);
    }

    let mut lines: Vec<&str> = vec![INTRO, &numbers];
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
    if caps.openui {
        prompt.push_str("\n\n");
        prompt.push_str(OPENUI_PROMPT.trim_end());
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
                    for openui in [false, true] {
                        out.push(VisualCapabilities {
                            svg,
                            mermaid,
                            math,
                            openui,
                        });
                    }
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
            assert_eq!(prompt.contains("```openui-lang"), caps.openui, "{caps:?}");
            assert_eq!(prompt.contains("OpenUI"), caps.openui, "{caps:?}");
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
            openui: false,
        });
        let without = visual_output_prompt(&VisualCapabilities {
            svg: true,
            mermaid: false,
            math: true,
            openui: false,
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
            openui: false,
        });
        assert!(prompt.contains("Never draw a chart in SVG"));
    }

    #[test]
    fn rules_are_ordered_numbers_formulas_graphs_pictures_prose() {
        let prompt = visual_output_prompt(&VisualCapabilities {
            svg: true,
            mermaid: true,
            math: true,
            openui: false,
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

    fn openui_caps() -> VisualCapabilities {
        VisualCapabilities {
            svg: true,
            mermaid: true,
            math: true,
            openui: true,
        }
    }

    #[test]
    fn openui_takes_charts_and_metrics() {
        let prompt = visual_output_prompt(&openui_caps());
        assert!(prompt.contains("```openui-lang block with a chart or KPI cards"));
        assert!(prompt.contains("Never draw a chart in SVG"));
    }

    #[test]
    fn openui_blocks_are_static() {
        // Blocks only display: nothing in the prompt invites input, clicks,
        // links, state, or tool calls.
        let prompt = visual_output_prompt(&openui_caps());
        for dynamic in [
            "@ToAssistant",
            "@OpenUrl",
            "ActionExpression",
            "$binding",
            "Form(",
            "Button(",
            "FollowUp",
            "Query(",
            "Mutation(",
            "sources",
        ] {
            assert!(!prompt.contains(dynamic), "{dynamic}");
        }
        assert!(prompt.contains("Blocks only display data"));
    }

    #[test]
    fn openui_library_section_comes_last() {
        let prompt = visual_output_prompt(&openui_caps());
        let svg_rules = prompt.find("SVG figures:").unwrap();
        let library = prompt
            .find("OpenUI blocks: a ```openui-lang fenced block")
            .unwrap();
        assert!(svg_rules < library);
        assert!(prompt.ends_with(OPENUI_PROMPT.trim_end()));
        assert!(prompt.contains("root = Chart("));
        assert!(prompt.contains("root = Stats("));
    }

    #[test]
    fn generated_openui_section_is_sane() {
        // Rewritten by the desktop test; guard against an empty, oversized,
        // or whole-response prompt sneaking in.
        assert!(OPENUI_PROMPT.starts_with("OpenUI blocks:"));
        assert!(OPENUI_PROMPT.len() < 3_000);
        assert!(!OPENUI_PROMPT.contains("ENTIRE response"));
    }
}
