use super::*;

fn palette(is_dark: bool) -> SvgPalette {
    SvgPalette {
        fg: "#111111".into(),
        muted: "#777777".into(),
        border: "#dddddd".into(),
        surface: "#f4f4f4".into(),
        accent: "#2563eb".into(),
        success: "#16a34a".into(),
        warning: "#d97706".into(),
        error: "#dc2626".into(),
        is_dark,
    }
}

fn prepare(source: &str) -> Result<PreparedSvg, SvgFigureError> {
    prepare_svg_figure(source, &palette(false), &PrepareOptions::default())
}

fn prepared(source: &str) -> String {
    prepare(source).expect("figure prepares").svg
}

const FLOW: &str = r##"<svg viewBox="0 0 400 120" xmlns="http://www.w3.org/2000/svg">
  <title>Flow</title>
  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--muted)"/></marker></defs>
  <rect x="10" y="40" width="120" height="40" rx="8" fill="var(--surface)" stroke="var(--border)"/>
  <text x="70" y="65" text-anchor="middle" font-size="14" fill="currentColor">Start &amp; go</text>
  <line x1="130" y1="60" x2="270" y2="60" stroke="var(--accent)" marker-end="url(#arrow)"/>
  <circle cx="300" cy="60" r="20" fill="var(--c2)"><animate attributeName="r" values="18;22;18" dur="2s" repeatCount="indefinite"/></circle>
</svg>"##;

// ---------------------------------------------------------------------------
// Repair
// ---------------------------------------------------------------------------

#[test]
fn repair_returns_none_before_root_tag_completes() {
    assert_eq!(close_partial_svg(""), None);
    assert_eq!(close_partial_svg("<svg viewBox=\"0 0 1"), None);
    assert_eq!(close_partial_svg("<!-- hi --><sv"), None);
}

#[test]
fn repair_passes_complete_documents_through() {
    assert_eq!(close_partial_svg(FLOW).as_deref(), Some(FLOW));
}

#[test]
fn repair_closes_open_elements_and_cuts_partial_constructs() {
    let partial = "<svg viewBox=\"0 0 10 10\"><g><rect x=\"1\"/><text>a &am";
    assert_eq!(
        close_partial_svg(partial).as_deref(),
        Some("<svg viewBox=\"0 0 10 10\"><g><rect x=\"1\"/><text>a </text></g></svg>")
    );
    let in_tag = "<svg viewBox=\"0 0 10 10\"><rect x=\"1\" fill=\"var(--c";
    assert_eq!(
        close_partial_svg(in_tag).as_deref(),
        Some("<svg viewBox=\"0 0 10 10\"></svg>")
    );
    let in_comment = "<svg viewBox=\"0 0 10 10\"><!-- still";
    assert_eq!(
        close_partial_svg(in_comment).as_deref(),
        Some("<svg viewBox=\"0 0 10 10\"></svg>")
    );
}

#[test]
fn every_prefix_repairs_to_incomplete_or_parseable() {
    for end in 0..=FLOW.len() {
        if !FLOW.is_char_boundary(end) {
            continue;
        }
        let options = PrepareOptions {
            streaming: true,
            strip_animation: false,
        };
        match prepare_svg_figure(&FLOW[..end], &palette(false), &options) {
            Ok(_) | Err(SvgFigureError::Incomplete) | Err(SvgFigureError::Empty) => {}
            Err(other) => panic!("prefix {end} failed: {other:?}\n{}", &FLOW[..end]),
        }
    }
}

// ---------------------------------------------------------------------------
// Theming and sizing
// ---------------------------------------------------------------------------

#[test]
fn resolves_tokens_and_sizes_from_view_box() {
    let figure = prepare(FLOW).unwrap();
    assert_eq!((figure.width, figure.height), (400.0, 120.0));
    assert!(figure.animated);
    assert!(!figure.svg.contains("var("));
    assert!(figure.svg.contains("fill=\"#f4f4f4\""));
    assert!(figure.svg.contains("stroke=\"#2563eb\""));
    assert!(figure.svg.contains("color=\"#111111\""));
    assert!(
        figure
            .svg
            .starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\"")
    );
    assert!(figure.svg.contains("marker-end=\"url(#arrow)\""));
    assert!(figure.svg.contains("Start &amp; go"));
}

#[test]
fn unknown_tokens_take_fallback_then_fg() {
    let svg = prepared(
        r#"<svg viewBox="0 0 10 10"><rect fill="var(--nope, rgb(1, 2, 3))" stroke="var(--nope)" width="1" height="1"/></svg>"#,
    );
    assert!(svg.contains("fill=\"rgb(1, 2, 3)\""));
    assert!(svg.contains("stroke=\"#111111\""));
}

#[test]
fn tokens_resolve_inside_style_sheets() {
    let svg = prepared(
        r#"<svg viewBox="0 0 10 10"><style>.a { fill: var(--error) }</style><rect class="a" width="1" height="1"/></svg>"#,
    );
    assert!(svg.contains(".a{fill:#dc2626}"), "{svg}");
}

#[test]
fn width_height_replace_missing_view_box() {
    let figure =
        prepare(r#"<svg width="200px" height="100"><rect width="1" height="1"/></svg>"#).unwrap();
    assert_eq!((figure.width, figure.height), (200.0, 100.0));
    assert!(figure.svg.contains("viewBox=\"0 0 200 100\""));
    assert!(!figure.svg.contains(" width=\"200px\""));
}

#[test]
fn missing_or_bad_size_is_a_parse_error() {
    assert!(matches!(
        prepare("<svg><rect/></svg>"),
        Err(SvgFigureError::Parse(_))
    ));
    assert!(matches!(
        prepare(r#"<svg viewBox="0 0 0 10"><rect/></svg>"#),
        Err(SvgFigureError::Parse(_))
    ));
    assert!(matches!(
        prepare(r#"<svg viewBox="0 0 99999 10"><rect/></svg>"#),
        Err(SvgFigureError::Parse(_))
    ));
}

#[test]
fn non_svg_root_and_empty_figures_fail() {
    assert_eq!(prepare("<div>hi</div>"), Err(SvgFigureError::NoRoot));
    assert_eq!(
        prepare(r#"<svg viewBox="0 0 10 10"><title>t</title></svg>"#),
        Err(SvgFigureError::Empty)
    );
    assert_eq!(
        prepare(r#"<svg viewBox="0 0 10 10"><script>alert(1)</script></svg>"#),
        Err(SvgFigureError::Empty)
    );
}

#[test]
fn series_colors_are_distinct_and_deterministic() {
    for dark in [false, true] {
        let colors = series_colors("#2563eb", dark);
        let again = series_colors("#2563eb", dark);
        assert_eq!(colors, again);
        for (at, color) in colors.iter().enumerate() {
            assert_eq!(color.len(), 7, "{color}");
            assert!(colors[at + 1..].iter().all(|other| other != color));
        }
    }
    // A gray (or unparseable) accent leads with the default blue family.
    assert_eq!(
        series_colors("#888888", false),
        series_colors("not-a-color", false)
    );
    assert_ne!(
        series_colors("#888888", false),
        series_colors("#e11d48", false)
    );
}

#[test]
fn gray_accent_falls_back_to_first_series_color() {
    let mut gray = palette(false);
    gray.accent = "#222222".into();
    let svg = prepare_svg_figure(
        r#"<svg viewBox="0 0 10 10"><rect width="1" height="1" fill="var(--accent)"/></svg>"#,
        &gray,
        &PrepareOptions::default(),
    )
    .unwrap()
    .svg;
    let c1 = &series_colors("#222222", false)[0];
    assert!(svg.contains(&format!("fill=\"{c1}\"")), "{svg}");
}

#[test]
fn light_and_dark_series_differ() {
    assert_ne!(
        series_colors("#2563eb", false),
        series_colors("#2563eb", true)
    );
}

// ---------------------------------------------------------------------------
// Sanitizer
// ---------------------------------------------------------------------------

/// Hostile constructs that must not survive in any form.
const ATTACKS: &[&str] = &[
    r#"<svg viewBox="0 0 10 10" onload="alert(1)"><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><script>alert(1)</script><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><rect width="1" height="1" onclick="alert(1)" onmouseover="x()"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><foreignObject><div xmlns="http://www.w3.org/1999/xhtml"><script>alert(1)</script></div></foreignObject><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10" xmlns:xlink="http://www.w3.org/1999/xlink"><a xlink:href="javascript:alert(1)"><rect width="1" height="1"/></a></svg>"#,
    r#"<svg viewBox="0 0 10 10"><use href="https://evil.example/x.svg#a"/><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><use href="data:image/svg+xml;base64,PHN2Zz4="/><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><image href="https://evil.example/p.png"/><rect width="1" height="1"/></svg>"#,
    r##"<svg viewBox="0 0 10 10"><a href="#x"><set attributeName="href" to="javascript:alert(1)"/></a><rect width="1" height="1"/></svg>"##,
    r#"<svg viewBox="0 0 10 10"><rect width="1" height="1"><animate attributeName="onclick" to="alert(1)"/></rect></svg>"#,
    r#"<svg viewBox="0 0 10 10"><rect width="1" height="1"><set attributeName="xlink:href" to="javascript:alert(1)"/></rect></svg>"#,
    r#"<svg viewBox="0 0 10 10"><style>@import url(https://evil.example/a.css); rect { fill: red }</style><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><style>rect { background: url(https://evil.example/beacon) }</style><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><style>@font-face { font-family: x; src: url(https://evil.example/f.woff) }</style><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><rect width="1" height="1" style="fill: url('https://evil.example/p')"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><rect width="1" height="1" style="fill: u\72l(https://evil.example/p)"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><rect width="1" height="1" fill="url(https://evil.example/p#g)"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><rect width="1" height="1" style="behavior: url(x.htc); -moz-binding: url(x)"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><iframe src="https://evil.example"/><embed src="x"/><object data="x"/><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><svg:script xmlns:svg="http://www.w3.org/2000/svg">alert(1)</svg:script><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><xhtml:script xmlns:xhtml="http://www.w3.org/1999/xhtml">alert(1)</xhtml:script><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><rect width="1" height="1" fill="var(--nope, url(https://evil.example/x))"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><style>rect { fill: var(--nope, url(https://evil.example/x)) }</style><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><rect width="1" height="1"><animate attributeName="fill" to="url(https://evil.example/x)"/></rect></svg>"#,
    r#"<svg viewBox="0 0 10 10" xml:base="https://evil.example/"><rect width="1" height="1"/></svg>"#,
    r#"<svg viewBox="0 0 10 10"><text><![CDATA[</text><script>alert(1)</script>]]></text><rect width="1" height="1"/></svg>"#,
];

fn assert_harmless(svg: &str) {
    let lower = svg.to_ascii_lowercase();
    for needle in [
        "<script",
        "onload",
        "onclick",
        "onmouseover",
        "javascript:",
        "foreignobject",
        "evil.example",
        "@import",
        "@font-face",
        "<iframe",
        "<embed",
        "<object",
        "<image",
        "xml:base",
        "behavior",
        "binding",
        "data:image",
    ] {
        assert!(!lower.contains(needle), "`{needle}` survived in:\n{svg}");
    }
}

#[test]
fn attack_corpus_is_neutralized() {
    for (at, attack) in ATTACKS.iter().enumerate() {
        match prepare(attack) {
            Ok(figure) => {
                assert_harmless(&figure.svg);
                assert!(
                    figure.removed > 0 || figure.svg.contains("&lt;"),
                    "attack {at} removed nothing: {}",
                    figure.svg
                );
            }
            Err(error) => assert!(
                matches!(error, SvgFigureError::Empty | SvgFigureError::Parse(_)),
                "attack {at}: {error:?}"
            ),
        }
    }
}

#[test]
fn local_references_survive() {
    let svg = prepared(
        r##"<svg viewBox="0 0 10 10" xmlns:xlink="http://www.w3.org/1999/xlink"><defs><linearGradient id="g"><stop offset="0" stop-color="var(--c1)"/></linearGradient><circle id="dot" r="1"/></defs><use xlink:href="#dot" fill="url(#g)"/></svg>"##,
    );
    assert!(
        svg.contains("<use href=\"#dot\" fill=\"url(#g)\"/>"),
        "{svg}"
    );
    assert!(!svg.contains("xlink"));
}

#[test]
fn anchors_are_unwrapped_not_dropped() {
    let svg = prepared(
        r##"<svg viewBox="0 0 10 10"><a href="#x"><rect width="1" height="1"/></a></svg>"##,
    );
    assert!(svg.contains("<rect width=\"1\" height=\"1\"/>"));
    assert!(!svg.contains("<a"));
}

#[test]
fn custom_entities_stay_literal_text() {
    let svg = prepared(
        r#"<!DOCTYPE svg [<!ENTITY x "<script>alert(1)</script>">]><svg viewBox="0 0 10 10"><text>&x;</text></svg>"#,
    );
    assert!(svg.contains("<text>&amp;x;</text>"), "{svg}");
}

#[test]
fn use_references_are_capped() {
    let mut source =
        String::from(r#"<svg viewBox="0 0 10 10"><defs><circle id="c" r="1"/></defs>"#);
    for _ in 0..(MAX_USE_REFERENCES + 20) {
        source.push_str(r##"<use href="#c"/>"##);
    }
    source.push_str("</svg>");
    let figure = prepare(&source).unwrap();
    assert_eq!(figure.svg.matches("<use").count(), MAX_USE_REFERENCES);
}

#[test]
fn limits_are_enforced() {
    let huge = format!(
        "<svg viewBox=\"0 0 10 10\"><desc>{}</desc></svg>",
        "a".repeat(MAX_SOURCE_BYTES)
    );
    assert_eq!(prepare(&huge), Err(SvgFigureError::TooLarge));

    let deep = format!(
        "<svg viewBox=\"0 0 10 10\">{}<rect/>{}</svg>",
        "<g>".repeat(MAX_DEPTH),
        "</g>".repeat(MAX_DEPTH)
    );
    assert_eq!(prepare(&deep), Err(SvgFigureError::TooLarge));

    let wide = format!(
        "<svg viewBox=\"0 0 10 10\">{}</svg>",
        "<rect/>".repeat(MAX_ELEMENTS + 1)
    );
    assert_eq!(prepare(&wide), Err(SvgFigureError::TooLarge));
}

// ---------------------------------------------------------------------------
// Animation
// ---------------------------------------------------------------------------

const ANIMATED_CSS: &str = r#"<svg viewBox="0 0 100 100">
<style>
  @keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }
  .s { animation: spin 2s linear infinite; transform-origin: 50px 50px; fill: var(--accent) }
  @media (prefers-reduced-motion: reduce) { .s { animation: none } }
</style>
<rect class="s" x="25" y="25" width="50" height="50" style="transition: fill 1s; opacity: .9"/>
<circle cx="10" cy="10" r="5"><animateTransform attributeName="transform" type="translate" values="0 0;5 0;0 0" dur="1s" repeatCount="indefinite"/></circle>
</svg>"#;

#[test]
fn animations_are_kept_by_default() {
    let figure = prepare(ANIMATED_CSS).unwrap();
    assert!(figure.animated);
    assert!(
        figure
            .svg
            .contains("@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}"),
        "{}",
        figure.svg
    );
    assert!(figure.svg.contains("animation:spin 2s linear infinite"));
    assert!(
        figure
            .svg
            .contains("@media (prefers-reduced-motion: reduce){.s{animation:none}}")
    );
    assert!(figure.svg.contains("<animateTransform"));
    assert!(
        figure
            .svg
            .contains("style=\"transition:fill 1s;opacity:.9\"")
    );
}

#[test]
fn strip_animation_removes_smil_and_css_motion() {
    let options = PrepareOptions {
        streaming: false,
        strip_animation: true,
    };
    let figure = prepare_svg_figure(ANIMATED_CSS, &palette(true), &options).unwrap();
    assert!(figure.animated, "the source still reports that it animates");
    for needle in ["@keyframes", "animation", "transition", "<animate"] {
        assert!(
            !figure.svg.contains(needle),
            "`{needle}` survived:\n{}",
            figure.svg
        );
    }
    assert!(figure.svg.contains("style=\"opacity:.9\""));
    assert!(figure.svg.contains("transform-origin:50px 50px"));
}

#[test]
fn static_figures_report_not_animated() {
    let figure = prepare(r#"<svg viewBox="0 0 10 10"><rect width="1" height="1"/></svg>"#).unwrap();
    assert!(!figure.animated);
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

#[test]
fn prompt_mentions_mermaid_only_when_rendered() {
    let with = svg_figure_prompt(true);
    let without = svg_figure_prompt(false);
    assert!(with.contains("```mermaid"));
    assert!(!without.contains("mermaid"));
    for prompt in [&with, &without] {
        assert!(prompt.contains("```svg"));
        assert!(prompt.contains("var(--accent)"));
        assert!(prompt.contains("@keyframes"));
        assert!(prompt.is_ascii(), "prompt must stay ASCII");
    }
}
