//! Allowlist sanitizer: rebuilds the tree from known-safe SVG only.
//!
//! Anything not listed is dropped (an element with its subtree, or the
//! attribute). Renderers additionally load the result as an image, where
//! scripts never run and nothing external loads; this pass makes the output
//! safe even without that barrier.

use super::MAX_USE_REFERENCES;
use super::css::{self, CssReport};
use super::tree::{Element, Node, local_name, prefix};

/// What sanitizing removed and found.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct Report {
    pub removed: u32,
    pub animated: bool,
}

const ELEMENTS: &[&str] = &[
    "svg",
    "g",
    "defs",
    "symbol",
    "use",
    "title",
    "desc",
    "switch",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "text",
    "tspan",
    "textPath",
    "linearGradient",
    "radialGradient",
    "stop",
    "pattern",
    "clipPath",
    "mask",
    "marker",
    "filter",
    "feBlend",
    "feColorMatrix",
    "feComponentTransfer",
    "feComposite",
    "feFlood",
    "feGaussianBlur",
    "feMerge",
    "feMergeNode",
    "feMorphology",
    "feOffset",
    "feDropShadow",
    "feFuncR",
    "feFuncG",
    "feFuncB",
    "feFuncA",
    "feTurbulence",
    "feDisplacementMap",
    "feTile",
    "animate",
    "animateTransform",
    "animateMotion",
    "mpath",
    "set",
    "style",
];

/// Elements whose children are kept while the element itself is dropped.
const UNWRAP: &[&str] = &["a"];

const ANIMATION_ELEMENTS: &[&str] = &["animate", "animateTransform", "animateMotion", "set"];

/// Elements that may carry a local-fragment `href`.
const HREF_ELEMENTS: &[&str] = &[
    "use",
    "textPath",
    "mpath",
    "linearGradient",
    "radialGradient",
    "pattern",
    "filter",
    "animate",
    "animateTransform",
    "animateMotion",
    "set",
];

const ATTRIBUTES: &[&str] = &[
    // Core and accessibility
    "id",
    "class",
    "style",
    "lang",
    "role",
    "aria-label",
    "aria-labelledby",
    "aria-describedby",
    "aria-hidden",
    "xml:space",
    "xml:lang",
    // Presentation
    "alignment-baseline",
    "baseline-shift",
    "clip-path",
    "clip-rule",
    "color",
    "color-interpolation",
    "color-interpolation-filters",
    "direction",
    "display",
    "dominant-baseline",
    "fill",
    "fill-opacity",
    "fill-rule",
    "filter",
    "flood-color",
    "flood-opacity",
    "font-family",
    "font-size",
    "font-size-adjust",
    "font-stretch",
    "font-style",
    "font-variant",
    "font-weight",
    "image-rendering",
    "letter-spacing",
    "lighting-color",
    "marker-end",
    "marker-mid",
    "marker-start",
    "mask",
    "mask-type",
    "opacity",
    "overflow",
    "paint-order",
    "pointer-events",
    "shape-rendering",
    "stop-color",
    "stop-opacity",
    "stroke",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "stroke-opacity",
    "stroke-width",
    "text-anchor",
    "text-decoration",
    "text-rendering",
    "transform",
    "transform-origin",
    "unicode-bidi",
    "vector-effect",
    "visibility",
    "word-spacing",
    "writing-mode",
    // Geometry and element-specific
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "fx",
    "fy",
    "fr",
    "width",
    "height",
    "d",
    "points",
    "pathLength",
    "viewBox",
    "preserveAspectRatio",
    "dx",
    "dy",
    "rotate",
    "textLength",
    "lengthAdjust",
    "startOffset",
    "method",
    "spacing",
    "side",
    // Paint servers, clipping, markers
    "gradientUnits",
    "gradientTransform",
    "spreadMethod",
    "offset",
    "patternUnits",
    "patternContentUnits",
    "patternTransform",
    "clipPathUnits",
    "maskUnits",
    "maskContentUnits",
    "markerWidth",
    "markerHeight",
    "markerUnits",
    "refX",
    "refY",
    "orient",
    // Filters
    "filterUnits",
    "primitiveUnits",
    "in",
    "in2",
    "result",
    "stdDeviation",
    "mode",
    "operator",
    "k1",
    "k2",
    "k3",
    "k4",
    "values",
    "type",
    "tableValues",
    "slope",
    "intercept",
    "amplitude",
    "exponent",
    "radius",
    "edgeMode",
    "baseFrequency",
    "numOctaves",
    "seed",
    "stitchTiles",
    "scale",
    "xChannelSelector",
    "yChannelSelector",
    // Animation
    "attributeName",
    "attributeType",
    "begin",
    "dur",
    "end",
    "min",
    "max",
    "restart",
    "repeatCount",
    "repeatDur",
    "from",
    "to",
    "by",
    "keyTimes",
    "keySplines",
    "calcMode",
    "additive",
    "accumulate",
    "keyPoints",
    "path",
    "origin",
    // References (checked separately)
    "href",
];

/// Attributes an animation may never target.
fn forbidden_animation_target(name: &str) -> bool {
    let local = local_name(name).to_ascii_lowercase();
    local == "href" || local.starts_with("on")
}

pub(super) fn sanitize(root: &mut Element, strip_animation: bool) -> Report {
    let mut state = State {
        report: Report::default(),
        strip_animation,
        uses: 0,
    };
    state.element(root, true);
    state.report
}

struct State {
    report: Report,
    strip_animation: bool,
    uses: usize,
}

impl State {
    fn element(&mut self, element: &mut Element, is_root: bool) {
        // Normalize `svg:rect` to `rect`; the output declares the SVG
        // namespace as default.
        if prefix(&element.name).is_some() {
            element.name = local_name(&element.name).to_string();
        }
        self.attributes(element, is_root);
        if element.name == "style" {
            let mut report = CssReport::default();
            let sheet =
                css::sanitize_stylesheet(&element.text(), self.strip_animation, &mut report);
            self.merge(report);
            element.attributes.clear();
            element.children = if sheet.is_empty() {
                Vec::new()
            } else {
                vec![Node::Text(sheet)]
            };
            return;
        }
        let children = std::mem::take(&mut element.children);
        element.children = self.children(children);
    }

    fn children(&mut self, children: Vec<Node>) -> Vec<Node> {
        let mut kept = Vec::with_capacity(children.len());
        for child in children {
            match child {
                Node::Text(text) => kept.push(Node::Text(text)),
                Node::Element(mut child) => {
                    let name = if prefix(&child.name).is_some_and(|p| p != "svg") {
                        // Foreign namespace (xhtml:div, sodipodi:…): never SVG.
                        self.report.removed += 1;
                        continue;
                    } else {
                        local_name(&child.name).to_string()
                    };
                    if UNWRAP.contains(&name.as_str()) {
                        self.report.removed += 1;
                        let grandchildren = std::mem::take(&mut child.children);
                        kept.extend(self.children(grandchildren));
                        continue;
                    }
                    if !ELEMENTS.contains(&name.as_str()) {
                        self.report.removed += 1;
                        continue;
                    }
                    if ANIMATION_ELEMENTS.contains(&name.as_str()) {
                        self.report.animated = true;
                        if self.strip_animation {
                            continue;
                        }
                        if child
                            .attribute("attributeName")
                            .is_some_and(forbidden_animation_target)
                        {
                            self.report.removed += 1;
                            continue;
                        }
                    }
                    if name == "use" {
                        self.uses += 1;
                        if self.uses > MAX_USE_REFERENCES {
                            self.report.removed += 1;
                            continue;
                        }
                    }
                    self.element(&mut child, false);
                    if name == "use" && child.attribute("href").is_none() {
                        // A `use` without a (local) reference draws nothing.
                        self.report.removed += 1;
                        continue;
                    }
                    kept.push(Node::Element(child));
                }
            }
        }
        kept
    }

    fn attributes(&mut self, element: &mut Element, is_root: bool) {
        let name = local_name(&element.name).to_string();
        let attributes = std::mem::take(&mut element.attributes);
        let mut kept: Vec<(String, String)> = Vec::with_capacity(attributes.len());
        for (key, value) in attributes {
            // `xlink:href` becomes `href` (SVG 2); every other prefix but
            // `xml:` is foreign.
            let key = if key == "xlink:href" {
                "href".to_string()
            } else {
                key
            };
            if key.starts_with("xmlns") {
                continue;
            }
            if !ATTRIBUTES.contains(&key.as_str()) {
                self.report.removed += 1;
                continue;
            }
            if key == "href" {
                if !HREF_ELEMENTS.contains(&name.as_str()) || !is_local_fragment(&value) {
                    self.report.removed += 1;
                    continue;
                }
            } else if key == "style" {
                let mut report = CssReport::default();
                let style = css::sanitize_declarations(&value, self.strip_animation, &mut report);
                self.merge(report);
                if !style.is_empty() {
                    kept.push((key, style));
                }
                continue;
            } else if !css::is_safe_value(&value) {
                self.report.removed += 1;
                continue;
            }
            // SMIL `from`/`to`/`values` can only set the targeted attribute,
            // which `forbidden_animation_target` already restricts.
            if kept.iter().any(|(existing, _)| *existing == key) {
                continue;
            }
            kept.push((key, value));
        }
        if is_root {
            // Root sizing is decided by the renderer from the viewBox.
            kept.retain(|(key, _)| key != "x" && key != "y");
        }
        element.attributes = kept;
    }

    fn merge(&mut self, report: CssReport) {
        self.report.removed += report.removed;
        self.report.animated |= report.animated;
    }
}

/// `#id` with a plain identifier.
fn is_local_fragment(value: &str) -> bool {
    let value = value.trim();
    value.len() > 1
        && value.starts_with('#')
        && value[1..]
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':'))
}
