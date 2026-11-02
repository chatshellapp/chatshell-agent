//! Minimal owned XML tree for figures: parse with quick-xml, serialize back.
//!
//! Values are stored unescaped and escaped again on output, so nothing the
//! source wrote survives as raw markup.

use quick_xml::Reader;
use quick_xml::XmlVersion;
use quick_xml::escape::resolve_predefined_entity;
use quick_xml::events::{BytesRef, BytesStart, Event};

use super::{MAX_DEPTH, MAX_ELEMENTS, SvgFigureError};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Element {
    /// Qualified name as written (`svg`, `svg:rect`, `xlink:foo`).
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub children: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Node {
    Element(Element),
    Text(String),
}

impl Node {
    /// Whether this node draws something: any element other than metadata.
    pub fn is_drawable(&self) -> bool {
        match self {
            Node::Element(element) => !matches!(
                local_name(&element.name),
                "title" | "desc" | "defs" | "style" | "metadata"
            ),
            Node::Text(_) => false,
        }
    }
}

impl Element {
    pub fn attribute(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    pub fn set_attribute(&mut self, name: &str, value: String) {
        match self.attributes.iter_mut().find(|(key, _)| key == name) {
            Some((_, existing)) => *existing = value,
            None => self.attributes.push((name.to_string(), value)),
        }
    }

    pub fn remove_attribute(&mut self, name: &str) -> Option<String> {
        let at = self.attributes.iter().position(|(key, _)| key == name)?;
        Some(self.attributes.remove(at).1)
    }

    /// Concatenated text children (for `<style>`).
    pub fn text(&self) -> String {
        let mut text = String::new();
        for child in &self.children {
            if let Node::Text(chunk) = child {
                text.push_str(chunk);
            }
        }
        text
    }
}

/// The part of a qualified name after its prefix.
pub(super) fn local_name(name: &str) -> &str {
    name.rsplit_once(':').map_or(name, |(_, local)| local)
}

/// The namespace prefix of a qualified name, if any.
pub(super) fn prefix(name: &str) -> Option<&str> {
    name.split_once(':').map(|(prefix, _)| prefix)
}

/// `svg` or `<prefix>:svg`.
pub(super) fn is_svg_name(name: &str) -> bool {
    local_name(name) == "svg"
}

/// Parse a document into its root element, enforcing element and depth
/// limits. DOCTYPE, comments, processing instructions are dropped.
pub(super) fn parse(document: &str) -> Result<Element, SvgFigureError> {
    let mut reader = Reader::from_str(document);
    let config = reader.config_mut();
    config.trim_text(false);
    config.check_end_names = true;
    config.expand_empty_elements = false;

    let mut stack: Vec<Element> = Vec::new();
    let mut root: Option<Element> = None;
    let mut elements = 0usize;
    loop {
        let event = reader
            .read_event()
            .map_err(|error| SvgFigureError::Parse(error.to_string()))?;
        match event {
            Event::Start(_) | Event::Empty(_) if root.is_some() => {
                return Err(SvgFigureError::Parse(
                    "content after the root element".into(),
                ));
            }
            Event::Start(start) => {
                elements += 1;
                if elements > MAX_ELEMENTS || stack.len() >= MAX_DEPTH {
                    return Err(SvgFigureError::TooLarge);
                }
                stack.push(element_from(&start)?);
            }
            Event::Empty(start) => {
                elements += 1;
                if elements > MAX_ELEMENTS || stack.len() >= MAX_DEPTH {
                    return Err(SvgFigureError::TooLarge);
                }
                let element = element_from(&start)?;
                match stack.last_mut() {
                    Some(parent) => parent.children.push(Node::Element(element)),
                    None => root = Some(element),
                }
            }
            Event::End(_) => {
                let element = stack
                    .pop()
                    .ok_or_else(|| SvgFigureError::Parse("unmatched end tag".into()))?;
                match stack.last_mut() {
                    Some(parent) => parent.children.push(Node::Element(element)),
                    None => root = Some(element),
                }
            }
            Event::Text(text) => {
                if let Some(parent) = stack.last_mut() {
                    push_text(parent, &text.xml_content(XmlVersion::Implicit1_0));
                } else if !text.trim().is_empty() {
                    return Err(SvgFigureError::Parse(
                        "text outside the root element".into(),
                    ));
                }
            }
            Event::CData(cdata) => {
                if let Some(parent) = stack.last_mut() {
                    let text = cdata.xml_content(XmlVersion::Implicit1_0);
                    push_text(parent, &text);
                }
            }
            Event::GeneralRef(reference) => {
                if let Some(parent) = stack.last_mut() {
                    let mut text = String::new();
                    push_reference(&mut text, &reference);
                    push_text(parent, &text);
                }
            }
            Event::Comment(_) | Event::PI(_) | Event::Decl(_) | Event::DocType(_) => {}
            Event::Eof => break,
        }
    }
    if !stack.is_empty() {
        return Err(SvgFigureError::Parse("unclosed element".into()));
    }
    root.ok_or(SvgFigureError::NoRoot)
}

fn push_text(parent: &mut Element, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(Node::Text(last)) = parent.children.last_mut() {
        last.push_str(text);
    } else {
        parent.children.push(Node::Text(text.to_string()));
    }
}

/// Character references resolve; predefined entities resolve; anything else
/// (custom entities from a DOCTYPE) is kept as literal text.
fn push_reference(out: &mut String, reference: &BytesRef<'_>) {
    if let Ok(Some(ch)) = reference.resolve_char_ref() {
        out.push(ch);
        return;
    }
    let name: &str = reference;
    match resolve_predefined_entity(name) {
        Some(value) => out.push_str(value),
        None => {
            out.push('&');
            out.push_str(name);
            out.push(';');
        }
    }
}

fn element_from(start: &BytesStart<'_>) -> Result<Element, SvgFigureError> {
    let name = start.name().as_ref().to_string();
    let mut attributes = Vec::new();
    for attribute in start.attributes() {
        let attribute = attribute.map_err(|error| SvgFigureError::Parse(error.to_string()))?;
        let key = attribute.key.as_ref().to_string();
        let value = attribute
            .normalized_value(XmlVersion::Implicit1_0)
            .map_err(|error| SvgFigureError::Parse(error.to_string()))?
            .into_owned();
        attributes.push((key, value));
    }
    Ok(Element {
        name,
        attributes,
        children: Vec::new(),
    })
}

/// Serialize with every text and attribute value escaped.
pub(super) fn serialize(root: &Element) -> String {
    let mut out = String::with_capacity(4096);
    write_element(&mut out, root);
    out
}

fn write_element(out: &mut String, element: &Element) {
    out.push('<');
    out.push_str(&element.name);
    for (key, value) in &element.attributes {
        out.push(' ');
        out.push_str(key);
        out.push_str("=\"");
        escape_into(out, value, true);
        out.push('"');
    }
    if element.children.is_empty() {
        out.push_str("/>");
        return;
    }
    out.push('>');
    for child in &element.children {
        match child {
            Node::Element(child) => write_element(out, child),
            Node::Text(text) => escape_into(out, text, false),
        }
    }
    out.push_str("</");
    out.push_str(&element.name);
    out.push('>');
}

fn escape_into(out: &mut String, value: &str, attribute: bool) {
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' if attribute => out.push_str("&quot;"),
            '\n' if attribute => out.push_str("&#10;"),
            '\t' if attribute => out.push_str("&#9;"),
            '\r' => out.push_str("&#13;"),
            _ => out.push(ch),
        }
    }
}
