//! Streaming repair: make a truncated SVG source well-formed.

/// Turn a possibly truncated SVG source into a well-formed document: cut the
/// trailing construct still being written (a tag without its `>`, an
/// unterminated comment, CDATA section, processing instruction or
/// declaration, a dangling `&entity`) and close every element left open,
/// innermost first. A complete document passes through unchanged. Returns
/// `None` until the root `<svg …>` start tag is complete.
pub fn close_partial_svg(source: &str) -> Option<String> {
    let bytes = source.as_bytes();
    let mut open: Vec<&str> = Vec::new();
    let mut root_seen = false;
    let mut end: usize;
    let mut index = 0;
    loop {
        let Some(lt) = source[index..].find('<').map(|at| at + index) else {
            // Trailing text: an entity reference still being written would not parse.
            end = match source[index..].rfind('&') {
                Some(amp) if !source[index + amp..].contains(';') => index + amp,
                _ => source.len(),
            };
            break;
        };
        end = lt;
        let Some(close) = construct_end(source, lt) else {
            break;
        };
        if source[lt..].starts_with("</") {
            let name = source[lt + 2..close - 1].trim();
            if let Some(at) = open.iter().rposition(|open_name| *open_name == name) {
                open.truncate(at);
            }
        } else if bytes.get(lt + 1) != Some(&b'!') && bytes.get(lt + 1) != Some(&b'?') {
            let tag = &source[lt + 1..close];
            let name_end = tag
                .find(|c: char| c.is_whitespace() || c == '/' || c == '>')
                .unwrap_or(tag.len());
            let name = &tag[..name_end];
            if !name.is_empty() {
                if bytes[close - 2] != b'/' {
                    open.push(name);
                }
                if !root_seen && (name == "svg" || name.ends_with(":svg")) {
                    root_seen = true;
                }
            }
        }
        index = close;
    }
    if !root_seen {
        return None;
    }
    let mut document = String::with_capacity(end + open.len() * 8);
    document.push_str(&source[..end]);
    for name in open.iter().rev() {
        document.push_str("</");
        document.push_str(name);
        document.push('>');
    }
    Some(document)
}

/// Index just past the markup construct starting at `<` (`lt`), or `None`
/// when it is not complete yet.
fn construct_end(source: &str, lt: usize) -> Option<usize> {
    let after = |terminator: &str, from: usize| {
        source[from..]
            .find(terminator)
            .map(|at| from + at + terminator.len())
    };
    let rest = &source[lt..];
    if rest.starts_with("<!--") {
        return after("-->", lt + 4);
    }
    if rest.starts_with("<![CDATA[") {
        return after("]]>", lt + 9);
    }
    if rest.starts_with("<?") {
        return after("?>", lt + 2);
    }
    // Tags and declarations end at the first `>` outside a quoted value; a
    // doctype's internal subset (`[…]`) may hold `>` of its own.
    let bytes = source.as_bytes();
    let declaration = bytes.get(lt + 1) == Some(&b'!');
    let mut quote: Option<u8> = None;
    let mut subset = false;
    for (at, &byte) in bytes.iter().enumerate().skip(lt + 1) {
        match quote {
            Some(q) if byte == q => quote = None,
            Some(_) => {}
            None => match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'[' if declaration => subset = true,
                b']' => subset = false,
                b'>' if !subset => return Some(at + 1),
                _ => {}
            },
        }
    }
    None
}
