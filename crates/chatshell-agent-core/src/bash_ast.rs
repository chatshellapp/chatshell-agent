//! AST-based bash command security analysis using tree-sitter.
//!
//! Parses bash commands into an AST and extracts security-relevant structure
//! that regex/character scanners cannot reliably detect. The key advantage is
//! that tree-sitter understands bash grammar, so it correctly distinguishes:
//! - Escaped operators (`\;` is a word, not a separator)
//! - Quoted content (no false positives from patterns inside quotes)
//! - Nested structures (command substitution depth, subshells)
//! - Actual compound operators vs arguments that look like operators

use tree_sitter::{Node, Parser};

const MAX_COMMAND_LENGTH: usize = 10_000;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Complete AST-based analysis of a bash command.
#[derive(Debug, Clone)]
pub struct AstAnalysis {
    pub compound: CompoundStructure,
    pub dangerous: DangerousPatterns,
    /// True when tree-sitter finds actual `;`, `&&`, `||` operator nodes
    /// (not `\;` which is a word argument).
    pub has_actual_operators: bool,
    /// True when the parse contained ERROR nodes (malformed syntax).
    pub has_parse_errors: bool,
    /// Every redirect of the command, with heredoc geometry (ADR 09).
    pub redirects: RedirectAnalysis,
    /// Some command's name cannot be read literally (`"c""d"`, `c\d`,
    /// `$'cd'`, `$x`), also after `builtin` / `command`: which command
    /// runs, `cd` included, is not provable from the text.
    pub unreadable_command_name: bool,
}

/// The shape of one redirect occurrence (ADR 09 §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedirectShape {
    /// `>` / `N>`
    Output,
    /// `>>` / `N>>`
    Append,
    /// `>|`
    Clobber,
    /// `&>`
    OutputBoth,
    /// `&>>`
    AppendBoth,
    /// `>&file` (a file target, not a descriptor)
    OutputDup,
    /// `2>&1`, `>&2`, `N>&-`, `<&-`: descriptor plumbing, no file target.
    FdDup,
    /// Plain `<` file input.
    Input,
    /// `<<<` herestring.
    Herestring,
    /// A heredoc whose own statement writes it to a file with a data-only
    /// command (`cat > f <<EOF`).
    HeredocWrite,
    /// Any other heredoc: pure input, or fed to a command that may execute
    /// its stdin.
    HeredocInput,
    /// Anything else (`<&N`, `<>`, an operator this table does not know).
    Other,
}

impl RedirectShape {
    /// Output-family shapes: each must carry a file target.
    pub fn writes_file(self) -> bool {
        matches!(
            self,
            RedirectShape::Output
                | RedirectShape::Append
                | RedirectShape::Clobber
                | RedirectShape::OutputBoth
                | RedirectShape::AppendBoth
                | RedirectShape::OutputDup
        )
    }
}

/// The destination of a file redirect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedirectTarget {
    /// A literal path (quotes removed; `~` not yet expanded).
    Literal(String),
    /// Expansion, glob, substitution, or concatenation: not placeable.
    Unresolvable,
}

/// One redirect occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Redirect {
    pub shape: RedirectShape,
    /// `None` for shapes without a destination (fd-dup, heredocs) and for
    /// an output operator whose destination could not be found.
    pub target: Option<RedirectTarget>,
}

/// Redirect structure of a command, read from the AST.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RedirectAnalysis {
    /// Every redirect node (heredoc body text never yields one; command
    /// substitutions inside unquoted bodies do, since they execute).
    pub redirects: Vec<Redirect>,
    /// Byte ranges of heredoc bodies (data, not shell syntax).
    pub heredoc_bodies: Vec<(usize, usize)>,
    /// A `<` or `>` byte outside strings, heredoc bodies, and comments
    /// that is not a redirect operator (`[[ a > b ]]`, `$((1 > 2))`, a
    /// recovered parse): the redirect set is not provably complete.
    pub stray_operator: bool,
    /// A command-separating newline that is neither a heredoc header
    /// boundary nor inside a heredoc body.
    pub unattributed_newline: bool,
}

/// Commands that only copy their stdin as data: a heredoc fed to one of
/// these is data. An allowlist, so a command that may execute its stdin
/// (`bash`, `python`, `env`, `eval`, ...) never qualifies.
const HEREDOC_DATA_COMMANDS: &[&str] = &["cat"];

/// Compound command structure extracted from the AST.
#[derive(Debug, Clone, Default)]
pub struct CompoundStructure {
    /// Top-level compound operators (`&&`, `||`, `;`).
    pub operators: Vec<String>,
    /// Individual command segments split by operators/pipes.
    pub segments: Vec<String>,
    pub has_pipeline: bool,
    pub has_subshell: bool,
    pub has_command_group: bool,
}

/// Dangerous patterns detected via AST node types.
#[derive(Debug, Clone, Default)]
pub struct DangerousPatterns {
    pub has_command_substitution: bool,
    pub has_process_substitution: bool,
    pub has_parameter_expansion: bool,
    pub has_heredoc: bool,
    pub has_comment: bool,
}

// ---------------------------------------------------------------------------
// Parser (thread-local for reuse)
// ---------------------------------------------------------------------------

thread_local! {
    static PARSER: std::cell::RefCell<Option<Parser>> = const { std::cell::RefCell::new(None) };
}

fn with_parser<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut Parser) -> R,
{
    PARSER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let mut parser = Parser::new();
            if parser
                .set_language(&tree_sitter_bash::LANGUAGE.into())
                .is_err()
            {
                return None;
            }
            *slot = Some(parser);
        }
        slot.as_mut().map(f)
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a command and perform full AST analysis.
/// Returns `None` if the command is too long or the parser is unavailable.
pub fn analyze(command: &str) -> Option<AstAnalysis> {
    if command.is_empty() || command.len() > MAX_COMMAND_LENGTH {
        return None;
    }

    with_parser(|parser| {
        let tree = parser.parse(command, None)?;
        let root = tree.root_node();

        let has_parse_errors = has_error_nodes(root);
        let compound = extract_compound_structure(root, command);
        let dangerous = extract_dangerous_patterns(root);
        let has_actual_operators = check_actual_operators(root);
        let redirects = analyze_redirects(root, command);
        let unreadable_command_name = has_unreadable_command_name(root, command);

        Some(AstAnalysis {
            compound,
            dangerous,
            has_actual_operators,
            has_parse_errors,
            redirects,
            unreadable_command_name,
        })
    })
    .flatten()
}

// ---------------------------------------------------------------------------
// Compound structure extraction
// ---------------------------------------------------------------------------

fn extract_compound_structure(root: Node, command: &str) -> CompoundStructure {
    let mut result = CompoundStructure::default();
    walk_top_level(root, command, &mut result);
    if result.segments.is_empty() {
        result.segments.push(command.to_string());
    }
    result
}

fn walk_top_level(node: Node, command: &str, out: &mut CompoundStructure) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "list" => walk_list(child, command, out),
            ";" => out.operators.push(";".into()),
            "pipeline" => {
                out.has_pipeline = true;
                out.segments.push(node_text(child, command).into());
            }
            "subshell" => {
                out.has_subshell = true;
                out.segments.push(node_text(child, command).into());
            }
            "compound_statement" => {
                out.has_command_group = true;
                out.segments.push(node_text(child, command).into());
            }
            "command" | "declaration_command" | "variable_assignment" => {
                out.segments.push(node_text(child, command).into());
            }
            "redirected_statement" => {
                let mut found_inner = false;
                let mut inner_cursor = child.walk();
                for inner in child.children(&mut inner_cursor) {
                    if inner.kind() == "file_redirect" || inner.kind() == "heredoc_redirect" {
                        continue;
                    }
                    found_inner = true;
                    walk_top_level(inner, command, out);
                }
                if !found_inner {
                    out.segments.push(node_text(child, command).into());
                }
            }
            "negated_command" => {
                out.segments.push(node_text(child, command).into());
                walk_top_level(child, command, out);
            }
            "if_statement"
            | "while_statement"
            | "for_statement"
            | "case_statement"
            | "function_definition" => {
                out.segments.push(node_text(child, command).into());
                walk_top_level(child, command, out);
            }
            _ => {}
        }
    }
}

fn walk_list(node: Node, command: &str, out: &mut CompoundStructure) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "&&" | "||" => out.operators.push(child.kind().into()),
            "list" | "redirected_statement" => {
                walk_top_level(node_as_parent(child), command, out);
            }
            "pipeline" => {
                out.has_pipeline = true;
                out.segments.push(node_text(child, command).into());
            }
            "subshell" => {
                out.has_subshell = true;
                out.segments.push(node_text(child, command).into());
            }
            "compound_statement" => {
                out.has_command_group = true;
                out.segments.push(node_text(child, command).into());
            }
            _ => {
                out.segments.push(node_text(child, command).into());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dangerous pattern detection
// ---------------------------------------------------------------------------

fn extract_dangerous_patterns(root: Node) -> DangerousPatterns {
    let mut result = DangerousPatterns::default();
    walk_dangerous(root, &mut result);
    result
}

fn walk_dangerous(node: Node, out: &mut DangerousPatterns) {
    match node.kind() {
        "command_substitution" => out.has_command_substitution = true,
        "process_substitution" => out.has_process_substitution = true,
        "expansion" => out.has_parameter_expansion = true,
        "heredoc_redirect" => out.has_heredoc = true,
        "comment" => out.has_comment = true,
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_dangerous(child, out);
    }
}

// ---------------------------------------------------------------------------
// Actual operator detection
// ---------------------------------------------------------------------------

/// Check if the AST contains actual operator nodes (`;`, `&&`, `||`).
/// Tree-sitter parses `\;` as part of a `word` node, NOT as a `;` operator.
fn check_actual_operators(root: Node) -> bool {
    walk_for_operators(root)
}

fn walk_for_operators(node: Node) -> bool {
    match node.kind() {
        ";" | "&&" | "||" => return true,
        "list" => return true,
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if walk_for_operators(child) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Error node detection
// ---------------------------------------------------------------------------

fn has_error_nodes(root: Node) -> bool {
    walk_for_errors(root)
}

fn walk_for_errors(node: Node) -> bool {
    if node.is_error() || node.is_missing() {
        return true;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if walk_for_errors(child) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Redirect analysis
// ---------------------------------------------------------------------------

/// Node kinds whose bytes are data, not shell operators.
const OPAQUE_KINDS: &[&str] = &[
    "string",
    "raw_string",
    "ansi_c_string",
    "translated_string",
    "heredoc_body",
    "comment",
];

#[derive(Default)]
struct RedirectWalk {
    redirects: Vec<Redirect>,
    bodies: Vec<(usize, usize)>,
    opaque: Vec<(usize, usize)>,
    operators: Vec<(usize, usize)>,
    /// Per heredoc: where its header boundary must end (body start, or
    /// the terminator for an empty body) and where its start token ends.
    headers: Vec<(usize, usize)>,
}

fn analyze_redirects(root: Node, command: &str) -> RedirectAnalysis {
    let mut walk = RedirectWalk::default();
    walk_redirects(root, command, &mut walk);
    let inside = |ranges: &[(usize, usize)], i: usize| ranges.iter().any(|&(a, b)| i >= a && i < b);
    let bytes = command.as_bytes();

    let stray_operator = bytes.iter().enumerate().any(|(i, b)| {
        matches!(b, b'<' | b'>') && !inside(&walk.opaque, i) && !inside(&walk.operators, i)
    });

    let mut unattributed_newline = false;
    for (i, &b) in bytes.iter().enumerate() {
        if !matches!(b, b'\n' | b'\r') || inside(&walk.opaque, i) {
            continue;
        }
        // Only a newline followed by more text separates commands.
        if command[i + 1..].trim().is_empty() {
            continue;
        }
        // Backslash continuation at a word boundary (as `validate_newlines`).
        if i > 0 && bytes[i - 1] == b'\\' && (i < 2 || bytes[i - 2].is_ascii_whitespace()) {
            continue;
        }
        let header = walk.headers.iter().any(|&(start_end, body_start)| {
            i >= start_end
                && i < body_start
                && command[i + 1..body_start]
                    .bytes()
                    .all(|c| matches!(c, b' ' | b'\t' | b'\r' | b'\n'))
        });
        if !header {
            unattributed_newline = true;
            break;
        }
    }

    RedirectAnalysis {
        redirects: walk.redirects,
        heredoc_bodies: walk.bodies,
        stray_operator,
        unattributed_newline,
    }
}

fn walk_redirects(node: Node, command: &str, out: &mut RedirectWalk) {
    if OPAQUE_KINDS.contains(&node.kind()) {
        out.opaque.push((node.start_byte(), node.end_byte()));
    }
    match node.kind() {
        "file_redirect" => {
            let redirect = file_redirect(node, command, out);
            out.redirects.push(redirect);
        }
        "herestring_redirect" => {
            mark_operator(node, out);
            out.redirects.push(Redirect {
                shape: RedirectShape::Herestring,
                target: None,
            });
        }
        "heredoc_redirect" => {
            mark_operator(node, out);
            out.redirects.push(Redirect {
                shape: heredoc_shape(node, command),
                target: None,
            });
            let mut start_end = None;
            let mut body = None;
            let mut end = None;
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                match child.kind() {
                    "heredoc_start" => start_end = Some(child.end_byte()),
                    "heredoc_body" => body = Some((child.start_byte(), child.end_byte())),
                    "heredoc_end" => end = Some(child.start_byte()),
                    _ => {}
                }
            }
            if let Some(range) = body {
                out.bodies.push(range);
            }
            let body_start = match body {
                Some((start, end_byte)) if end_byte > start => Some(start),
                _ => end.or(body.map(|(start, _)| start)),
            };
            if let (Some(start_end), Some(body_start)) = (start_end, body_start) {
                out.headers.push((start_end, body_start));
            }
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_redirects(child, command, out);
    }
}

/// Record the anonymous operator token(s) of a redirect node.
fn mark_operator(node: Node, out: &mut RedirectWalk) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if !child.is_named() {
            out.operators.push((child.start_byte(), child.end_byte()));
        }
    }
}

fn file_redirect(node: Node, command: &str, out: &mut RedirectWalk) -> Redirect {
    let mut operator = None;
    let mut destination = None;
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if !child.is_named() {
            if operator.is_none() {
                operator = Some(node_text(child, command));
            }
            out.operators.push((child.start_byte(), child.end_byte()));
        } else if child.kind() != "file_descriptor" && operator.is_some() && destination.is_none() {
            destination = Some(child);
        }
    }
    let target = destination.map(|dest| redirect_target(dest, command));
    let shape = match operator.unwrap_or("") {
        ">" => RedirectShape::Output,
        ">>" => RedirectShape::Append,
        ">|" => RedirectShape::Clobber,
        "&>" => RedirectShape::OutputBoth,
        "&>>" => RedirectShape::AppendBoth,
        ">&-" | "<&-" => RedirectShape::FdDup,
        ">&" => match destination {
            Some(dest) if dest.kind() == "number" => RedirectShape::FdDup,
            Some(dest) if node_text(dest, command) == "-" => RedirectShape::FdDup,
            _ => RedirectShape::OutputDup,
        },
        "<" => RedirectShape::Input,
        _ => RedirectShape::Other,
    };
    Redirect {
        shape,
        target: if shape == RedirectShape::FdDup {
            None
        } else {
            target
        },
    }
}

/// A destination node as a literal path, when it is one.
fn redirect_target(node: Node, command: &str) -> RedirectTarget {
    let text = node_text(node, command);
    let literal = match node.kind() {
        "word" | "number" => Some(text.to_string()),
        "raw_string" => text
            .strip_prefix('\'')
            .and_then(|t| t.strip_suffix('\''))
            .map(str::to_string),
        "string" => {
            let mut cursor = node.walk();
            let plain = node
                .named_children(&mut cursor)
                .all(|child| child.kind() == "string_content");
            plain
                .then(|| text.strip_prefix('"').and_then(|t| t.strip_suffix('"')))
                .flatten()
                .map(str::to_string)
        }
        _ => None,
    };
    match literal {
        // Quoted text cannot expand, but a bare word still can: tilde
        // forms other than `~` / `~/`, globs, braces, and escapes.
        Some(path) if node.kind() == "word" && !plain_word(&path) => RedirectTarget::Unresolvable,
        Some(path) if path.contains(['$', '`']) && node.kind() == "string" => {
            RedirectTarget::Unresolvable
        }
        Some(path) if !path.is_empty() => RedirectTarget::Literal(path),
        _ => RedirectTarget::Unresolvable,
    }
}

fn plain_word(word: &str) -> bool {
    if word.contains(['$', '`', '*', '?', '[', '{', '\\', '\'', '"']) {
        return false;
    }
    !word.starts_with('~') || word == "~" || word.starts_with("~/")
}

/// Classify a heredoc by its own statement.
fn heredoc_shape(heredoc: Node, command: &str) -> RedirectShape {
    let Some(statement) = heredoc
        .parent()
        .filter(|p| p.kind() == "redirected_statement")
    else {
        return RedirectShape::HeredocInput;
    };
    // The statement's command must copy stdin as data.
    let data_command = statement
        .child_by_field_name("body")
        .filter(|body| body.kind() == "command")
        .and_then(|body| body.child_by_field_name("name"))
        .map(|name| node_text(name, command))
        .map(|name| name.rsplit('/').next().unwrap_or(name))
        .is_some_and(|name| HEREDOC_DATA_COMMANDS.contains(&name));
    if !data_command {
        return RedirectShape::HeredocInput;
    }
    // And the statement itself must write to a file: a write redirect of
    // its own (beside the heredoc, or nested in it by the grammar).
    let mut writes = false;
    let mut visit = |node: Node| {
        if node.kind() == "file_redirect" {
            let mut cursor = node.walk();
            let op = node
                .children(&mut cursor)
                .find(|c| !c.is_named())
                .map(|c| node_text(c, command));
            if matches!(op, Some(">" | ">>" | ">|" | "&>" | "&>>")) {
                writes = true;
            }
        }
    };
    let mut cursor = statement.walk();
    for child in statement.children(&mut cursor) {
        visit(child);
        if child.id() == heredoc.id() {
            let mut inner = child.walk();
            for nested in child.children(&mut inner) {
                visit(nested);
            }
        }
    }
    if writes {
        RedirectShape::HeredocWrite
    } else {
        RedirectShape::HeredocInput
    }
}

// ---------------------------------------------------------------------------
// Command names
// ---------------------------------------------------------------------------

/// Builtins that run the builtin named by their first operand.
const NAME_WRAPPERS: &[&str] = &["builtin", "command"];

/// Whether any command's effective name (past `builtin` / `command` and
/// their options) is not a plain literal word.
fn has_unreadable_command_name(node: Node, command: &str) -> bool {
    if node.kind() == "command" && !command_name_readable(node, command) {
        return true;
    }
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .any(|child| has_unreadable_command_name(child, command))
}

fn command_name_readable(node: Node, command: &str) -> bool {
    let mut words: Vec<Node> = node.child_by_field_name("name").into_iter().collect();
    let mut cursor = node.walk();
    words.extend(node.children_by_field_name("argument", &mut cursor));
    let mut wrapped = false;
    for word in words {
        let Some(text) = literal_word(word, command) else {
            return false;
        };
        if wrapped && text.starts_with('-') {
            continue;
        }
        if NAME_WRAPPERS.contains(&text) {
            wrapped = true;
            continue;
        }
        return true;
    }
    // A wrapper with no operand runs nothing; no name is fine too.
    true
}

/// The text of a plain literal word (a command name or operand), or
/// `None` when quoting, escapes, or expansion make it unreadable.
fn literal_word<'a>(node: Node, command: &'a str) -> Option<&'a str> {
    let node = if node.kind() == "command_name" {
        node.named_child(0)?
    } else {
        node
    };
    if !matches!(node.kind(), "word" | "number") {
        return None;
    }
    let text = node_text(node, command);
    (!text.contains(['$', '`', '\'', '"', '\\', '{', '*', '?', '['])).then_some(text)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn node_text<'a>(node: Node, source: &'a str) -> &'a str {
    &source[node.byte_range()]
}

/// Wraps a child node so we can call walk_top_level on it.
/// Since walk_top_level iterates over children, we pass the node itself
/// as if it were a parent — the function will iterate its children.
fn node_as_parent(node: Node) -> Node {
    node
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_command() {
        let a = analyze("echo hello").unwrap();
        assert!(!a.has_parse_errors);
        assert!(!a.has_actual_operators);
        assert!(!a.dangerous.has_command_substitution);
        assert_eq!(a.compound.segments.len(), 1);
    }

    #[test]
    fn compound_and() {
        let a = analyze("cd /tmp && ls -la").unwrap();
        assert!(a.has_actual_operators);
        assert!(a.compound.operators.contains(&"&&".to_string()));
        assert!(a.compound.segments.len() >= 2);
    }

    #[test]
    fn compound_semicolon() {
        let a = analyze("echo a; echo b").unwrap();
        assert!(a.has_actual_operators);
        assert!(a.compound.segments.len() >= 2);
    }

    #[test]
    fn escaped_semicolon_is_not_operator() {
        let a = analyze(r"find . -name '*.txt' -exec rm {} \;").unwrap();
        assert!(
            !a.has_actual_operators,
            "\\; should not be an actual operator"
        );
    }

    #[test]
    fn pipeline() {
        let a = analyze("cat file | grep foo | wc -l").unwrap();
        assert!(a.compound.has_pipeline);
    }

    #[test]
    fn command_substitution() {
        let a = analyze("echo $(whoami)").unwrap();
        assert!(a.dangerous.has_command_substitution);
    }

    #[test]
    fn command_substitution_in_single_quotes_still_parsed() {
        // tree-sitter sees the raw_string, but $(whoami) inside is literal
        // The key point: tree-sitter does NOT create a command_substitution node
        // inside single quotes (it's raw_string content).
        let a = analyze("echo '$(whoami)'").unwrap();
        assert!(
            !a.dangerous.has_command_substitution,
            "single-quoted $() should not be command substitution"
        );
    }

    #[test]
    fn command_substitution_in_double_quotes() {
        let a = analyze(r#"echo "$(whoami)""#).unwrap();
        assert!(a.dangerous.has_command_substitution);
    }

    #[test]
    fn process_substitution() {
        let a = analyze("diff <(ls dir1) <(ls dir2)").unwrap();
        assert!(a.dangerous.has_process_substitution);
    }

    #[test]
    fn parameter_expansion() {
        let a = analyze("echo ${HOME}").unwrap();
        assert!(a.dangerous.has_parameter_expansion);
    }

    #[test]
    fn heredoc() {
        let a = analyze("cat <<EOF\nhello\nEOF").unwrap();
        assert!(a.dangerous.has_heredoc);
    }

    #[test]
    fn subshell() {
        let a = analyze("(cd /tmp && ls)").unwrap();
        assert!(a.compound.has_subshell);
    }

    #[test]
    fn command_group() {
        let a = analyze("{ echo a; echo b; }").unwrap();
        assert!(a.compound.has_command_group);
    }

    #[test]
    fn comment() {
        let a = analyze("echo hello # this is a comment").unwrap();
        assert!(a.dangerous.has_comment);
    }

    #[test]
    fn parse_error() {
        let a = analyze("echo 'unterminated").unwrap();
        assert!(a.has_parse_errors);
    }

    #[test]
    fn empty_returns_none() {
        assert!(analyze("").is_none());
    }

    #[test]
    fn too_long_returns_none() {
        let long = "a".repeat(MAX_COMMAND_LENGTH + 1);
        assert!(analyze(&long).is_none());
    }

    #[test]
    fn backtick_substitution() {
        let a = analyze("echo `whoami`").unwrap();
        assert!(a.dangerous.has_command_substitution);
    }

    #[test]
    fn if_statement() {
        let a = analyze("if true; then echo yes; fi").unwrap();
        assert!(a.has_actual_operators);
        assert!(!a.compound.segments.is_empty());
    }

    #[test]
    fn nested_substitution() {
        let a = analyze("echo $(echo $(date))").unwrap();
        assert!(a.dangerous.has_command_substitution);
    }

    #[test]
    fn real_world_safe() {
        let a = analyze("git status && cargo build --release").unwrap();
        assert!(a.has_actual_operators);
        assert!(!a.dangerous.has_command_substitution);
        assert!(!a.compound.has_subshell);
    }

    #[test]
    fn real_world_find_exec() {
        let a = analyze(r"find . -type f -name '*.rs' -exec grep 'TODO' {} \;").unwrap();
        assert!(!a.has_actual_operators);
        assert!(!a.dangerous.has_command_substitution);
    }

    fn shapes(command: &str) -> Vec<RedirectShape> {
        analyze(command)
            .unwrap()
            .redirects
            .redirects
            .iter()
            .map(|r| r.shape)
            .collect()
    }

    #[test]
    fn redirect_shapes_come_from_the_ast() {
        use RedirectShape::*;
        assert_eq!(shapes("echo hi > f"), vec![Output]);
        assert_eq!(shapes("echo hi >> f"), vec![Append]);
        assert_eq!(shapes("echo hi >| f"), vec![Clobber]);
        assert_eq!(shapes("cmd &> f"), vec![OutputBoth]);
        assert_eq!(shapes("cmd &>> f"), vec![AppendBoth]);
        assert_eq!(shapes("echo hi >& f"), vec![OutputDup]);
        assert_eq!(shapes("cmd >&2"), vec![FdDup]);
        assert_eq!(shapes("cmd 2>&1"), vec![FdDup]);
        assert_eq!(shapes("cmd 3>&-"), vec![FdDup]);
        assert_eq!(shapes("diff < f"), vec![Input]);
        assert_eq!(shapes("cat <<< x"), vec![Herestring]);
        assert_eq!(
            shapes("cat > f <<'EOF'\nx\nEOF"),
            vec![Output, HeredocWrite]
        );
        assert_eq!(shapes("cat <<EOF > f\nx\nEOF"), vec![HeredocWrite, Output]);
        assert_eq!(shapes("cat <<'EOF'\nx\nEOF"), vec![HeredocInput]);
        assert_eq!(
            shapes("bash > f <<'EOF'\nls\nEOF"),
            vec![Output, HeredocInput]
        );
        // `> 2` writes a file named 2.
        let analysis = analyze("echo hi > 2").unwrap().redirects;
        assert_eq!(
            analysis.redirects[0].target,
            Some(RedirectTarget::Literal("2".into()))
        );
    }

    #[test]
    fn redirect_targets_resolve_only_when_literal() {
        let target = |command: &str| {
            analyze(command).unwrap().redirects.redirects[0]
                .target
                .clone()
        };
        assert_eq!(
            target("echo > 'a b'"),
            Some(RedirectTarget::Literal("a b".into()))
        );
        assert_eq!(
            target("echo > \"a b\""),
            Some(RedirectTarget::Literal("a b".into()))
        );
        assert_eq!(
            target("echo > ~/x"),
            Some(RedirectTarget::Literal("~/x".into()))
        );
        assert_eq!(
            target("echo > \"$out\""),
            Some(RedirectTarget::Unresolvable)
        );
        assert_eq!(target("echo > $out"), Some(RedirectTarget::Unresolvable));
        assert_eq!(target("echo > *.log"), Some(RedirectTarget::Unresolvable));
        assert_eq!(target("echo > x\"y\""), Some(RedirectTarget::Unresolvable));
        assert_eq!(target("echo > ~root/x"), Some(RedirectTarget::Unresolvable));
    }

    #[test]
    fn heredoc_geometry_attributes_header_newlines() {
        let r = |command: &str| analyze(command).unwrap().redirects;
        assert!(!r("cat > f <<'EOF'\nx\nEOF").unattributed_newline);
        assert!(!r("cat > f <<-'EOF'\n\tx\n\tEOF").unattributed_newline);
        assert!(!r("cat > f <<EOF\nEOF").unattributed_newline);
        assert!(!r("cat > f <<'EOF'\nx\nEOF\n").unattributed_newline);
        assert!(r("cat > f <<'EOF'\nx\nEOF\nls").unattributed_newline);
        assert!(r("cat > f <<'EOF'\nit's\nEOF\necho x > g").unattributed_newline);
        assert!(r("ls\npwd").unattributed_newline);
        assert!(!r("echo a \\\n  b").unattributed_newline);
        // Body text is data: no operators, no stray bytes.
        let body = r("cat > f <<'EOF'\na > b < c\nEOF");
        assert!(!body.stray_operator);
        assert_eq!(body.redirects.len(), 2);
        // A `>` that is not a redirect operator is stray.
        assert!(r("[[ a > b ]]").stray_operator);
        assert!(r("echo $(( 1 > 2 ))").stray_operator);
        assert!(!r("echo 'a > b' \"c < d\"").stray_operator);
    }

    #[test]
    fn unreadable_command_names_are_flagged() {
        let flagged = |command: &str| analyze(command).unwrap().unreadable_command_name;
        assert!(flagged("\"c\"\"d\" ~ && echo hi > out.txt"));
        assert!(flagged("c\\d ~"));
        assert!(flagged("$'cd' ~"));
        assert!(flagged("$x ~"));
        assert!(flagged("builtin \"c\"\"d\" ~"));
        assert!(flagged("command -p 'cd' ~"));
        assert!(flagged("ls; if true; then \"c\"d /; fi"));
        assert!(!flagged("cd ~ && ls"));
        assert!(!flagged("echo \"a b\" > 'out file'"));
        assert!(!flagged("builtin echo hi"));
        assert!(!flagged("X=1 ./build.sh --flag"));
        assert!(!flagged("cat > f <<'EOF'\n\"c\"\"d\" /\nEOF"));
    }
}
