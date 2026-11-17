//! Parser for ChatShell's OpenUI block library: `Chart`, `Series`, `Stats`,
//! and `Stat` in a ```` ```openui-lang ```` fence.
//!
//! Desktop renders these blocks with `@openuidev/lang-core` and its own
//! library (`chatshell-desktop/src/lib/openui/library.ts`). Clients that
//! draw them natively (iOS Swift Charts, later Android Compose) need the
//! same reading of the same source, so the parse lives here. It covers only
//! the static subset the prompt teaches (`visual_output/openui_prompt.txt`):
//! `name = Expression` statements whose expressions are literals, arrays,
//! component calls, and references. Anything dynamic (`$state`, `@builtins`,
//! operators, `Query`) makes its statement unusable and is skipped.
//!
//! Streaming follows lang-core: unclosed strings and brackets are closed
//! before parsing, references to names not written yet resolve to nothing,
//! and only complete series and stats are kept, so a block can render while
//! it streams.

use std::collections::HashMap;

/// A parsed block, ready to draw.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenUiBlock {
    Chart(OpenUiChart),
    Stats(Vec<OpenUiStat>),
}

/// `Chart(type, title, labels, series, description?, variant?, unit?)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenUiChart {
    pub kind: ChartKind,
    pub title: String,
    pub labels: Vec<String>,
    pub series: Vec<OpenUiSeries>,
    pub description: Option<String>,
    /// Only a variant valid for `kind` is kept (stacked: bar and area;
    /// horizontal: bar; donut: pie).
    pub variant: Option<ChartVariant>,
    /// A currency code or short suffix; blank or prose units (over 8
    /// characters) are dropped, as desktop's `chart-format.ts` ignores them.
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartKind {
    Bar,
    Line,
    Area,
    Pie,
    Radar,
    Radial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartVariant {
    Stacked,
    Horizontal,
    Donut,
}

/// `Series(name, values, total?)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenUiSeries {
    pub name: String,
    pub values: Vec<f64>,
    pub total: Option<String>,
}

/// `Stat(label, value, change?, note?)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenUiStat {
    pub label: String,
    pub value: String,
    pub change: Option<String>,
    pub note: Option<String>,
}

/// Longest unit the chart formats; longer ones are prose for the description.
const MAX_UNIT_CHARS: usize = 8;

/// Deepest reference chain followed; also stops reference cycles.
const MAX_DEPTH: usize = 32;

/// Parse a fence body into a block. `None` when there is nothing drawable
/// yet (still streaming) or the program is not for this library (a
/// finished block then falls back to its source).
pub fn parse_openui_block(source: &str) -> Option<OpenUiBlock> {
    let closed = auto_close(source);
    let tokens = tokenize(&closed);
    let statements = split_statements(&tokens);
    let mut symbols: HashMap<String, Value> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for (id, expr) in statements {
        let Some(value) = Parser::new(expr).parse_all() else {
            continue;
        };
        if !symbols.contains_key(&id) {
            order.push(id.clone());
        }
        symbols.insert(id, value);
    }
    let entry = if symbols.contains_key("root") {
        "root".to_string()
    } else {
        // lang-core's fallback: the first statement calling a root component.
        order.into_iter().find(|id| {
            matches!(&symbols[id], Value::Call { name, .. } if name == "Chart" || name == "Stats")
        })?
    };
    let root = Resolver { symbols: &symbols }.resolve(&symbols[&entry], 0)?;
    build_block(&root)
}

// ── Values ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
    Array(Vec<Value>),
    Call {
        name: String,
        args: Vec<Value>,
    },
    Ref(String),
    /// Something outside the static subset (an object literal).
    Other,
}

struct Resolver<'a> {
    symbols: &'a HashMap<String, Value>,
}

impl Resolver<'_> {
    /// Replace references with their values. An unresolved reference is
    /// `None`; arrays drop such entries, call arguments become `Null`.
    fn resolve(&self, value: &Value, depth: usize) -> Option<Value> {
        if depth > MAX_DEPTH {
            return None;
        }
        match value {
            Value::Ref(name) => self.resolve(self.symbols.get(name)?, depth + 1),
            Value::Array(items) => Some(Value::Array(
                items
                    .iter()
                    .filter_map(|item| self.resolve(item, depth + 1))
                    .collect(),
            )),
            Value::Call { name, args } => Some(Value::Call {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|arg| self.resolve(arg, depth + 1).unwrap_or(Value::Null))
                    .collect(),
            }),
            other => Some(other.clone()),
        }
    }
}

fn build_block(root: &Value) -> Option<OpenUiBlock> {
    let Value::Call { name, args } = root else {
        return None;
    };
    match name.as_str() {
        "Chart" => build_chart(args).map(OpenUiBlock::Chart),
        "Stats" => {
            let items: Vec<OpenUiStat> = calls_named(args.first(), "Stat")
                .filter_map(build_stat)
                .collect();
            (!items.is_empty()).then_some(OpenUiBlock::Stats(items))
        }
        _ => None,
    }
}

fn build_chart(args: &[Value]) -> Option<OpenUiChart> {
    let kind = match string_arg(args, 0)?.as_str() {
        "bar" => ChartKind::Bar,
        "line" => ChartKind::Line,
        "area" => ChartKind::Area,
        "pie" => ChartKind::Pie,
        "radar" => ChartKind::Radar,
        "radial" => ChartKind::Radial,
        _ => return None,
    };
    let labels = match args.get(2) {
        Some(Value::Array(items)) => items.iter().map(label_text).collect(),
        _ => Vec::new(),
    };
    let series = calls_named(args.get(3), "Series")
        .filter_map(build_series)
        .collect();
    let variant = string_arg(args, 5).and_then(|variant| match (variant.as_str(), kind) {
        ("stacked", ChartKind::Bar | ChartKind::Area) => Some(ChartVariant::Stacked),
        ("horizontal", ChartKind::Bar) => Some(ChartVariant::Horizontal),
        ("donut", ChartKind::Pie) => Some(ChartVariant::Donut),
        _ => None,
    });
    let unit = string_arg(args, 6)
        .map(|unit| unit.trim().to_string())
        .filter(|unit| !unit.is_empty() && unit.chars().count() <= MAX_UNIT_CHARS);
    Some(OpenUiChart {
        kind,
        title: string_arg(args, 1).unwrap_or_default(),
        labels,
        series,
        description: string_arg(args, 4).filter(|text| !text.trim().is_empty()),
        variant,
        unit,
    })
}

fn build_series(args: &[Value]) -> Option<OpenUiSeries> {
    let name = string_arg(args, 0)?;
    let Some(Value::Array(items)) = args.get(1) else {
        return None;
    };
    let values = items
        .iter()
        .filter_map(|item| match item {
            Value::Num(number) if number.is_finite() => Some(*number),
            _ => None,
        })
        .collect();
    Some(OpenUiSeries {
        name,
        values,
        total: string_arg(args, 2),
    })
}

fn build_stat(args: &[Value]) -> Option<OpenUiStat> {
    Some(OpenUiStat {
        label: string_arg(args, 0)?,
        value: string_arg(args, 1)?,
        change: string_arg(args, 2),
        note: string_arg(args, 3),
    })
}

/// The argument calls named `name`: a list of them, or a lone one (models
/// write `price` for `[price]`).
fn calls_named<'a>(
    value: Option<&'a Value>,
    name: &'a str,
) -> impl Iterator<Item = &'a [Value]> + 'a {
    let items: &'a [Value] = match value {
        Some(Value::Array(items)) => items,
        Some(single) => std::slice::from_ref(single),
        None => &[],
    };
    items.iter().filter_map(move |item| match item {
        Value::Call { name: call, args } if call == name => Some(args.as_slice()),
        _ => None,
    })
}

fn string_arg(args: &[Value], index: usize) -> Option<String> {
    match args.get(index) {
        Some(Value::Str(text)) => Some(text.clone()),
        _ => None,
    }
}

/// A label as desktop's `String(item ?? '')` writes it.
fn label_text(value: &Value) -> String {
    match value {
        Value::Str(text) => text.clone(),
        Value::Num(number) => format_number(*number),
        Value::Bool(flag) => flag.to_string(),
        _ => String::new(),
    }
}

fn format_number(number: f64) -> String {
    if number.fract() == 0.0 && number.abs() < 1e15 {
        format!("{}", number as i64)
    } else {
        format!("{number}")
    }
}

// ── Lexing ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Newline,
    LParen,
    RParen,
    LBrack,
    RBrack,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Equals,
    Minus,
    Str(String),
    Num(f64),
    /// Lowercase-initial name: a reference.
    Ident(String),
    /// Uppercase-initial name: a component.
    Type(String),
    True,
    False,
    Null,
    /// An operator or sigil outside the static subset.
    Other,
}

/// Close unclosed strings and brackets, as lang-core's `autoClose` does.
fn auto_close(input: &str) -> String {
    let mut stack: Vec<char> = Vec::new();
    let mut in_str: Option<char> = None;
    let mut escaped = false;
    for c in input.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if let Some(quote) = in_str {
            if c == '\\' {
                escaped = true;
            } else if c == quote {
                in_str = None;
            }
            continue;
        }
        match c {
            '"' | '\'' => in_str = Some(c),
            '(' | '[' | '{' => stack.push(c),
            ')' | ']' | '}' => {
                let open = match c {
                    ')' => '(',
                    ']' => '[',
                    _ => '{',
                };
                if stack.last() == Some(&open) {
                    stack.pop();
                }
            }
            _ => {}
        }
    }
    let mut out = input.to_string();
    if let Some(quote) = in_str {
        if escaped {
            out.push('\\');
        }
        out.push(quote);
    }
    for open in stack.iter().rev() {
        out.push(match open {
            '(' => ')',
            '[' => ']',
            _ => '}',
        });
    }
    out
}

fn tokenize(src: &str) -> Vec<Token> {
    let chars: Vec<char> = src.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\r' => i += 1,
            '\n' => {
                tokens.push(Token::Newline);
                i += 1;
            }
            '(' | ')' | '[' | ']' | '{' | '}' | ',' | ':' => {
                tokens.push(match c {
                    '(' => Token::LParen,
                    ')' => Token::RParen,
                    '[' => Token::LBrack,
                    ']' => Token::RBrack,
                    '{' => Token::LBrace,
                    '}' => Token::RBrace,
                    ',' => Token::Comma,
                    _ => Token::Colon,
                });
                i += 1;
            }
            '=' => {
                if chars.get(i + 1) == Some(&'=') {
                    tokens.push(Token::Other);
                    i += 2;
                } else {
                    tokens.push(Token::Equals);
                    i += 1;
                }
            }
            '"' | '\'' => {
                let (text, next) = read_string(&chars, i);
                tokens.push(Token::Str(text));
                i = next;
            }
            // A negative literal is `Minus Num` to the parser.
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '0'..='9' => {
                let start = i;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                if chars.get(i) == Some(&'.') && chars.get(i + 1).is_some_and(char::is_ascii_digit)
                {
                    i += 1;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if matches!(chars.get(i), Some('e' | 'E')) {
                    let mark = i;
                    i += 1;
                    if matches!(chars.get(i), Some('+' | '-')) {
                        i += 1;
                    }
                    if chars.get(i).is_some_and(char::is_ascii_digit) {
                        while i < chars.len() && chars[i].is_ascii_digit() {
                            i += 1;
                        }
                    } else {
                        i = mark;
                    }
                }
                let text: String = chars[start..i].iter().collect();
                tokens.push(text.parse().map_or(Token::Other, Token::Num));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                tokens.push(match word.as_str() {
                    "true" => Token::True,
                    "false" => Token::False,
                    "null" => Token::Null,
                    _ if c.is_ascii_uppercase() => Token::Type(word),
                    _ => Token::Ident(word),
                });
            }
            '$' | '@' | '!' | '<' | '>' | '&' | '|' | '.' | '?' | '+' | '*' | '/' | '%' => {
                tokens.push(Token::Other);
                i += 1;
            }
            // Anything else (`#`, emoji) is skipped, as in lang-core.
            _ => i += 1,
        }
    }
    tokens
}

/// Read a quoted string starting at `start`; returns the text and the index
/// after the closing quote. Escapes follow JSON (`\n`, `\t`, `\uXXXX`, …).
fn read_string(chars: &[char], start: usize) -> (String, usize) {
    let quote = chars[start];
    let mut out = String::new();
    let mut i = start + 1;
    while i < chars.len() {
        let c = chars[i];
        if c == quote {
            return (out, i + 1);
        }
        if c != '\\' {
            out.push(c);
            i += 1;
            continue;
        }
        let Some(&escaped) = chars.get(i + 1) else {
            break;
        };
        i += 2;
        match escaped {
            'n' => out.push('\n'),
            't' => out.push('\t'),
            'r' => out.push('\r'),
            'b' => out.push('\u{8}'),
            'f' => out.push('\u{c}'),
            'u' => {
                let hex: String = chars.iter().skip(i).take(4).collect();
                match u32::from_str_radix(&hex, 16)
                    .ok()
                    .filter(|_| hex.len() == 4)
                {
                    Some(code) => {
                        i += 4;
                        // A surrogate pair spells one scalar in two escapes.
                        if (0xD800..0xDC00).contains(&code)
                            && chars.get(i) == Some(&'\\')
                            && chars.get(i + 1) == Some(&'u')
                        {
                            let low: String = chars.iter().skip(i + 2).take(4).collect();
                            if let Ok(low) = u32::from_str_radix(&low, 16)
                                && (0xDC00..0xE000).contains(&low)
                            {
                                let scalar = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                out.extend(char::from_u32(scalar));
                                i += 6;
                                continue;
                            }
                        }
                        out.extend(char::from_u32(code));
                    }
                    None => out.push('u'),
                }
            }
            other => out.push(other),
        }
    }
    (out, chars.len())
}

/// Split tokens into `(id, expression tokens)` at depth-0 newlines, as
/// lang-core's `split` does. Lines without `name =` are skipped.
fn split_statements(tokens: &[Token]) -> Vec<(String, &[Token])> {
    let mut statements = Vec::new();
    let mut pos = 0;
    while pos < tokens.len() {
        while tokens.get(pos) == Some(&Token::Newline) {
            pos += 1;
        }
        if pos >= tokens.len() {
            break;
        }
        let id = match (&tokens[pos], tokens.get(pos + 1)) {
            (Token::Ident(id) | Token::Type(id), Some(Token::Equals)) => id.clone(),
            _ => {
                while pos < tokens.len() && tokens[pos] != Token::Newline {
                    pos += 1;
                }
                continue;
            }
        };
        pos += 2;
        let start = pos;
        let mut depth = 0usize;
        while pos < tokens.len() {
            match tokens[pos] {
                Token::Newline if depth == 0 => break,
                Token::LParen | Token::LBrack | Token::LBrace => depth += 1,
                Token::RParen | Token::RBrack | Token::RBrace => depth = depth.saturating_sub(1),
                _ => {}
            }
            pos += 1;
        }
        let expr = &tokens[start..pos];
        if !expr.is_empty() {
            statements.push((id, expr));
        }
    }
    statements
}

// ── Parsing ─────────────────────────────────────────────────────────────

struct Parser<'a> {
    tokens: Vec<&'a Token>,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        // Newlines inside brackets carry no meaning.
        let tokens = tokens.iter().filter(|t| **t != Token::Newline).collect();
        Self { tokens, pos: 0 }
    }

    /// The whole statement as one expression; trailing tokens (an operator
    /// expression) make the statement unusable.
    fn parse_all(mut self) -> Option<Value> {
        let value = self.expr(0)?;
        (self.pos == self.tokens.len()).then_some(value)
    }

    fn next(&mut self) -> Option<&'a Token> {
        let token = self.tokens.get(self.pos).copied();
        self.pos += 1;
        token
    }

    fn peek(&self) -> Option<&'a Token> {
        self.tokens.get(self.pos).copied()
    }

    fn expr(&mut self, depth: usize) -> Option<Value> {
        if depth > MAX_DEPTH {
            return None;
        }
        match self.next()? {
            Token::Str(text) => Some(Value::Str(text.clone())),
            Token::Num(number) => Some(Value::Num(*number)),
            Token::Minus => match self.next()? {
                Token::Num(number) => Some(Value::Num(-number)),
                _ => None,
            },
            Token::True => Some(Value::Bool(true)),
            Token::False => Some(Value::Bool(false)),
            Token::Null => Some(Value::Null),
            Token::Ident(name) => Some(Value::Ref(name.clone())),
            Token::Type(name) => {
                if self.peek() != Some(&Token::LParen) {
                    return Some(Value::Ref(name.clone()));
                }
                self.pos += 1;
                let args = self.list(&Token::RParen, depth)?;
                Some(Value::Call {
                    name: name.clone(),
                    args,
                })
            }
            Token::LBrack => self.list(&Token::RBrack, depth).map(Value::Array),
            Token::LBrace => {
                // Object literals are outside the subset; skip to the brace.
                let mut nested = 1usize;
                while nested > 0 {
                    match self.next()? {
                        Token::LBrace => nested += 1,
                        Token::RBrace => nested -= 1,
                        _ => {}
                    }
                }
                Some(Value::Other)
            }
            _ => None,
        }
    }

    /// Comma-separated expressions up to `close`; a trailing comma is fine.
    fn list(&mut self, close: &Token, depth: usize) -> Option<Vec<Value>> {
        let mut items = Vec::new();
        loop {
            if self.peek() == Some(close) {
                self.pos += 1;
                return Some(items);
            }
            items.push(self.expr(depth + 1)?);
            match self.next()? {
                Token::Comma => {}
                token if token == close => return Some(items),
                _ => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chart(source: &str) -> OpenUiChart {
        match parse_openui_block(source) {
            Some(OpenUiBlock::Chart(chart)) => chart,
            other => panic!("expected a chart, got {other:?}"),
        }
    }

    #[test]
    fn prompt_examples_parse() {
        let prompt = crate::visual_output::OPENUI_PROMPT;
        let mut found = 0;
        for block in prompt.split("```openui-lang\n").skip(1) {
            let body = block.split("```").next().unwrap();
            assert!(parse_openui_block(body).is_some(), "example failed: {body}");
            found += 1;
        }
        assert_eq!(found, 3);
    }

    #[test]
    fn grouped_bar_chart_resolves_references() {
        let c = chart(
            r#"root = Chart("bar", "Quarterly revenue", quarters, [north, south], "USD thousands")
quarters = ["Q1", "Q2", "Q3", "Q4"]
north = Series("North", [120, 135, 150, 170])
south = Series("South", [90, 110, 105, 130])"#,
        );
        assert_eq!(c.kind, ChartKind::Bar);
        assert_eq!(c.title, "Quarterly revenue");
        assert_eq!(c.labels, ["Q1", "Q2", "Q3", "Q4"]);
        assert_eq!(c.series.len(), 2);
        assert_eq!(c.series[1].values, [90.0, 110.0, 105.0, 130.0]);
        assert_eq!(c.description.as_deref(), Some("USD thousands"));
        assert_eq!(c.variant, None);
        assert_eq!(c.unit, None);
    }

    #[test]
    fn stats_parse_with_optional_fields() {
        let block = parse_openui_block(
            r#"root = Stats([revenue, churn])
revenue = Stat("Revenue", "$48.2k", "+8.1%", "vs last month")
churn = Stat("Churn", "1.8%")"#,
        );
        let Some(OpenUiBlock::Stats(items)) = block else {
            panic!("expected stats");
        };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].change.as_deref(), Some("+8.1%"));
        assert_eq!(items[1].change, None);
        assert_eq!(items[1].note, None);
    }

    #[test]
    fn lone_series_without_brackets_is_accepted() {
        let c = chart(
            r#"root = Chart("line", "ACME", days, close, null, null, "USD")
days = ["2026-05-11", "2026-05-12"]
close = Series("ACME", [41.2, -0.5])"#,
        );
        assert_eq!(c.series.len(), 1);
        assert_eq!(c.series[0].values, [41.2, -0.5]);
        assert_eq!(c.description, None);
        assert_eq!(c.unit.as_deref(), Some("USD"));
    }

    #[test]
    fn streaming_prefix_keeps_what_is_complete() {
        let source = r#"root = Chart("bar", "Sales", ["A", "B"], [a, b])
a = Series("A", [1, 2])
b = Series("B", [3"#;
        let c = chart(source);
        assert_eq!(c.series.len(), 2);
        assert_eq!(c.series[1].values, [3.0]);

        // Before the series are written, the card shell already renders.
        let shell = chart(r#"root = Chart("bar", "Sal"#);
        assert_eq!(shell.title, "Sal");
        assert!(shell.series.is_empty());

        // A cut-off type string names no chart type yet.
        assert_eq!(parse_openui_block(r#"root = Chart("ba"#), None);
    }

    #[test]
    fn other_libraries_and_dynamic_programs_do_not_parse() {
        assert_eq!(parse_openui_block(r#"root = Card([Text("hi")])"#), None);
        assert_eq!(parse_openui_block("root = $count + 1"), None);
        assert_eq!(parse_openui_block(""), None);
        assert_eq!(parse_openui_block(r#"root = Stats([])"#), None);
    }

    #[test]
    fn root_falls_back_to_first_chart_statement() {
        let c = chart(
            r#"sales = Chart("pie", "Share", ["a", "b"], [s], null, "donut")
s = Series("S", [1, 2])"#,
        );
        assert_eq!(c.kind, ChartKind::Pie);
        assert_eq!(c.variant, Some(ChartVariant::Donut));
    }

    #[test]
    fn variants_invalid_for_the_type_and_prose_units_are_dropped() {
        let c = chart(r#"root = Chart("line", "T", [], [], null, "donut", "per active user")"#);
        assert_eq!(c.variant, None);
        assert_eq!(c.unit, None);
        let c = chart(r#"root = Chart("bar", "T", [], [], null, "horizontal", " % ")"#);
        assert_eq!(c.variant, Some(ChartVariant::Horizontal));
        assert_eq!(c.unit.as_deref(), Some("%"));
    }

    #[test]
    fn reference_cycles_and_unknown_names_resolve_to_nothing() {
        let c = chart(
            r#"root = Chart("bar", "T", labels, [a, missing])
labels = labels
a = Series("A", [1])"#,
        );
        assert!(c.labels.is_empty());
        assert_eq!(c.series.len(), 1);
    }

    #[test]
    fn labels_strings_escapes_and_numbers() {
        let c = chart(
            "root = Chart('radar', \"Caf\\u00e9 \\\"best\\\"\", [2024, 2025.5, \"x\\ny\"], [])",
        );
        assert_eq!(c.kind, ChartKind::Radar);
        assert_eq!(c.title, "Caf\u{e9} \"best\"");
        assert_eq!(c.labels, ["2024", "2025.5", "x\ny"]);
    }

    #[test]
    fn surrogate_pair_escapes_spell_one_character() {
        let c = chart(r#"root = Chart("bar", "Up \ud83d\udcc8", [], [])"#);
        assert_eq!(c.title, "Up \u{1f4c8}");
    }

    #[test]
    fn multiline_calls_and_trailing_commas() {
        let c = chart(
            "root = Chart(\n  \"area\",\n  \"T\",\n  [\"a\", \"b\",],\n  [s],\n  null,\n  \"stacked\",\n)\ns = Series(\"S\", [1e3, 2.5E-1])",
        );
        assert_eq!(c.variant, Some(ChartVariant::Stacked));
        assert_eq!(c.labels, ["a", "b"]);
        assert_eq!(c.series[0].values, [1000.0, 0.25]);
    }
}
