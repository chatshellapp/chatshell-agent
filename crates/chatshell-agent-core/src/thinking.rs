use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

static THINKING_TAGS: Lazy<Vec<Regex>> = Lazy::new(|| {
    ["think", "thinking", "reasoning"]
        .into_iter()
        .map(|tag| Regex::new(&format!(r"(?is)<{tag}>(.*?)</{tag}>")).unwrap())
        .collect()
});

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedContent {
    pub content: String,
    pub thinking_content: Option<String>,
}

pub fn parse_thinking_content(text: &str) -> ParsedContent {
    let mut content = text.to_string();
    let mut thinking = Vec::new();

    for regex in THINKING_TAGS.iter() {
        thinking.extend(
            regex
                .captures_iter(text)
                .filter_map(|capture| capture.get(1))
                .map(|value| value.as_str().trim().to_string())
                .filter(|value| !value.is_empty()),
        );
        content = regex.replace_all(&content, "").to_string();
    }

    ParsedContent {
        content: content.trim().to_string(),
        thinking_content: (!thinking.is_empty()).then(|| thinking.join("\n\n")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_reasoning_and_visible_content() {
        let parsed = parse_thinking_content("<think>Check first</think>\n\nAnswer");
        assert_eq!(parsed.content, "Answer");
        assert_eq!(parsed.thinking_content.as_deref(), Some("Check first"));
    }
}
