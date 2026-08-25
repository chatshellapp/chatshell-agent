use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

static THINKING_TAGS: &[&str] = &["think", "thinking", "reasoning"];

static COMPLETED_TAG_REGEXES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    THINKING_TAGS
        .iter()
        .map(|tag| Regex::new(&format!(r"(?is)<{tag}>(.*?)</{tag}>")).unwrap())
        .collect()
});

static OPEN_TAG_REGEXES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    THINKING_TAGS
        .iter()
        .map(|tag| Regex::new(&format!(r"(?i)<{tag}>")).unwrap())
        .collect()
});

/// Output of [`parse_thinking_content`]: the visible content with thinking
/// blocks stripped, the extracted thinking text, and whether an unclosed
/// thinking tag was found (indicating the model is still producing reasoning
/// during streaming).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedContent {
    pub content: String,
    pub thinking_content: Option<String>,
    /// `true` when the text contains an unclosed `<think>` (or similar) tag,
    /// meaning the model is mid-reasoning and the trailing text is the
    /// in-progress thinking content. Desktop ignores this during final
    /// persistence; iOS uses it to keep the thinking indicator live.
    #[serde(default)]
    pub is_thinking_in_progress: bool,
}

/// Extract `<think>`, `<thinking>`, and `<reasoning>` blocks from model output.
///
/// Completed blocks (`<tag>...</tag>`) are extracted into `thinking_content`
/// and removed from `content`. If an unclosed `<tag>` is found (common during
/// streaming), the trailing text after the open tag is treated as in-progress
/// thinking, `is_thinking_in_progress` is set to `true`, and the open tag plus
/// trailing text is removed from `content`.
pub fn parse_thinking_content(text: &str) -> ParsedContent {
    let mut content = text.to_string();
    let mut thinking = Vec::new();

    // Phase 1: extract completed <tag>...</tag> blocks.
    for regex in COMPLETED_TAG_REGEXES.iter() {
        thinking.extend(
            regex
                .captures_iter(text)
                .filter_map(|capture| capture.get(1))
                .map(|value| value.as_str().trim().to_string())
                .filter(|value| !value.is_empty()),
        );
        content = regex.replace_all(&content, "").to_string();
    }

    // Phase 2: detect unclosed <tag> (streaming in progress). After removing
    // completed blocks, any remaining <tag> is unclosed. The text after it
    // is in-progress reasoning that hasn't been closed yet.
    let mut is_thinking_in_progress = false;
    for regex in OPEN_TAG_REGEXES.iter() {
        if let Some(mat) = regex.find(&content) {
            let trailing = content[mat.end()..].trim();
            if !trailing.is_empty() {
                thinking.push(trailing.to_string());
            }
            content = content[..mat.start()].to_string();
            is_thinking_in_progress = true;
        }
    }

    ParsedContent {
        content: content.trim().to_string(),
        thinking_content: (!thinking.is_empty()).then(|| thinking.join("\n\n")),
        is_thinking_in_progress,
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
        assert!(!parsed.is_thinking_in_progress);
    }

    #[test]
    fn detects_unclosed_tag_during_streaming() {
        let parsed = parse_thinking_content("<think>Working on it...");
        assert_eq!(parsed.content, "");
        assert_eq!(parsed.thinking_content.as_deref(), Some("Working on it..."));
        assert!(parsed.is_thinking_in_progress);
    }

    #[test]
    fn closed_then_open_tag() {
        let parsed = parse_thinking_content("<think>Done</think>Answer<think>More thinking");
        assert_eq!(parsed.content, "Answer");
        assert_eq!(
            parsed.thinking_content.as_deref(),
            Some("Done\n\nMore thinking")
        );
        assert!(parsed.is_thinking_in_progress);
    }

    #[test]
    fn no_thinking_tags() {
        let text = "Just a regular response.";
        let parsed = parse_thinking_content(text);
        assert_eq!(parsed.content, text);
        assert_eq!(parsed.thinking_content, None);
        assert!(!parsed.is_thinking_in_progress);
    }
}
