//! Shared CJK-aware tokenizer for local FTS message search.
//!
//! Desktop and iOS each build a local `messages_fts` FTS5 table whose
//! `content` column holds jieba-segmented, space-joined text; queries are
//! segmented the same way so FTS `MATCH` lines up with indexed content.
//! The tokenizer lives here so both platforms segment identically.

use jieba_rs::Jieba;
use std::sync::LazyLock;

static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);

/// Tokenize text for indexing in FTS. Uses search mode, which emits the
/// dictionary words inside each segment in addition to the segments
/// themselves, so CJK text without word boundaries still matches on
/// substrings: "深度学习框架" indexes 深度/学习/框架 alongside the whole
/// run. Japanese and Korean input ("機械学習", "기계 학습") flows through
/// the same segmentation path.
pub fn tokenize_for_search(text: &str) -> String {
    JIEBA
        .cut_for_search(text, true)
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Tokenize a search query the same way indexed content was tokenized so
/// `MATCH` terms exist in the index.
pub fn tokenize_query(query: &str) -> String {
    tokenize_for_search(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_search_mode_emits_overlapping_segments() {
        let tokens = tokenize_for_search("深度学习框架");
        assert!(tokens.contains("深度"));
        assert!(tokens.contains("学习"));
        assert!(tokens.contains("框架"));

        // Jieba may split Japanese and Korean words into individual characters;
        // verify that segmentation preserves the full text in both scripts.
        for text in ["ディープラーニング フレームワーク", "딥러닝 프레임워크"]
        {
            let tokens = tokenize_for_search(text);
            assert_eq!(
                tokens.split_whitespace().collect::<String>(),
                text.split_whitespace().collect::<String>(),
                "segmentation must preserve text: {text}"
            );
        }
    }

    #[test]
    fn ascii_text_passes_through_as_words() {
        assert_eq!(tokenize_for_search("hello world"), "hello world");
    }

    #[test]
    fn query_and_content_tokenizers_agree() {
        for text in ["机器学习", "機械学習", "기계 학습"] {
            assert_eq!(
                tokenize_query(text),
                tokenize_for_search(text),
                "query and content tokenization must agree: {text}"
            );
        }
    }
}
