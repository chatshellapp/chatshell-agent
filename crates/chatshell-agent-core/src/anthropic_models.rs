//! Request defaults for Claude models on the Anthropic Messages wire.
//!
//! Two defaults depend on the model generation, so they are derived from
//! the model id instead of being one constant:
//!
//! - **Output bound.** Anthropic requires `max_tokens`. A flat 8192 cuts
//!   current models off mid-answer (adaptive thinking spends part of the
//!   bound before any text), so models with large output windows get a
//!   larger default.
//! - **Visible thinking.** On Claude Opus 4.7 and later, Sonnet 5 and later,
//!   Haiku 5 and later, and the Fable / Mythos lines, thinking blocks stream
//!   with empty text unless the request sets `thinking.display:
//!   "summarized"`. Without it the UI shows no reasoning at all even though
//!   the model thought (and was billed for it).
//!
//! Model ids arrive in several spellings: first-party (`claude-sonnet-5-5`),
//! gateway-prefixed with dots (`anthropic/claude-sonnet-5.5`), dated
//! (`claude-opus-4-1-20250805`), legacy order (`claude-3-5-sonnet-latest`),
//! Bedrock (`us.anthropic.claude-sonnet-4-5-v1:0`), and Vertex
//! (`claude-opus-4-5@20251101`).

use serde_json::{Value, json};

/// Default `max_tokens` for models that are not recognized as Claude, or
/// for older Claude models with small output windows.
pub const ANTHROPIC_FALLBACK_MAX_TOKENS: u64 = 8192;

/// Default `max_tokens` for current Claude models. Responses are streamed,
/// so a large bound does not risk HTTP timeouts; it only stops long
/// answers (and the thinking before them) from being cut off.
pub const ANTHROPIC_LARGE_MAX_TOKENS: u64 = 64_000;

/// Default `max_tokens` for Claude Opus 4 / 4.1 (32K output window).
const OPUS_4_MAX_TOKENS: u64 = 32_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Opus,
    Sonnet,
    Haiku,
    Fable,
    Mythos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClaudeModel {
    family: Family,
    version: (u32, u32),
}

impl ClaudeModel {
    fn at_least(&self, major: u32, minor: u32) -> bool {
        self.version >= (major, minor)
    }

    /// Accepts `thinking: {type: "adaptive"}`.
    fn supports_adaptive_thinking(&self) -> bool {
        match self.family {
            Family::Fable | Family::Mythos => true,
            Family::Opus | Family::Sonnet => self.at_least(4, 6),
            Family::Haiku => self.at_least(5, 0),
        }
    }

    /// Hides thinking text unless `display: "summarized"` is requested.
    /// (Opus 4.6 and Sonnet 4.6 summarize by default.)
    fn omits_thinking_by_default(&self) -> bool {
        match self.family {
            Family::Fable | Family::Mythos => true,
            Family::Opus => self.at_least(4, 7),
            Family::Sonnet | Family::Haiku => self.at_least(5, 0),
        }
    }

    fn default_max_tokens(&self) -> u64 {
        let large = match self.family {
            Family::Fable | Family::Mythos => true,
            Family::Opus => self.at_least(4, 5),
            Family::Sonnet => self.at_least(3, 7),
            Family::Haiku => self.at_least(4, 5),
        };
        if large {
            ANTHROPIC_LARGE_MAX_TOKENS
        } else if self.family == Family::Opus && self.at_least(4, 0) {
            OPUS_4_MAX_TOKENS
        } else {
            ANTHROPIC_FALLBACK_MAX_TOKENS
        }
    }
}

/// Parse a Claude model id in any of the spellings listed in the module
/// docs. `None` for anything that is not recognizably a Claude model.
fn parse_claude_model(model_id: &str) -> Option<ClaudeModel> {
    let lower = model_id.to_ascii_lowercase();
    let start = lower.rfind("claude")?;
    // Vertex (`@20251101`) and Bedrock (`:0`) suffixes carry no version.
    let name = lower[start..].split(['@', ':']).next()?;

    let mut family = None;
    let mut numbers = Vec::new();
    for token in name.split(['-', '.', '_']) {
        match token {
            "opus" => family = Some(Family::Opus),
            "sonnet" => family = Some(Family::Sonnet),
            "haiku" => family = Some(Family::Haiku),
            "fable" => family = Some(Family::Fable),
            "mythos" => family = Some(Family::Mythos),
            // Version parts are short; 8-digit snapshot dates are not.
            _ if !token.is_empty() && token.len() <= 2 => {
                if let Ok(n) = token.parse::<u32>() {
                    numbers.push(n);
                }
            }
            _ => {}
        }
    }

    let family = family?;
    let major = *numbers.first()?;
    let minor = numbers.get(1).copied().unwrap_or(0);
    Some(ClaudeModel {
        family,
        version: (major, minor),
    })
}

/// The `max_tokens` to send when the caller set none.
pub fn anthropic_default_max_tokens(model_id: &str) -> u64 {
    parse_claude_model(model_id)
        .map(|model| model.default_max_tokens())
        .unwrap_or(ANTHROPIC_FALLBACK_MAX_TOKENS)
}

/// The `thinking` field to send when the caller configured none, so the
/// model's reasoning comes back readable. `None` when the model takes no
/// adaptive thinking, or when a custom temperature is set (thinking
/// requests reject non-default sampling).
pub fn anthropic_default_thinking(model_id: &str, temperature: Option<f64>) -> Option<Value> {
    if temperature.is_some() {
        return None;
    }
    let model = parse_claude_model(model_id)?;
    if !model.supports_adaptive_thinking() {
        return None;
    }
    Some(if model.omits_thinking_by_default() {
        json!({ "type": "adaptive", "display": "summarized" })
    } else {
        json!({ "type": "adaptive" })
    })
}

/// `additional` with the default `thinking` field added, unless the caller
/// already set `thinking` (their choice wins) or the model gets no default.
/// Returns `additional` unchanged otherwise.
pub fn with_anthropic_thinking_default(
    model_id: &str,
    temperature: Option<f64>,
    additional: Option<Value>,
) -> Option<Value> {
    if additional
        .as_ref()
        .is_some_and(|params| params.get("thinking").is_some())
    {
        return additional;
    }
    let Some(thinking) = anthropic_default_thinking(model_id, temperature) else {
        return additional;
    };
    match additional {
        Some(Value::Object(mut map)) => {
            map.insert("thinking".to_string(), thinking);
            Some(Value::Object(map))
        }
        // A non-object payload is the caller's to keep as-is.
        Some(other) => Some(other),
        None => Some(json!({ "thinking": thinking })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(id: &str) -> Option<(Family, (u32, u32))> {
        parse_claude_model(id).map(|m| (m.family, m.version))
    }

    #[test]
    fn parses_every_model_id_spelling() {
        assert_eq!(parsed("claude-sonnet-5-5"), Some((Family::Sonnet, (5, 5))));
        assert_eq!(
            parsed("anthropic/claude-sonnet-5.5"),
            Some((Family::Sonnet, (5, 5)))
        );
        assert_eq!(
            parsed("claude-opus-4-1-20250805"),
            Some((Family::Opus, (4, 1)))
        );
        assert_eq!(
            parsed("claude-3-5-sonnet-latest"),
            Some((Family::Sonnet, (3, 5)))
        );
        assert_eq!(
            parsed("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            Some((Family::Sonnet, (4, 5)))
        );
        assert_eq!(
            parsed("claude-opus-4-5@20251101"),
            Some((Family::Opus, (4, 5)))
        );
        assert_eq!(parsed("claude-opus-5"), Some((Family::Opus, (5, 0))));
        assert_eq!(parsed("claude-fable-5-1"), Some((Family::Fable, (5, 1))));
        assert_eq!(
            parsed("anthropic/claude-haiku-5.5"),
            Some((Family::Haiku, (5, 5)))
        );
        assert_eq!(parsed("gpt-6-luna"), None);
        assert_eq!(parsed("claude-unknown"), None);
    }

    #[test]
    fn max_tokens_defaults_follow_the_output_window() {
        assert_eq!(anthropic_default_max_tokens("claude-sonnet-5-5"), 64_000);
        assert_eq!(
            anthropic_default_max_tokens("anthropic/claude-opus-5.5"),
            64_000
        );
        assert_eq!(anthropic_default_max_tokens("claude-haiku-4-5"), 64_000);
        assert_eq!(
            anthropic_default_max_tokens("claude-3-7-sonnet-latest"),
            64_000
        );
        assert_eq!(anthropic_default_max_tokens("claude-opus-4-1"), 32_000);
        assert_eq!(
            anthropic_default_max_tokens("claude-3-5-haiku-latest"),
            8192
        );
        assert_eq!(anthropic_default_max_tokens("some-gateway-model"), 8192);
    }

    #[test]
    fn current_models_get_summarized_adaptive_thinking() {
        let summarized = json!({ "type": "adaptive", "display": "summarized" });
        for id in [
            "claude-sonnet-5-5",
            "anthropic/claude-sonnet-5.5",
            "anthropic/claude-haiku-5.5",
            "claude-opus-4-7",
            "claude-opus-5-5",
            "claude-fable-5-1",
        ] {
            assert_eq!(
                anthropic_default_thinking(id, None),
                Some(summarized.clone()),
                "{id}"
            );
        }
        // 4.6 summarizes by default and may not know the display field.
        assert_eq!(
            anthropic_default_thinking("claude-sonnet-4-6", None),
            Some(json!({ "type": "adaptive" }))
        );
    }

    #[test]
    fn older_models_and_custom_temperature_get_no_thinking_default() {
        assert_eq!(anthropic_default_thinking("claude-haiku-4-5", None), None);
        assert_eq!(anthropic_default_thinking("claude-sonnet-4-5", None), None);
        assert_eq!(anthropic_default_thinking("claude-3-7-sonnet", None), None);
        assert_eq!(anthropic_default_thinking("not-claude", None), None);
        assert_eq!(
            anthropic_default_thinking("claude-sonnet-5-5", Some(0.7)),
            None
        );
    }

    #[test]
    fn caller_thinking_wins_and_other_params_are_kept() {
        let own = Some(json!({ "thinking": { "type": "between_tools" } }));
        assert_eq!(
            with_anthropic_thinking_default("claude-sonnet-5-5", None, own.clone()),
            own
        );
        assert_eq!(
            with_anthropic_thinking_default(
                "claude-sonnet-5-5",
                None,
                Some(json!({ "metadata": { "user_id": "u" } }))
            ),
            Some(json!({
                "metadata": { "user_id": "u" },
                "thinking": { "type": "adaptive", "display": "summarized" }
            }))
        );
        assert_eq!(
            with_anthropic_thinking_default("claude-sonnet-5-5", None, None),
            Some(json!({ "thinking": { "type": "adaptive", "display": "summarized" } }))
        );
        assert_eq!(
            with_anthropic_thinking_default("claude-haiku-4-5", None, None),
            None
        );
    }
}
