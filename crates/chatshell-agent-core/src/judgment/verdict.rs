//! The judge's two-axis output contract and its tolerant parser (AR-7).
//!
//! `{risk_level, user_authorization, outcome, rationale}`. Only `outcome`
//! gates; the other fields feed markers and the decision log. Parsing is
//! tolerant of prose and code fences around the object, but fail-safe:
//! a missing or invalid `outcome` is a parse failure (a review failure,
//! never an allow), and missing or invalid secondary fields take their
//! deny-side defaults. One contradiction is resolved deterministically:
//! `allow` with `critical` risk becomes `deny` (the matrix never allows
//! critical risk).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Longest rationale kept (characters).
pub const MAX_RATIONALE_CHARS: usize = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Authorization {
    Unknown,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Allow,
    Deny,
}

/// A parsed judge verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Verdict {
    pub risk_level: RiskLevel,
    pub user_authorization: Authorization,
    pub outcome: Outcome,
    pub rationale: String,
}

/// Why a response could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("the reviewer returned no JSON object")]
    NoObject,
    #[error("the reviewer returned no valid outcome")]
    NoOutcome,
}

/// Parse a judge response.
pub fn parse_verdict(text: &str) -> Result<Verdict, ParseError> {
    let object = find_object(text).ok_or(ParseError::NoObject)?;
    let field = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_str)
            .map(|s| s.trim().to_ascii_lowercase())
    };
    let outcome = match field("outcome").as_deref() {
        Some("allow") => Outcome::Allow,
        Some("deny" | "block") => Outcome::Deny,
        _ => return Err(ParseError::NoOutcome),
    };
    let risk_level = match field("risk_level").as_deref() {
        Some("low") => RiskLevel::Low,
        Some("medium") => RiskLevel::Medium,
        Some("high") => RiskLevel::High,
        Some("critical") => RiskLevel::Critical,
        // Deny side: an allow without a stated risk is still recorded as
        // high risk, never as low.
        _ => RiskLevel::High,
    };
    let user_authorization = match field("user_authorization").as_deref() {
        Some("low") => Authorization::Low,
        Some("medium") => Authorization::Medium,
        Some("high") => Authorization::High,
        _ => Authorization::Unknown,
    };
    let rationale: String = object
        .get("rationale")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(MAX_RATIONALE_CHARS)
        .collect();
    let outcome = if risk_level == RiskLevel::Critical {
        Outcome::Deny
    } else {
        outcome
    };
    Ok(Verdict {
        risk_level,
        user_authorization,
        outcome,
        rationale,
    })
}

/// The first JSON object in `text` that parses: the whole text, else
/// each balanced `{...}` span in order (string-aware), so prose and code
/// fences around the object are tolerated.
fn find_object(text: &str) -> Option<serde_json::Map<String, Value>> {
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(text.trim()) {
        return Some(map);
    }
    let bytes = text.as_bytes();
    let mut start = 0;
    while let Some(offset) = text[start..].find('{') {
        let open = start + offset;
        if let Some(close) = balanced_end(bytes, open)
            && let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&text[open..=close])
        {
            return Some(map);
        }
        start = open + 1;
    }
    None
}

/// Index of the `}` closing the `{` at `open`, honoring JSON strings.
fn balanced_end(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, &byte) in bytes.iter().enumerate().skip(open) {
        if in_string {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid() {
        let verdict = parse_verdict(
            r#"{"risk_level":"medium","user_authorization":"high","outcome":"allow","rationale":"User asked to fetch the docs."}"#,
        )
        .unwrap();
        assert_eq!(
            verdict,
            Verdict {
                risk_level: RiskLevel::Medium,
                user_authorization: Authorization::High,
                outcome: Outcome::Allow,
                rationale: "User asked to fetch the docs.".to_string(),
            }
        );
    }

    #[test]
    fn prose_and_fence_wrapped() {
        let text = "Sure. Here is my assessment:\n```json\n{\"risk_level\": \"High\", \
                    \"user_authorization\": \"low\", \"outcome\": \"Deny\", \
                    \"rationale\": \"Sends {secrets} away\"}\n```\nThanks!";
        let verdict = parse_verdict(text).unwrap();
        assert_eq!(verdict.outcome, Outcome::Deny);
        assert_eq!(verdict.risk_level, RiskLevel::High);
        assert_eq!(verdict.rationale, "Sends {secrets} away");
        // A stray brace before the object does not derail the search.
        let verdict =
            parse_verdict("a { b\n{\"outcome\":\"allow\",\"risk_level\":\"low\"}").unwrap();
        assert_eq!(verdict.outcome, Outcome::Allow);
    }

    #[test]
    fn missing_fields_take_deny_side_defaults() {
        let verdict = parse_verdict(r#"{"outcome":"allow"}"#).unwrap();
        assert_eq!(verdict.outcome, Outcome::Allow);
        assert_eq!(verdict.risk_level, RiskLevel::High);
        assert_eq!(verdict.user_authorization, Authorization::Unknown);
        assert_eq!(verdict.rationale, "");
        let verdict =
            parse_verdict(r#"{"outcome":"allow","risk_level":"extreme","user_authorization":7}"#)
                .unwrap();
        assert_eq!(verdict.risk_level, RiskLevel::High);
        assert_eq!(verdict.user_authorization, Authorization::Unknown);
    }

    #[test]
    fn missing_or_invalid_outcome_is_a_failure() {
        assert_eq!(
            parse_verdict(r#"{"risk_level":"low"}"#),
            Err(ParseError::NoOutcome)
        );
        assert_eq!(
            parse_verdict(r#"{"outcome":"maybe"}"#),
            Err(ParseError::NoOutcome)
        );
    }

    #[test]
    fn garbage_is_a_failure() {
        for text in ["", "allow", "I think this is fine.", "{not json}", "[1,2]"] {
            assert!(parse_verdict(text).is_err(), "{text:?}");
        }
    }

    #[test]
    fn critical_risk_never_allows() {
        let verdict = parse_verdict(
            r#"{"outcome":"allow","risk_level":"critical","user_authorization":"high"}"#,
        )
        .unwrap();
        assert_eq!(verdict.outcome, Outcome::Deny);
    }

    #[test]
    fn rationale_is_bounded_and_single_line() {
        let long = "word\n".repeat(400);
        let text = serde_json::json!({ "outcome": "deny", "rationale": long }).to_string();
        let verdict = parse_verdict(&text).unwrap();
        assert!(verdict.rationale.chars().count() <= MAX_RATIONALE_CHARS);
        assert!(!verdict.rationale.contains('\n'));
    }
}
