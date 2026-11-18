//! `ask_user` for client-side tool callers (iOS, Android).
//!
//! The clients run tools themselves, so they advertise
//! [`ask_user_definition`], validate each call with
//! [`parse_ask_user_request`], show the questions, and turn what the user
//! did into the tool result with [`format_ask_user_outcome`]. The contract
//! (schema, validation, result text) is the core's, identical to desktop.

use chatshell_agent_core as core;
use core::tools::ask_user as engine;

use crate::ClientToolDefinition;

/// One option. `id` is runtime-assigned (`o1`, `o2`, ...).
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct AskUserOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub recommended: bool,
}

/// One question. No options means a free-text question.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct AskUserQuestion {
    pub id: String,
    pub question: String,
    pub header: Option<String>,
    pub multi: bool,
    pub options: Vec<AskUserOption>,
}

/// A validated `ask_user` call, ready to show.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct AskUserRequest {
    pub questions: Vec<AskUserQuestion>,
}

/// What the user entered for one question: selected option ids, and typed
/// text (the answer when nothing is selected, otherwise a note).
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct AskUserAnswer {
    pub question_id: String,
    pub selected: Vec<String>,
    pub text: Option<String>,
}

/// How a request ended.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Enum)]
pub enum AskUserOutcome {
    /// Submitted; questions without an entry count as skipped.
    Answered { answers: Vec<AskUserAnswer> },
    /// The user wants to discuss in chat; answers given so far are kept.
    Discuss { answers: Vec<AskUserAnswer> },
    /// Dismissed without answering.
    Declined,
    /// Closed without the user (the turn was stopped).
    Cancelled,
}

/// Result of validating the model's arguments.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Enum)]
pub enum AskUserParse {
    Valid {
        request: AskUserRequest,
    },
    /// Return `message` to the model as the tool result; show nothing.
    Invalid {
        message: String,
    },
}

/// Definition of the `ask_user` tool; advertise it only when a person can
/// answer.
#[uniffi::export]
pub fn ask_user_definition() -> ClientToolDefinition {
    core::tool_definitions::ask_user_definition().into()
}

/// Validate and normalize an `ask_user` call's arguments.
#[uniffi::export]
pub fn parse_ask_user_request(arguments_json: String) -> AskUserParse {
    let args = match serde_json::from_str::<serde_json::Value>(&arguments_json) {
        Ok(args) => args,
        Err(error) => {
            return AskUserParse::Invalid {
                message: format!("ask_user arguments are not valid JSON: {error}"),
            };
        }
    };
    match engine::parse_request(&args) {
        Ok(request) => AskUserParse::Valid {
            request: request.into(),
        },
        Err(message) => AskUserParse::Invalid { message },
    }
}

/// The tool-result text for `outcome`.
#[uniffi::export]
pub fn format_ask_user_outcome(request: AskUserRequest, outcome: AskUserOutcome) -> String {
    engine::format_outcome(&request.into(), &outcome.into())
}

impl From<engine::AskUserRequest> for AskUserRequest {
    fn from(value: engine::AskUserRequest) -> Self {
        Self {
            questions: value
                .questions
                .into_iter()
                .map(|q| AskUserQuestion {
                    id: q.id,
                    question: q.question,
                    header: q.header,
                    multi: q.multi,
                    options: q
                        .options
                        .into_iter()
                        .map(|o| AskUserOption {
                            id: o.id,
                            label: o.label,
                            description: o.description,
                            recommended: o.recommended,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl From<AskUserRequest> for engine::AskUserRequest {
    fn from(value: AskUserRequest) -> Self {
        Self {
            questions: value
                .questions
                .into_iter()
                .map(|q| engine::AskUserQuestion {
                    id: q.id,
                    question: q.question,
                    header: q.header,
                    multi: q.multi,
                    options: q
                        .options
                        .into_iter()
                        .map(|o| engine::AskUserOption {
                            id: o.id,
                            label: o.label,
                            description: o.description,
                            recommended: o.recommended,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

fn answers(answers: Vec<AskUserAnswer>) -> Vec<engine::AskUserAnswer> {
    answers
        .into_iter()
        .map(|a| engine::AskUserAnswer {
            question_id: a.question_id,
            selected: a.selected,
            text: a.text,
        })
        .collect()
}

impl From<AskUserOutcome> for engine::AskUserOutcome {
    fn from(value: AskUserOutcome) -> Self {
        match value {
            AskUserOutcome::Answered { answers: a } => Self::Answered(answers(a)),
            AskUserOutcome::Discuss { answers: a } => Self::Discuss(answers(a)),
            AskUserOutcome::Declined => Self::Declined,
            AskUserOutcome::Cancelled => Self::Cancelled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid(json: &str) -> AskUserRequest {
        match parse_ask_user_request(json.to_string()) {
            AskUserParse::Valid { request } => request,
            AskUserParse::Invalid { message } => panic!("invalid: {message}"),
        }
    }

    #[test]
    fn definition_matches_the_core_tool() {
        let definition = ask_user_definition();
        assert_eq!(definition.name, "ask_user");
        assert!(definition.parameters_json.contains("\"questions\""));
    }

    #[test]
    fn parse_round_trips_through_format() {
        let request = valid(
            r#"{"questions":[{"id":"db","question":"Which?","options":[
                {"label":"SQLite"},{"label":"PostgreSQL","recommended":true}]}]}"#,
        );
        assert_eq!(request.questions[0].options[0].label, "PostgreSQL");
        assert_eq!(request.questions[0].options[0].id, "o1");
        let text = format_ask_user_outcome(
            request,
            AskUserOutcome::Answered {
                answers: vec![AskUserAnswer {
                    question_id: "db".into(),
                    selected: vec!["o2".into()],
                    text: Some("for now".into()),
                }],
            },
        );
        assert!(text.contains("\"selected\":[\"SQLite\"]"), "{text}");
        assert!(text.contains("\"note\":\"for now\""), "{text}");
    }

    #[test]
    fn invalid_calls_and_bad_json_explain_themselves() {
        assert!(matches!(
            parse_ask_user_request(r#"{"questions":[]}"#.into()),
            AskUserParse::Invalid { message } if message.contains("between 1 and 4")
        ));
        assert!(matches!(
            parse_ask_user_request("{".into()),
            AskUserParse::Invalid { message } if message.contains("not valid JSON")
        ));
    }

    #[test]
    fn declined_and_cancelled_need_no_answers() {
        let request = valid(r#"{"questions":[{"id":"name","question":"Name?"}]}"#);
        assert!(
            format_ask_user_outcome(request.clone(), AskUserOutcome::Declined).contains("declined")
        );
        assert!(format_ask_user_outcome(request, AskUserOutcome::Cancelled).contains("cancelled"));
    }
}
