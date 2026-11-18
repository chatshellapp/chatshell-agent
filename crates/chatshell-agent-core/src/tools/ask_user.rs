//! `ask_user`: the model asks the user one to four structured questions and
//! waits for the answers (host-injected seam).
//!
//! The core owns the model-facing contract — schema, description, input
//! validation, answer normalization and the tool-result text — so every
//! host advertises and reports the tool identically. Hosts only implement
//! [`AskUserPrompter`]: show the validated questions, return what the user
//! did. Desktop registers [`AskUserTool`] with rig; client-side tool callers
//! (iOS) use [`parse_request`] and [`format_outcome`] around their own UI.
//!
//! Design notes (internal-docs research, 2026-11-18, §8):
//! * Options are optional (a free-text question); when present, 2-4.
//! * Option ids are assigned by the runtime from position; the model only
//!   writes labels. Answers report labels back, never ids.
//! * `recommended` is a per-option flag; the recommended option is moved
//!   first. At most one per single-choice question.
//! * The user can always type their own answer; "Other"-style labels are
//!   reserved and rejected.
//! * A skipped question, a dismissed dialog and a stopped turn are all
//!   reported explicitly — none of them is consent.

use std::collections::HashSet;
use std::sync::Arc;

use futures::future::BoxFuture;
use rig::agent::tool::{Tool, ToolContext, ToolErrorKind, ToolExecutionError};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Provider-facing tool name.
pub const ASK_USER_TOOL_NAME: &str = "ask_user";

pub const MAX_QUESTIONS: usize = 4;
pub const MIN_OPTIONS: usize = 2;
pub const MAX_OPTIONS: usize = 4;
pub const MAX_HEADER_CHARS: usize = 12;
pub const MAX_ID_CHARS: usize = 64;
pub const MAX_QUESTION_CHARS: usize = 1_000;
pub const MAX_LABEL_CHARS: usize = 120;
pub const MAX_DESCRIPTION_CHARS: usize = 500;
/// Cap on the user's typed text per question.
pub const MAX_ANSWER_CHARS: usize = 4_000;

/// Labels the host renders itself (free-text row, discuss action); a model
/// option with one of these names would duplicate a row. Compared
/// case-insensitively.
const RESERVED_LABELS: &[&str] = &[
    "other",
    "other (type your own)",
    "none of the above",
    "chat about this",
];

const DESCRIPTION: &str = "Ask the user one to four short questions and wait for their answers. \
Use this only when a decision materially changes what you will do and you cannot resolve it from \
the conversation, the files, or sensible defaults. Do not use it for permission, confirmation, or \
progress check-ins.\n\
- Before calling, write the context the user needs to decide in your reply text; keep each \
question itself short.\n\
- Batch related questions into one call. Give each question a short snake_case `id`; answers are \
reported under it.\n\
- Offer 2-4 distinct options with a one-line `description` of the trade-off, or omit `options` \
for an open-ended question.\n\
- Set `recommended: true` on at most one option of a single-choice question, and only when one \
choice is clearly best practice rather than a matter of the user's taste.\n\
- Never add an \"Other\" option: the user can always type their own answer.\n\
- A skipped question or a dismissed dialog is not consent. Do not assume an answer, and do not \
ask the same question again with this tool.";

/// One validated question, as shown to the user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskUserQuestion {
    /// Model-chosen stable id; answers are keyed by it.
    pub id: String,
    pub question: String,
    /// Short chip label (at most [`MAX_HEADER_CHARS`] characters).
    pub header: Option<String>,
    /// Several options may be selected. Always false without options.
    pub multi: bool,
    /// Empty for a free-text question. The recommended option, if any,
    /// comes first.
    pub options: Vec<AskUserOption>,
}

/// One validated option. `id` is runtime-assigned (`o1`, `o2`, ...).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskUserOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub recommended: bool,
}

/// A validated `ask_user` call, ready for the host UI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskUserRequest {
    pub questions: Vec<AskUserQuestion>,
}

/// What the user entered for one question, as reported by the host.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskUserAnswer {
    pub question_id: String,
    /// Selected option ids (`o1`, ...). Unknown ids are ignored.
    #[serde(default)]
    pub selected: Vec<String>,
    /// Typed text: the answer itself when no option is selected, otherwise
    /// a note on the selection.
    #[serde(default)]
    pub text: Option<String>,
}

/// How an `ask_user` request ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AskUserOutcome {
    /// The user submitted. Questions without an entry count as skipped.
    Answered(Vec<AskUserAnswer>),
    /// The user wants to discuss in chat instead; any answers they already
    /// gave are kept.
    Discuss(Vec<AskUserAnswer>),
    /// The user dismissed the questions without answering.
    Declined,
    /// The request closed without the user (turn stopped, no UI).
    Cancelled,
}

/// Host-provided UI: show `request` and resolve when the user is done.
///
/// The future must resolve (with [`AskUserOutcome::Cancelled`] at worst)
/// when the turn is stopped; dropping it must close any open dialog.
pub trait AskUserPrompter: Send + Sync {
    fn prompt(&self, request: AskUserRequest) -> BoxFuture<'static, AskUserOutcome>;
}

/// Model-facing description.
pub fn description() -> &'static str {
    DESCRIPTION
}

/// JSON Schema for the tool arguments. Plain JSON Schema only (no
/// `anyOf`/`oneOf`), so every provider dialect accepts it.
pub fn parameters_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_QUESTIONS,
                "description": "The questions to ask, in display order (1-4).",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short snake_case identifier, unique within this call. The answer is reported under it."
                        },
                        "question": {
                            "type": "string",
                            "description": "The complete question, ending with a question mark."
                        },
                        "header": {
                            "type": "string",
                            "maxLength": MAX_HEADER_CHARS,
                            "description": "Very short label shown as a chip, at most 12 characters (e.g. \"Database\")."
                        },
                        "multi": {
                            "type": "boolean",
                            "description": "Allow the user to select several options. Default false."
                        },
                        "options": {
                            "type": "array",
                            "minItems": MIN_OPTIONS,
                            "maxItems": MAX_OPTIONS,
                            "description": "2-4 distinct choices. Omit for an open-ended question. Never include an \"Other\" option; the user can always type their own answer.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "Concise choice text (1-5 words)."
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "One line on what this choice means or its trade-off."
                                    },
                                    "recommended": {
                                        "type": "boolean",
                                        "description": "Mark the option you recommend. At most one per single-choice question."
                                    }
                                },
                                "required": ["label"]
                            }
                        }
                    },
                    "required": ["id", "question"]
                }
            }
        },
        "required": ["questions"]
    })
}

#[derive(Deserialize)]
struct RawArgs {
    questions: Vec<RawQuestion>,
}

#[derive(Deserialize)]
struct RawQuestion {
    id: String,
    question: String,
    #[serde(default)]
    header: Option<String>,
    #[serde(default)]
    multi: Option<bool>,
    #[serde(default)]
    options: Option<Vec<RawOption>>,
}

#[derive(Deserialize)]
struct RawOption {
    label: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    recommended: Option<bool>,
}

/// Drop carriage returns (some models emit `\r` runs) and trim.
fn clean_text(text: &str) -> String {
    text.replace('\r', "").trim().to_string()
}

/// Single-line form: whitespace runs (including newlines) collapse.
fn clean_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn char_len(text: &str) -> usize {
    text.chars().count()
}

fn non_empty(text: Option<String>) -> Option<String> {
    text.filter(|t| !t.is_empty())
}

/// Validate and normalize the model's arguments. Errors are plain text
/// naming the offending value, meant to be returned to the model.
pub fn parse_request(args: &Value) -> Result<AskUserRequest, String> {
    let mut args = args.clone();
    // Some models send nested arrays as a JSON string.
    if let Some(Value::String(text)) = args.get("questions") {
        let parsed: Value = serde_json::from_str(text)
            .map_err(|e| format!("`questions` must be an array of question objects: {e}"))?;
        args["questions"] = parsed;
    }
    let raw: RawArgs = serde_json::from_value(args).map_err(|e| {
        format!("Invalid ask_user arguments: {e}. Expected {{\"questions\": [{{\"id\", \"question\", \"options\"?}}]}}.")
    })?;

    if raw.questions.is_empty() || raw.questions.len() > MAX_QUESTIONS {
        return Err(format!(
            "Ask between 1 and {MAX_QUESTIONS} questions per call (got {}).",
            raw.questions.len()
        ));
    }

    let mut seen_ids = HashSet::new();
    let mut questions = Vec::with_capacity(raw.questions.len());
    for raw_question in raw.questions {
        let id = clean_line(&raw_question.id);
        if id.is_empty() {
            return Err("Every question needs a non-empty `id`.".to_string());
        }
        if char_len(&id) > MAX_ID_CHARS
            || !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "Question id \"{id}\" must be a short snake_case identifier (letters, digits, `_`, `-`; at most {MAX_ID_CHARS} characters)."
            ));
        }
        if !seen_ids.insert(id.clone()) {
            return Err(format!("Question id \"{id}\" is used more than once."));
        }

        let question = clean_text(&raw_question.question);
        if question.is_empty() {
            return Err(format!("Question \"{id}\" has empty `question` text."));
        }
        if char_len(&question) > MAX_QUESTION_CHARS {
            return Err(format!(
                "Question \"{id}\" is too long (at most {MAX_QUESTION_CHARS} characters). Put context in your reply text instead."
            ));
        }

        let header = non_empty(raw_question.header.as_deref().map(clean_line));
        if let Some(header) = &header
            && char_len(header) > MAX_HEADER_CHARS
        {
            return Err(format!(
                "Header \"{header}\" of question \"{id}\" is longer than {MAX_HEADER_CHARS} characters."
            ));
        }

        let raw_options = match raw_question.options {
            None => Vec::new(),
            Some(options) if options.is_empty() => {
                return Err(format!(
                    "Question \"{id}\" has an empty `options` array. Omit `options` for an open-ended question."
                ));
            }
            Some(options) => options,
        };
        if !raw_options.is_empty()
            && (raw_options.len() < MIN_OPTIONS || raw_options.len() > MAX_OPTIONS)
        {
            return Err(format!(
                "Question \"{id}\" must offer between {MIN_OPTIONS} and {MAX_OPTIONS} options (got {}), or omit `options`.",
                raw_options.len()
            ));
        }
        let multi = raw_question.multi.unwrap_or(false) && !raw_options.is_empty();

        let mut seen_labels = HashSet::new();
        let mut options = Vec::with_capacity(raw_options.len());
        for raw_option in raw_options {
            let label = clean_line(&raw_option.label);
            if label.is_empty() {
                return Err(format!(
                    "Question \"{id}\" has an option with an empty label."
                ));
            }
            if char_len(&label) > MAX_LABEL_CHARS {
                return Err(format!(
                    "Option \"{label}\" of question \"{id}\" is too long (at most {MAX_LABEL_CHARS} characters). Move detail into `description`."
                ));
            }
            let folded = label.to_lowercase();
            if RESERVED_LABELS.contains(&folded.as_str()) {
                return Err(format!(
                    "Option \"{label}\" of question \"{id}\" is reserved: the user can always type their own answer, so remove it."
                ));
            }
            if !seen_labels.insert(folded) {
                return Err(format!(
                    "Option \"{label}\" appears more than once in question \"{id}\"."
                ));
            }
            let description = non_empty(raw_option.description.as_deref().map(clean_text));
            if let Some(description) = &description
                && char_len(description) > MAX_DESCRIPTION_CHARS
            {
                return Err(format!(
                    "The description of option \"{label}\" in question \"{id}\" is too long (at most {MAX_DESCRIPTION_CHARS} characters)."
                ));
            }
            options.push(AskUserOption {
                id: String::new(),
                label,
                description,
                recommended: raw_option.recommended.unwrap_or(false),
            });
        }

        let recommended = options.iter().filter(|o| o.recommended).count();
        if !multi && recommended > 1 {
            return Err(format!(
                "Question \"{id}\" marks {recommended} options as recommended; mark at most one."
            ));
        }
        // Recommended first, otherwise the model's order; ids by position.
        options.sort_by_key(|o| !o.recommended);
        for (index, option) in options.iter_mut().enumerate() {
            option.id = format!("o{}", index + 1);
        }

        questions.push(AskUserQuestion {
            id,
            question,
            header,
            multi,
            options,
        });
    }
    Ok(AskUserRequest { questions })
}

/// One question's normalized result, as reported to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct QuestionResult {
    id: String,
    question: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    selected: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    free_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    skipped: bool,
}

fn truncate_chars(text: String, max: usize) -> String {
    if char_len(&text) <= max {
        text
    } else {
        text.chars().take(max).collect()
    }
}

/// Resolve host answers against the request: unknown questions and option
/// ids are ignored, single-choice keeps the first selection, typed text is
/// cleaned and capped, and questions without input are marked skipped.
fn resolve_answers(request: &AskUserRequest, answers: &[AskUserAnswer]) -> Vec<QuestionResult> {
    request
        .questions
        .iter()
        .map(|question| {
            let answer = answers.iter().find(|a| a.question_id == question.id);
            let mut selected: Vec<String> = Vec::new();
            if let Some(answer) = answer {
                for option in &question.options {
                    if answer.selected.iter().any(|id| id == &option.id) {
                        selected.push(option.label.clone());
                    }
                }
                // Keep the user's first pick, not the first in display order.
                if !question.multi && selected.len() > 1 {
                    let first = answer.selected.iter().find_map(|id| {
                        question
                            .options
                            .iter()
                            .find(|o| &o.id == id)
                            .map(|o| o.label.clone())
                    });
                    selected = first.into_iter().collect();
                }
            }
            let text = answer
                .and_then(|a| a.text.as_deref())
                .map(clean_text)
                .filter(|t| !t.is_empty())
                .map(|t| truncate_chars(t, MAX_ANSWER_CHARS));
            let (free_text, note) = if selected.is_empty() {
                (text, None)
            } else {
                (None, text)
            };
            let skipped = selected.is_empty() && free_text.is_none();
            QuestionResult {
                id: question.id.clone(),
                question: question.question.clone(),
                selected,
                free_text,
                note,
                skipped,
            }
        })
        .collect()
}

/// The tool-result text for `outcome`: a one-line instruction for the
/// model followed by the structured result.
pub fn format_outcome(request: &AskUserRequest, outcome: &AskUserOutcome) -> String {
    match outcome {
        AskUserOutcome::Answered(answers) => {
            let results = resolve_answers(request, answers);
            let any_skipped = results.iter().any(|r| r.skipped);
            let lead = if results.iter().all(|r| r.skipped) {
                "The user submitted without answering any question. This is not consent: do not \
                 assume answers, and do not ask the same questions again with ask_user."
            } else if any_skipped {
                "The user answered your questions. Skipped questions are not consent: do not \
                 assume an answer for them."
            } else {
                "The user answered your questions. Continue with their answers in mind."
            };
            let body = json!({ "status": "answered", "answers": results });
            format!("{lead}\n{body}")
        }
        AskUserOutcome::Discuss(answers) => {
            let results = resolve_answers(request, answers);
            let body = json!({ "status": "discuss", "answers": results });
            format!(
                "The user wants to discuss these questions in chat instead of answering them \
                 here. Do not call ask_user again for them: reply in plain text and end your \
                 turn so the user can respond.\n{body}"
            )
        }
        AskUserOutcome::Declined => {
            let body = json!({ "status": "declined" });
            format!(
                "The user dismissed the questions without answering. This is not consent. Do \
                 not ask them again with ask_user: continue with what you know and state the \
                 assumptions you make, or ask in plain text and end your turn.\n{body}"
            )
        }
        AskUserOutcome::Cancelled => {
            let body = json!({ "status": "cancelled" });
            format!("The questions closed before the user answered.\n{body}")
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct AskUserError(String);

/// The rig tool: validates the call, awaits the host prompter, reports the
/// outcome.
#[derive(Clone)]
pub struct AskUserTool {
    prompter: Arc<dyn AskUserPrompter>,
}

impl std::fmt::Debug for AskUserTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AskUserTool").finish()
    }
}

impl AskUserTool {
    pub fn new(prompter: Arc<dyn AskUserPrompter>) -> Self {
        Self { prompter }
    }
}

impl Tool for AskUserTool {
    const NAME: &'static str = ASK_USER_TOOL_NAME;

    type Error = AskUserError;
    type Args = Value;
    type Output = String;

    fn description(&self) -> String {
        DESCRIPTION.to_string()
    }

    fn parameters(&self) -> Value {
        parameters_schema()
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::new(ToolErrorKind::InvalidArgs, error.0.clone())
            .with_model_feedback(error.0)
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let request = parse_request(&args).map_err(AskUserError)?;
        tracing::info!(
            "🔧 [tool-call] ask_user: {} question(s)",
            request.questions.len()
        );
        let outcome = self.prompter.prompt(request.clone()).await;
        tracing::info!("❓ [ask_user] outcome: {}", outcome_name(&outcome));
        Ok(format_outcome(&request, &outcome))
    }
}

fn outcome_name(outcome: &AskUserOutcome) -> &'static str {
    match outcome {
        AskUserOutcome::Answered(_) => "answered",
        AskUserOutcome::Discuss(_) => "discuss",
        AskUserOutcome::Declined => "declined",
        AskUserOutcome::Cancelled => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn db_question() -> Value {
        json!({
            "id": "database",
            "question": "Which database should the service use?",
            "header": "Database",
            "options": [
                {"label": "SQLite", "description": "Embedded, zero setup"},
                {"label": "PostgreSQL", "description": "Server, scales", "recommended": true}
            ]
        })
    }

    fn parse(questions: Value) -> Result<AskUserRequest, String> {
        parse_request(&json!({ "questions": questions }))
    }

    fn err(questions: Value) -> String {
        parse(questions).unwrap_err()
    }

    #[test]
    fn recommended_option_moves_first_and_ids_follow_position() {
        let request = parse(json!([db_question()])).unwrap();
        let q = &request.questions[0];
        assert_eq!(q.header.as_deref(), Some("Database"));
        assert!(!q.multi);
        let labels: Vec<_> = q
            .options
            .iter()
            .map(|o| (o.id.as_str(), o.label.as_str()))
            .collect();
        assert_eq!(labels, vec![("o1", "PostgreSQL"), ("o2", "SQLite")]);
        assert!(q.options[0].recommended);
    }

    #[test]
    fn omitted_options_make_a_free_text_question() {
        let request =
            parse(json!([{"id": "name", "question": "Project name?", "multi": true}])).unwrap();
        assert!(request.questions[0].options.is_empty());
        assert!(
            !request.questions[0].multi,
            "multi is meaningless without options"
        );
    }

    #[test]
    fn stringified_questions_are_accepted() {
        let text = serde_json::to_string(&json!([db_question()])).unwrap();
        let request = parse_request(&json!({ "questions": text })).unwrap();
        assert_eq!(request.questions[0].id, "database");
    }

    #[test]
    fn carriage_returns_and_whitespace_are_cleaned() {
        let request = parse(json!([{
            "id": " scope ",
            "question": "\r\r  Which scope?\r\n",
            "options": [{"label": "  Just\nthis   file "}, {"label": "Whole repo"}]
        }]))
        .unwrap();
        let q = &request.questions[0];
        assert_eq!(q.id, "scope");
        assert_eq!(q.question, "Which scope?");
        assert_eq!(q.options[0].label, "Just this file");
    }

    #[test]
    fn counts_are_enforced() {
        assert!(err(json!([])).contains("between 1 and 4"));
        let five: Vec<Value> = (0..5)
            .map(|i| json!({"id": format!("q{i}"), "question": "?"}))
            .collect();
        assert!(err(json!(five)).contains("between 1 and 4"));
        assert!(
            err(json!([{"id": "a", "question": "?", "options": []}])).contains("Omit `options`")
        );
        assert!(
            err(json!([{"id": "a", "question": "?", "options": [{"label": "x"}]}]))
                .contains("between 2 and 4")
        );
        let five_options: Vec<Value> = (0..5).map(|i| json!({"label": format!("o{i}")})).collect();
        assert!(
            err(json!([{"id": "a", "question": "?", "options": five_options}]))
                .contains("between 2 and 4")
        );
    }

    #[test]
    fn ids_must_be_unique_and_identifier_shaped() {
        assert!(
            err(json!([{"id": "a", "question": "?"}, {"id": "a", "question": "?"}]))
                .contains("more than once")
        );
        assert!(err(json!([{"id": "has space", "question": "?"}])).contains("snake_case"));
        assert!(err(json!([{"id": "", "question": "?"}])).contains("non-empty"));
    }

    #[test]
    fn labels_must_be_unique_and_not_reserved() {
        assert!(
            err(json!([{"id": "a", "question": "?", "options": [{"label": "Yes"}, {"label": "yes"}]}]))
                .contains("more than once")
        );
        assert!(
            err(json!([{"id": "a", "question": "?", "options": [{"label": "Yes"}, {"label": "Other"}]}]))
                .contains("reserved")
        );
        assert!(
            err(
                json!([{"id": "a", "question": "?", "options": [{"label": " "}, {"label": "No"}]}])
            )
            .contains("empty label")
        );
    }

    #[test]
    fn header_length_and_single_recommendation_are_enforced() {
        assert!(
            err(json!([{"id": "a", "question": "?", "header": "Far too long header"}]))
                .contains("longer than 12")
        );
        let two = json!([{"id": "a", "question": "?", "options": [
            {"label": "x", "recommended": true}, {"label": "y", "recommended": true}
        ]}]);
        assert!(err(two).contains("at most one"));
        let multi = json!([{"id": "a", "question": "?", "multi": true, "options": [
            {"label": "x", "recommended": true}, {"label": "y", "recommended": true}
        ]}]);
        assert!(parse(multi).is_ok(), "multi-select may recommend several");
    }

    #[test]
    fn malformed_arguments_name_the_expected_shape() {
        let message = parse_request(&json!({"question": "?"})).unwrap_err();
        assert!(message.contains("Expected"), "{message}");
    }

    fn two_questions() -> AskUserRequest {
        parse(json!([
            db_question(),
            {"id": "name", "question": "Project name?"}
        ]))
        .unwrap()
    }

    fn body(text: &str) -> Value {
        serde_json::from_str(text.lines().last().unwrap()).unwrap()
    }

    #[test]
    fn answers_report_labels_notes_and_free_text() {
        let request = two_questions();
        let text = format_outcome(
            &request,
            &AskUserOutcome::Answered(vec![
                AskUserAnswer {
                    question_id: "database".into(),
                    selected: vec!["o2".into()],
                    text: Some("  for the prototype \r".into()),
                },
                AskUserAnswer {
                    question_id: "name".into(),
                    selected: vec![],
                    text: Some("atlas".into()),
                },
            ]),
        );
        assert!(text.starts_with("The user answered your questions. Continue"));
        let body = body(&text);
        assert_eq!(body["status"], "answered");
        assert_eq!(body["answers"][0]["selected"], json!(["SQLite"]));
        assert_eq!(body["answers"][0]["note"], "for the prototype");
        assert_eq!(body["answers"][1]["free_text"], "atlas");
        assert!(body["answers"][1].get("skipped").is_none());
    }

    #[test]
    fn missing_answers_are_reported_as_skipped() {
        let request = two_questions();
        let text = format_outcome(
            &request,
            &AskUserOutcome::Answered(vec![AskUserAnswer {
                question_id: "database".into(),
                selected: vec!["o1".into()],
                text: None,
            }]),
        );
        assert!(text.contains("Skipped questions are not consent"));
        let body = body(&text);
        assert_eq!(body["answers"][1]["skipped"], true);

        let none = format_outcome(&request, &AskUserOutcome::Answered(vec![]));
        assert!(none.contains("without answering any question"));
    }

    #[test]
    fn unknown_ids_are_ignored_and_single_choice_keeps_the_first_pick() {
        let request = two_questions();
        let text = format_outcome(
            &request,
            &AskUserOutcome::Answered(vec![
                AskUserAnswer {
                    question_id: "database".into(),
                    selected: vec!["o9".into(), "o2".into(), "o1".into()],
                    text: None,
                },
                AskUserAnswer {
                    question_id: "ghost".into(),
                    selected: vec!["o1".into()],
                    text: Some("x".into()),
                },
            ]),
        );
        let body = body(&text);
        assert_eq!(body["answers"][0]["selected"], json!(["SQLite"]));
        assert_eq!(body["answers"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn typed_text_is_capped() {
        let request = parse(json!([{"id": "name", "question": "?"}])).unwrap();
        let text = format_outcome(
            &request,
            &AskUserOutcome::Answered(vec![AskUserAnswer {
                question_id: "name".into(),
                selected: vec![],
                text: Some("x".repeat(MAX_ANSWER_CHARS + 50)),
            }]),
        );
        let body = body(&text);
        assert_eq!(
            body["answers"][0]["free_text"]
                .as_str()
                .unwrap()
                .chars()
                .count(),
            MAX_ANSWER_CHARS
        );
    }

    #[test]
    fn discuss_declined_and_cancelled_are_explicit() {
        let request = two_questions();
        let discuss = format_outcome(
            &request,
            &AskUserOutcome::Discuss(vec![AskUserAnswer {
                question_id: "database".into(),
                selected: vec!["o1".into()],
                text: None,
            }]),
        );
        assert!(discuss.contains("discuss these questions in chat"));
        assert_eq!(
            body(&discuss)["answers"][0]["selected"],
            json!(["PostgreSQL"])
        );

        let declined = format_outcome(&request, &AskUserOutcome::Declined);
        assert!(declined.contains("not consent"));
        assert_eq!(body(&declined)["status"], "declined");

        let cancelled = format_outcome(&request, &AskUserOutcome::Cancelled);
        assert_eq!(body(&cancelled)["status"], "cancelled");
    }

    struct Scripted {
        seen: Mutex<Option<AskUserRequest>>,
        outcome: AskUserOutcome,
    }

    impl AskUserPrompter for Scripted {
        fn prompt(&self, request: AskUserRequest) -> BoxFuture<'static, AskUserOutcome> {
            *self.seen.lock().unwrap() = Some(request);
            let outcome = self.outcome.clone();
            Box::pin(async move { outcome })
        }
    }

    #[tokio::test]
    async fn tool_hands_the_validated_request_to_the_prompter() {
        let prompter = Arc::new(Scripted {
            seen: Mutex::new(None),
            outcome: AskUserOutcome::Answered(vec![AskUserAnswer {
                question_id: "database".into(),
                selected: vec!["o1".into()],
                text: None,
            }]),
        });
        let tool = AskUserTool::new(prompter.clone());
        let output = tool
            .call(
                &mut ToolContext::new(),
                json!({ "questions": [db_question()] }),
            )
            .await
            .unwrap();
        assert_eq!(
            body(&output)["answers"][0]["selected"],
            json!(["PostgreSQL"])
        );
        let seen = prompter.seen.lock().unwrap().clone().unwrap();
        assert_eq!(seen.questions[0].options[0].label, "PostgreSQL");
    }

    #[tokio::test]
    async fn invalid_calls_never_reach_the_prompter_and_explain_why() {
        let prompter = Arc::new(Scripted {
            seen: Mutex::new(None),
            outcome: AskUserOutcome::Declined,
        });
        let tool = AskUserTool::new(prompter.clone());
        let error = tool
            .call(&mut ToolContext::new(), json!({ "questions": [] }))
            .await
            .unwrap_err();
        let mapped = tool.map_error(error);
        assert!(mapped.model_feedback().unwrap().contains("between 1 and 4"));
        assert!(prompter.seen.lock().unwrap().is_none());
    }

    #[test]
    fn schema_uses_no_combinators() {
        let schema = parameters_schema().to_string();
        for combinator in ["anyOf", "oneOf", "allOf"] {
            assert!(!schema.contains(combinator), "{combinator} in schema");
        }
        assert_eq!(<AskUserTool as Tool>::NAME, "ask_user");
    }
}
