//! The auto-mode judge engine (AR-5, AR-8, AR-9).
//!
//! Provider-agnostic: the host implements [`JudgeModel`] (one completion
//! call with a system prompt and a single user message); everything else
//! — layout, temperature 0, the output bound, the per-attempt timeout,
//! the single retry on transient errors, cancellation, verdict parsing,
//! and the failure semantics — lives here and is testable against a mock.
//!
//! Failure semantics never produce a bare allow: [`ReviewOutcome::Allow`]
//! is only ever built from a parsed `allow` verdict. Every other path
//! ends in a block or a review failure, each with model-facing wording.

use super::JudgmentRequest;
use super::ruleset::{self, Posture};
use super::transcript::AssembledTranscript;
use super::verdict::{self, Outcome, Verdict};
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Per-attempt timeout (OQ-1 default).
pub const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(60);

/// Output bound (OQ-7 default).
pub const MAX_OUTPUT_TOKENS: u32 = 256;

/// Output bound of the one retry after a reply ran out of budget. Models
/// that think before answering spend output tokens on reasoning, and how
/// much varies by model and endpoint, not by vendor; a larger bound is the
/// one remedy every provider accepts.
pub const ESCALATED_OUTPUT_TOKENS: u32 = 2_048;

/// One completion call.
#[derive(Debug, Clone)]
pub struct JudgeCall<'a> {
    pub system: &'a str,
    pub user: &'a str,
    /// `Some(0.0)` by default; `None` for models that reject or ignore
    /// the parameter.
    pub temperature: Option<f64>,
    pub max_tokens: u32,
    /// Provider-specific request fields (thinking or reasoning-effort
    /// controls), merged into the request body.
    pub extra: Option<&'a serde_json::Value>,
    pub cancel: CancellationToken,
}

/// A failed completion call.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JudgeError {
    /// Worth one retry: rate limit, 5xx, overload, network, timeout.
    #[error("{0}")]
    Transient(String),
    /// Not worth retrying: auth, bad request, unsupported parameter.
    #[error("{0}")]
    Permanent(String),
    /// The reply ran out of output budget before it held an answer
    /// (usually reasoning tokens). Worth one retry with a larger bound.
    #[error("{0}")]
    Truncated(String),
    #[error("cancelled")]
    Cancelled,
}

/// The host's model call. `label` names the model in the decision log.
pub trait JudgeModel: Send + Sync {
    fn label(&self) -> String;
    fn complete<'a>(&'a self, call: JudgeCall<'a>) -> BoxFuture<'a, Result<String, JudgeError>>;
}

/// Call parameters for a model: the host's preferred profile, reduced by
/// the probe to what the provider accepts.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeParams {
    pub temperature: Option<f64>,
    pub max_tokens: u32,
    pub attempt_timeout: Duration,
    /// Provider-specific request fields that keep thinking minimal (for
    /// example `{"reasoning_effort": "low"}`); `None` sends nothing extra.
    pub extra: Option<serde_json::Value>,
}

impl Default for JudgeParams {
    fn default() -> Self {
        Self {
            temperature: Some(0.0),
            max_tokens: MAX_OUTPUT_TOKENS,
            attempt_timeout: ATTEMPT_TIMEOUT,
            extra: None,
        }
    }
}

/// Why a review produced no verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewFailure {
    /// The provider failed (after the retry, when transient).
    Unavailable(String),
    /// The provider answered, but not with a parseable verdict.
    Unparseable(String),
    /// The reply ran out of output budget, even at the larger bound.
    Truncated(String),
}

impl ReviewFailure {
    fn detail(&self) -> &str {
        match self {
            ReviewFailure::Unavailable(detail)
            | ReviewFailure::Unparseable(detail)
            | ReviewFailure::Truncated(detail) => detail,
        }
    }
}

/// Result of one review.
#[derive(Debug, Clone, PartialEq)]
pub enum ReviewOutcome {
    Allow {
        verdict: Verdict,
    },
    Block {
        verdict: Verdict,
    },
    Failed {
        failure: ReviewFailure,
    },
    /// The stream ended while the review was in flight.
    Cancelled,
}

impl ReviewOutcome {
    /// The tool-result text fed back to the model for a non-allow outcome.
    pub fn model_reason(&self) -> Option<String> {
        match self {
            ReviewOutcome::Allow { .. } => None,
            ReviewOutcome::Block { verdict } => Some(block_reason(verdict)),
            ReviewOutcome::Failed { failure } => Some(failure_reason(failure)),
            ReviewOutcome::Cancelled => Some("Auto-review was cancelled.".to_string()),
        }
    }

    pub fn verdict(&self) -> Option<&Verdict> {
        match self {
            ReviewOutcome::Allow { verdict } | ReviewOutcome::Block { verdict } => Some(verdict),
            _ => None,
        }
    }
}

/// Denial wording: the reason, then anti-workaround guidance.
pub fn block_reason(verdict: &Verdict) -> String {
    let because = if verdict.rationale.is_empty() {
        "the action was judged too risky for what the user asked".to_string()
    } else {
        verdict.rationale.clone()
    };
    format!(
        "Auto-review blocked this action: {because} Do not tunnel the same action \
         through another path or tool; use a safer alternative or ask the user."
    )
}

/// Review-failure wording: not a safety determination, retry spent.
pub fn failure_reason(failure: &ReviewFailure) -> String {
    format!(
        "Auto-review could not judge this action ({}). This is a review failure, not a \
         determination that the action is unsafe. A retry was already attempted; do not \
         bypass the review. Ask the user, or continue with a different approach.",
        failure.detail()
    )
}

/// A completed review with its accounting.
#[derive(Debug, Clone, PartialEq)]
pub struct Review {
    pub outcome: ReviewOutcome,
    /// Calls made: 1, plus one retry after a transient failure and one
    /// after a reply ran out of budget.
    pub attempts: u32,
    pub duration: Duration,
    pub model: String,
    /// Output bound of the last call (raised when a reply ran out of
    /// budget). Hosts keep a raised bound for the model's later reviews.
    pub max_tokens: u32,
}

/// Run one review.
pub async fn review(
    model: &dyn JudgeModel,
    params: &JudgeParams,
    posture: &Posture,
    transcript: &AssembledTranscript,
    request: &JudgmentRequest,
    cancel: &CancellationToken,
) -> Review {
    let started = Instant::now();
    let user = ruleset::render_request(posture, transcript, request);
    let (outcome, attempts, max_tokens) = run(model, params, &user, cancel).await;
    Review {
        outcome,
        attempts,
        duration: started.elapsed(),
        model: model.label(),
        max_tokens,
    }
}

async fn run(
    model: &dyn JudgeModel,
    params: &JudgeParams,
    user: &str,
    cancel: &CancellationToken,
) -> (ReviewOutcome, u32, u32) {
    let mut attempts = 0;
    let mut max_tokens = params.max_tokens;
    // One retry on a transient failure, and separately one retry with a
    // larger output bound when a reply ran out of budget.
    let mut transient_retry = true;
    let mut budget_retry = max_tokens < ESCALATED_OUTPUT_TOKENS;
    loop {
        attempts += 1;
        let failure = match attempt(model, params, max_tokens, user, cancel).await {
            Ok(text) => {
                // Inline `<think>` blocks are reasoning, never the verdict.
                let answer = crate::thinking::parse_thinking_content(&text).content;
                let outcome = match verdict::parse_verdict(&answer) {
                    Ok(verdict) if verdict.outcome == Outcome::Allow => {
                        ReviewOutcome::Allow { verdict }
                    }
                    Ok(verdict) => ReviewOutcome::Block { verdict },
                    // A verdict cut off by the output bound is a budget
                    // problem, not a model that cannot answer.
                    Err(_) if verdict::looks_cut_off(&text) && budget_retry => {
                        budget_retry = false;
                        max_tokens = ESCALATED_OUTPUT_TOKENS;
                        continue;
                    }
                    Err(_) if verdict::looks_cut_off(&text) => ReviewOutcome::Failed {
                        failure: ReviewFailure::Truncated(
                            "the reply was cut off by the output bound".to_string(),
                        ),
                    },
                    Err(err) => ReviewOutcome::Failed {
                        failure: ReviewFailure::Unparseable(err.to_string()),
                    },
                };
                return (outcome, attempts, max_tokens);
            }
            Err(JudgeError::Cancelled) => return (ReviewOutcome::Cancelled, attempts, max_tokens),
            Err(JudgeError::Permanent(detail)) => ReviewFailure::Unavailable(detail),
            Err(JudgeError::Truncated(_)) if budget_retry => {
                budget_retry = false;
                max_tokens = ESCALATED_OUTPUT_TOKENS;
                continue;
            }
            Err(JudgeError::Truncated(detail)) => ReviewFailure::Truncated(detail),
            Err(JudgeError::Transient(_)) if transient_retry => {
                transient_retry = false;
                continue;
            }
            Err(JudgeError::Transient(detail)) => ReviewFailure::Unavailable(detail),
        };
        return (ReviewOutcome::Failed { failure }, attempts, max_tokens);
    }
}

async fn attempt(
    model: &dyn JudgeModel,
    params: &JudgeParams,
    max_tokens: u32,
    user: &str,
    cancel: &CancellationToken,
) -> Result<String, JudgeError> {
    if cancel.is_cancelled() {
        return Err(JudgeError::Cancelled);
    }
    let call = JudgeCall {
        system: ruleset::SYSTEM_PROMPT,
        user,
        temperature: params.temperature,
        max_tokens,
        extra: params.extra.as_ref(),
        cancel: cancel.child_token(),
    };
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(JudgeError::Cancelled),
        result = tokio::time::timeout(params.attempt_timeout, model.complete(call)) => {
            match result {
                Ok(result) => result,
                Err(_elapsed) => Err(JudgeError::Transient(format!(
                    "the reviewer timed out after {:?}",
                    params.attempt_timeout
                ))),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Denial counters (AR-8)
// ---------------------------------------------------------------------------

/// Consecutive denials that trigger the mode-switch suggestion.
pub const CONSECUTIVE_DENIAL_LIMIT: u32 = 3;
/// Total denials per conversation that trigger it.
pub const TOTAL_DENIAL_LIMIT: u32 = 20;

#[derive(Debug, Default, Clone, Copy)]
struct DenialState {
    consecutive: u32,
    total: u32,
    suggested: bool,
}

/// Per-conversation denial counters. Blocks and review failures count as
/// denials; any allow resets the consecutive count. The suggestion fires
/// at most once per conversation and never aborts the turn.
#[derive(Debug, Default)]
pub struct DenialCounters {
    states: Mutex<HashMap<String, DenialState>>,
}

impl DenialCounters {
    /// Record a review result. Returns `true` exactly when the one-time
    /// mode-switch suggestion should be shown now.
    pub fn record(&self, conversation_id: &str, outcome: &ReviewOutcome) -> bool {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let state = states.entry(conversation_id.to_string()).or_default();
        match outcome {
            ReviewOutcome::Allow { .. } => {
                state.consecutive = 0;
                false
            }
            ReviewOutcome::Block { .. } | ReviewOutcome::Failed { .. } => {
                state.consecutive += 1;
                state.total += 1;
                let due = state.consecutive >= CONSECUTIVE_DENIAL_LIMIT
                    || state.total >= TOTAL_DENIAL_LIMIT;
                if due && !state.suggested {
                    state.suggested = true;
                    true
                } else {
                    false
                }
            }
            ReviewOutcome::Cancelled => false,
        }
    }

    /// Drop a conversation's counters (conversation deleted).
    pub fn forget(&self, conversation_id: &str) {
        self.states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(conversation_id);
    }
}

// ---------------------------------------------------------------------------
// Capability gate (AR-9)
// ---------------------------------------------------------------------------

/// Whether a model can serve as the judge.
#[derive(Debug, Clone, PartialEq)]
pub enum Eligibility {
    Eligible(JudgeParams),
    Ineligible { reason: String },
}

/// How `auto` resolves for a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoResolution {
    /// Classifier-judged, zero prompts.
    Judged,
    /// No eligible judge: v2 semantics (sandbox plus a few prompts).
    SandboxPrompts,
    /// No eligible judge and no sandbox backend.
    Ask,
}

/// Resolve `auto` from eligibility and sandbox availability.
pub fn resolve_auto(eligibility: Option<&Eligibility>, sandbox_available: bool) -> AutoResolution {
    match eligibility {
        Some(Eligibility::Eligible(_)) => AutoResolution::Judged,
        _ if sandbox_available => AutoResolution::SandboxPrompts,
        _ => AutoResolution::Ask,
    }
}

/// The probe's canned action: a realistic judgment, not a trivial one. A
/// model that thinks before answering thinks about as long here as on
/// real calls, so the output bound the probe settles on also holds for
/// them. The user asked for the deletion, so either verdict is plausible;
/// eligibility only needs a parseable one.
fn probe_inputs() -> (Posture, AssembledTranscript, JudgmentRequest) {
    use super::{Evidence, JudgmentSource, TranscriptBudget, TranscriptEntry};
    use crate::permission::{ForcedPrompt, ForcedPromptKind, GateContext};
    let ctx = GateContext {
        workspace: Some("/Users/demo/projects/site".into()),
        sandbox_available: false,
        network_allowed: false,
        scratch_dir: None,
    };
    let command = "rm -rf /Users/demo/Documents/old-drafts";
    let args = serde_json::json!({ "command": command });
    let request = JudgmentRequest {
        subject: command.to_string(),
        source: JudgmentSource::OutsideDeletion,
        forced: Some(ForcedPrompt {
            kind: ForcedPromptKind::OutsideDeletion,
            detail: "Permanently deletes an existing path outside the workspace: \
                     /Users/demo/Documents/old-drafts"
                .to_string(),
        }),
        evidence: Evidence::collect("bash", &args, &ctx),
    };
    let transcript = super::assemble(
        &[
            TranscriptEntry::User {
                text: "Move the finished drafts into the site's posts folder, then \
                       delete ~/Documents/old-drafts — I don't need it anymore."
                    .to_string(),
            },
            TranscriptEntry::ToolInput {
                tool: "glob".to_string(),
                input: serde_json::json!({
                    "pattern": "*.md",
                    "path": "/Users/demo/Documents/old-drafts"
                }),
            },
            TranscriptEntry::ToolInput {
                tool: "bash".to_string(),
                input: serde_json::json!({
                    "command": "cp /Users/demo/Documents/old-drafts/*.md content/posts/"
                }),
            },
        ],
        TranscriptBudget::default(),
    );
    let posture = Posture {
        os: std::env::consts::OS.to_string(),
        workspace: Some("/Users/demo/projects/site".to_string()),
        jail: super::JailProfile::Absent,
        mcp_servers: Vec::new(),
    };
    (posture, transcript, request)
}

/// Setup-time capability probe with the default parameters and no
/// thinking controls.
pub async fn probe(model: &dyn JudgeModel, cancel: &CancellationToken) -> Eligibility {
    probe_with(model, JudgeParams::default(), &[], cancel).await
}

/// Setup-time capability probe: one realistic judgment, repeated until a
/// parameter set yields a parseable verdict.
///
/// `thinking` lists request fields that reduce reasoning, most reducing
/// first (the host knows its wires: `reasoning_effort` levels, `thinking:
/// disabled`, `think: false`, ...). Each is tried in order, then none.
/// Which values a model accepts varies by model and endpoint, so the
/// probe finds out instead of keeping a vendor table:
///
/// - a rejected request (a permanent error) moves to the next candidate,
///   after one try without temperature (some models reject that instead);
/// - a reply that runs out of budget is retried once at
///   [`ESCALATED_OUTPUT_TOKENS`] by the review itself; if it still runs
///   out, the next candidate may think less;
/// - an unparseable reply moves to the next candidate.
///
/// The parameters that worked are returned for reuse, including a raised
/// output bound. Tests capability, not names: any model that passes is
/// eligible.
pub async fn probe_with(
    model: &dyn JudgeModel,
    base: JudgeParams,
    thinking: &[serde_json::Value],
    cancel: &CancellationToken,
) -> Eligibility {
    let (posture, transcript, request) = probe_inputs();
    let candidates = thinking.iter().cloned().map(Some).chain([None]);
    let mut last = None;
    for extra in candidates {
        let mut params = JudgeParams {
            extra,
            ..base.clone()
        };
        loop {
            let review = review(model, &params, &posture, &transcript, &request, cancel).await;
            match review.outcome {
                ReviewOutcome::Allow { .. } | ReviewOutcome::Block { .. } => {
                    params.max_tokens = review.max_tokens;
                    return Eligibility::Eligible(params);
                }
                ReviewOutcome::Cancelled => {
                    return Eligibility::Ineligible {
                        reason: "The capability check was cancelled.".to_string(),
                    };
                }
                ReviewOutcome::Failed {
                    failure: ReviewFailure::Unavailable(_),
                } if params.temperature.is_some() => params.temperature = None,
                ReviewOutcome::Failed { failure } => {
                    last = Some(failure);
                    break;
                }
            }
        }
    }
    Eligibility::Ineligible {
        reason: match last {
            Some(ReviewFailure::Unavailable(detail)) => {
                format!("The model could not be reached for review: {detail}")
            }
            Some(ReviewFailure::Truncated(_)) => format!(
                "The model ran out of output budget before giving a verdict, even with \
                 {ESCALATED_OUTPUT_TOKENS} tokens."
            ),
            Some(ReviewFailure::Unparseable(_)) | None => {
                "The model did not return a structured verdict.".to_string()
            }
        },
    }
}

/// Probe results, cached until the model or provider changes or the app
/// restarts (the cache is in memory). Never probed per call: the host
/// probes at model selection, model switch, classifier-model change, and
/// app launch, and the gate only reads.
#[derive(Debug, Default)]
pub struct EligibilityCache {
    entries: Mutex<HashMap<String, Eligibility>>,
}

impl EligibilityCache {
    /// Cache key for a model: provider, model, and endpoint.
    pub fn key(provider: &str, model: &str, base_url: Option<&str>) -> String {
        format!("{provider}\u{1f}{model}\u{1f}{}", base_url.unwrap_or(""))
    }

    pub fn get(&self, key: &str) -> Option<Eligibility> {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned()
    }

    pub fn insert(&self, key: String, eligibility: Eligibility) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, eligibility);
    }

    /// Drop an entry so the next selection re-probes.
    pub fn invalidate(&self, key: &str) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judgment::verdict::{Authorization, RiskLevel};
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Scripted mock: each call pops the next response.
    enum Step {
        Reply(&'static str),
        Fail(JudgeError),
        Hang,
    }

    struct Mock {
        steps: Mutex<VecDeque<Step>>,
        calls: AtomicU32,
        seen: Mutex<Vec<(String, Option<f64>, u32)>>,
    }

    impl Mock {
        fn new(steps: Vec<Step>) -> Arc<Self> {
            Arc::new(Self {
                steps: Mutex::new(steps.into()),
                calls: AtomicU32::new(0),
                seen: Mutex::new(Vec::new()),
            })
        }
    }

    impl JudgeModel for Mock {
        fn label(&self) -> String {
            "mock-model".to_string()
        }
        fn complete<'a>(
            &'a self,
            call: JudgeCall<'a>,
        ) -> BoxFuture<'a, Result<String, JudgeError>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen.lock().unwrap().push((
                call.system.to_string(),
                call.temperature,
                call.max_tokens,
            ));
            let step = self.steps.lock().unwrap().pop_front();
            Box::pin(async move {
                match step {
                    Some(Step::Reply(text)) => Ok(text.to_string()),
                    Some(Step::Fail(err)) => Err(err),
                    Some(Step::Hang) | None => {
                        call.cancel.cancelled().await;
                        Err(JudgeError::Cancelled)
                    }
                }
            })
        }
    }

    const ALLOW: &str = r#"{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"Listing files was requested."}"#;
    const DENY: &str = r#"{"risk_level":"high","user_authorization":"unknown","outcome":"deny","rationale":"Uploads a private key."}"#;

    async fn run_mock(mock: &Mock, params: JudgeParams) -> Review {
        let params = &params;
        let (posture, transcript, request) = probe_inputs();
        review(
            mock,
            params,
            &posture,
            &transcript,
            &request,
            &CancellationToken::new(),
        )
        .await
    }

    #[tokio::test]
    async fn allow() {
        let mock = Mock::new(vec![Step::Reply(ALLOW)]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        let ReviewOutcome::Allow { verdict } = &review.outcome else {
            panic!("{review:?}");
        };
        assert_eq!(verdict.risk_level, RiskLevel::Low);
        assert_eq!(verdict.user_authorization, Authorization::High);
        assert_eq!(review.attempts, 1);
        assert_eq!(review.model, "mock-model");
        assert_eq!(review.outcome.model_reason(), None);
        // Temperature 0, bounded output, the bundled ruleset as system.
        let seen = mock.seen.lock().unwrap();
        assert_eq!(seen[0].0, ruleset::SYSTEM_PROMPT);
        assert_eq!(seen[0].1, Some(0.0));
        assert_eq!(seen[0].2, MAX_OUTPUT_TOKENS);
    }

    #[tokio::test]
    async fn deny_carries_reason_and_anti_workaround_guidance() {
        let mock = Mock::new(vec![Step::Reply(DENY)]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert!(matches!(review.outcome, ReviewOutcome::Block { .. }));
        let reason = review.outcome.model_reason().unwrap();
        assert!(reason.contains("Uploads a private key."));
        assert!(reason.contains("Do not tunnel the same action"));
    }

    #[tokio::test]
    async fn unparseable_is_a_review_failure_without_retry() {
        let mock = Mock::new(vec![Step::Reply("Looks fine to me!"), Step::Reply(ALLOW)]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert!(matches!(
            review.outcome,
            ReviewOutcome::Failed {
                failure: ReviewFailure::Unparseable(_)
            }
        ));
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
        let reason = review.outcome.model_reason().unwrap();
        assert!(reason.contains("This is a review failure, not a determination"));
        assert!(reason.contains("A retry was already attempted"));
    }

    #[tokio::test]
    async fn transient_error_retries_once() {
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Transient("429 rate limited".into())),
            Step::Reply(ALLOW),
        ]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert!(matches!(review.outcome, ReviewOutcome::Allow { .. }));
        assert_eq!(review.attempts, 2);

        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Transient("503".into())),
            Step::Fail(JudgeError::Transient("503 again".into())),
            Step::Reply(ALLOW),
        ]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert_eq!(
            review.outcome,
            ReviewOutcome::Failed {
                failure: ReviewFailure::Unavailable("503 again".into())
            }
        );
        assert_eq!(mock.calls.load(Ordering::SeqCst), 2, "retry count <= 1");
    }

    #[tokio::test]
    async fn permanent_error_does_not_retry() {
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Permanent("401 invalid key".into())),
            Step::Reply(ALLOW),
        ]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert!(matches!(review.outcome, ReviewOutcome::Failed { .. }));
        assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn timeout_counts_as_transient_and_is_retried_once() {
        assert_eq!(JudgeParams::default().attempt_timeout, ATTEMPT_TIMEOUT);
        let mock = Mock::new(vec![Step::Hang, Step::Hang]);
        let params = JudgeParams {
            attempt_timeout: Duration::from_millis(20),
            ..JudgeParams::default()
        };
        let review = run_mock(&mock, params).await;
        let ReviewOutcome::Failed {
            failure: ReviewFailure::Unavailable(detail),
        } = &review.outcome
        else {
            panic!("{review:?}");
        };
        assert!(detail.contains("timed out"));
        assert_eq!(review.attempts, 2);
    }

    #[tokio::test]
    async fn cancel_aborts_an_in_flight_call() {
        let mock = Mock::new(vec![Step::Hang]);
        let cancel = CancellationToken::new();
        let (posture, transcript, request) = probe_inputs();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            trigger.cancel();
        });
        let review = review(
            mock.as_ref(),
            &JudgeParams::default(),
            &posture,
            &transcript,
            &request,
            &cancel,
        )
        .await;
        assert_eq!(review.outcome, ReviewOutcome::Cancelled);
        assert_eq!(review.attempts, 1);
    }

    #[test]
    fn no_failure_path_produces_an_allow() {
        // Allow is only constructible from a parsed allow verdict; every
        // failure wording is a denial text.
        for failure in [
            ReviewFailure::Unavailable("x".into()),
            ReviewFailure::Unparseable("y".into()),
        ] {
            let outcome = ReviewOutcome::Failed { failure };
            assert!(outcome.model_reason().is_some());
            assert!(outcome.verdict().is_none());
        }
        assert!(ReviewOutcome::Cancelled.model_reason().is_some());
    }

    fn blocked() -> ReviewOutcome {
        ReviewOutcome::Block {
            verdict: verdict::parse_verdict(DENY).unwrap(),
        }
    }

    fn allowed() -> ReviewOutcome {
        ReviewOutcome::Allow {
            verdict: verdict::parse_verdict(ALLOW).unwrap(),
        }
    }

    #[test]
    fn consecutive_denials_suggest_once_per_conversation() {
        let counters = DenialCounters::default();
        assert!(!counters.record("c1", &blocked()));
        assert!(!counters.record("c1", &blocked()));
        assert!(counters.record("c1", &blocked()));
        // Never again in this conversation.
        for _ in 0..30 {
            assert!(!counters.record("c1", &blocked()));
        }
        // Other conversations count separately.
        assert!(!counters.record("c2", &blocked()));
    }

    #[test]
    fn an_allow_resets_the_consecutive_count() {
        let counters = DenialCounters::default();
        counters.record("c", &blocked());
        counters.record("c", &blocked());
        counters.record("c", &allowed());
        assert!(!counters.record("c", &blocked()));
        assert!(!counters.record("c", &blocked()));
        assert!(counters.record("c", &blocked()));
    }

    #[test]
    fn total_denials_suggest_at_twenty() {
        let counters = DenialCounters::default();
        let failed = ReviewOutcome::Failed {
            failure: ReviewFailure::Unavailable("x".into()),
        };
        // Deny, fail, allow, ...: never three denials in a row, so only
        // the total limit can fire.
        let mut denials = 0;
        let mut fired_at = None;
        while denials < 25 {
            for outcome in [&blocked(), &failed] {
                denials += 1;
                if counters.record("c", outcome) {
                    assert!(fired_at.is_none(), "fires once");
                    fired_at = Some(denials);
                }
            }
            assert!(!counters.record("c", &allowed()));
        }
        assert_eq!(fired_at, Some(TOTAL_DENIAL_LIMIT));
        assert!(!counters.record("c", &ReviewOutcome::Cancelled));
    }

    #[tokio::test]
    async fn probe_accepts_any_parseable_verdict() {
        for reply in [ALLOW, DENY] {
            let mock = Mock::new(vec![Step::Reply(reply)]);
            assert_eq!(
                probe(mock.as_ref(), &CancellationToken::new()).await,
                Eligibility::Eligible(JudgeParams::default())
            );
        }
    }

    #[tokio::test]
    async fn probe_drops_temperature_when_the_model_rejects_it() {
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Permanent(
                "400 Unsupported value: 'temperature'".into(),
            )),
            Step::Reply(ALLOW),
        ]);
        let Eligibility::Eligible(params) = probe(mock.as_ref(), &CancellationToken::new()).await
        else {
            panic!("should be eligible");
        };
        assert_eq!(params.temperature, None);
        let seen = mock.seen.lock().unwrap();
        assert_eq!(seen[1].1, None);
    }

    fn effort(level: &str) -> serde_json::Value {
        serde_json::json!({ "reasoning_effort": level })
    }

    #[tokio::test]
    async fn probe_keeps_the_first_thinking_control_the_model_accepts() {
        let candidates = [effort("none"), effort("low")];
        // `none` rejected (also without temperature), `low` accepted.
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Permanent(
                "400 reasoning_effort `none` is not supported".into(),
            )),
            Step::Fail(JudgeError::Permanent(
                "400 reasoning_effort `none` is not supported".into(),
            )),
            Step::Reply(ALLOW),
        ]);
        let Eligibility::Eligible(params) = probe_with(
            mock.as_ref(),
            JudgeParams::default(),
            &candidates,
            &CancellationToken::new(),
        )
        .await
        else {
            panic!("should be eligible");
        };
        assert_eq!(params.extra, Some(effort("low")));
        // Temperature comes back for the next candidate.
        assert_eq!(params.temperature, Some(0.0));
        assert_eq!(params.max_tokens, MAX_OUTPUT_TOKENS);
        assert_eq!(mock.calls.load(Ordering::SeqCst), 3);

        // Every control rejected: the probe ends without one.
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Permanent("400 unknown field".into())),
            Step::Fail(JudgeError::Permanent("400 unknown field".into())),
            Step::Reply(DENY),
        ]);
        let Eligibility::Eligible(params) = probe_with(
            mock.as_ref(),
            JudgeParams::default(),
            &[effort("none")],
            &CancellationToken::new(),
        )
        .await
        else {
            panic!("should be eligible");
        };
        assert_eq!(params.extra, None);
    }

    #[tokio::test]
    async fn running_out_of_budget_is_not_a_rejection() {
        // Out of budget at 256, verdict at the larger bound: the control
        // stays, and the probe keeps the larger bound.
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Truncated("finish_reason=Length".into())),
            Step::Reply(ALLOW),
        ]);
        let Eligibility::Eligible(params) = probe_with(
            mock.as_ref(),
            JudgeParams::default(),
            &[effort("low")],
            &CancellationToken::new(),
        )
        .await
        else {
            panic!("should be eligible");
        };
        assert_eq!(params.extra, Some(effort("low")));
        assert_eq!(params.max_tokens, ESCALATED_OUTPUT_TOKENS);
        let seen: Vec<u32> = mock.seen.lock().unwrap().iter().map(|s| s.2).collect();
        assert_eq!(seen, vec![MAX_OUTPUT_TOKENS, ESCALATED_OUTPUT_TOKENS]);

        // Out of budget even at the larger bound: the next control (which
        // may think less) gets its turn.
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Truncated("finish_reason=Length".into())),
            Step::Fail(JudgeError::Truncated("finish_reason=Length".into())),
            Step::Reply(ALLOW),
        ]);
        let Eligibility::Eligible(params) = probe_with(
            mock.as_ref(),
            JudgeParams::default(),
            &[effort("low"), effort("none")],
            &CancellationToken::new(),
        )
        .await
        else {
            panic!("should be eligible");
        };
        assert_eq!(params.extra, Some(effort("none")));
        assert_eq!(params.max_tokens, MAX_OUTPUT_TOKENS);
    }

    #[tokio::test]
    async fn probe_names_an_exhausted_budget() {
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Truncated("finish_reason=Length".into())),
            Step::Fail(JudgeError::Truncated("finish_reason=Length".into())),
        ]);
        let Eligibility::Ineligible { reason } =
            probe(mock.as_ref(), &CancellationToken::new()).await
        else {
            panic!("should be ineligible");
        };
        assert!(reason.contains("output budget"), "{reason}");
    }

    #[tokio::test]
    async fn a_verdict_cut_off_mid_object_gets_the_budget_retry() {
        let mock = Mock::new(vec![
            Step::Reply(r#"{"risk_level":"high","user_authorization":"low","outcome":"den"#),
            Step::Reply(DENY),
        ]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert!(matches!(review.outcome, ReviewOutcome::Block { .. }));
        assert_eq!(review.max_tokens, ESCALATED_OUTPUT_TOKENS);
        // Still cut off at the larger bound: truncated, not unparseable,
        // so the probe tries a control that thinks less.
        let mock = Mock::new(vec![
            Step::Reply("<think>Deleting a folder the user named"),
            Step::Reply("<think>Deleting a folder the user named, which"),
        ]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert!(matches!(
            review.outcome,
            ReviewOutcome::Failed {
                failure: ReviewFailure::Truncated(_)
            }
        ));
        // A complete reply without a verdict is not retried.
        let mock = Mock::new(vec![Step::Reply("Sure! The command lists files.")]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert_eq!(review.attempts, 1);
    }

    #[tokio::test]
    async fn a_budget_retry_is_separate_from_the_transient_retry() {
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Transient("503".into())),
            Step::Fail(JudgeError::Truncated("finish_reason=Length".into())),
            Step::Reply(DENY),
        ]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert!(matches!(review.outcome, ReviewOutcome::Block { .. }));
        assert_eq!(review.attempts, 3);
        assert_eq!(review.max_tokens, ESCALATED_OUTPUT_TOKENS);
        // A bound already at the larger value gets no budget retry.
        let mock = Mock::new(vec![Step::Fail(JudgeError::Truncated(
            "finish_reason=Length".into(),
        ))]);
        let review = run_mock(
            &mock,
            JudgeParams {
                max_tokens: ESCALATED_OUTPUT_TOKENS,
                ..JudgeParams::default()
            },
        )
        .await;
        assert_eq!(
            review.outcome,
            ReviewOutcome::Failed {
                failure: ReviewFailure::Truncated("finish_reason=Length".into())
            }
        );
        assert_eq!(review.attempts, 1);
    }

    #[tokio::test]
    async fn inline_thinking_is_not_the_verdict() {
        let mock = Mock::new(vec![Step::Reply(
            "<think>Maybe {\"outcome\":\"allow\"}? No, it uploads a key.</think>\n{\"risk_level\":\"high\",\"user_authorization\":\"unknown\",\"outcome\":\"deny\",\"rationale\":\"Uploads a key.\"}",
        )]);
        let review = run_mock(&mock, JudgeParams::default()).await;
        assert!(
            matches!(review.outcome, ReviewOutcome::Block { .. }),
            "{review:?}"
        );
    }

    #[tokio::test]
    async fn probe_rejects_models_without_structured_verdicts() {
        let mock = Mock::new(vec![Step::Reply("Sure! The command lists files.")]);
        let Eligibility::Ineligible { reason } =
            probe(mock.as_ref(), &CancellationToken::new()).await
        else {
            panic!("should be ineligible");
        };
        assert!(reason.contains("structured verdict"));
        // Offline: ineligible, with the reason.
        let mock = Mock::new(vec![
            Step::Fail(JudgeError::Transient("connection refused".into())),
            Step::Fail(JudgeError::Transient("connection refused".into())),
            Step::Fail(JudgeError::Transient("connection refused".into())),
            Step::Fail(JudgeError::Transient("connection refused".into())),
        ]);
        assert!(matches!(
            probe(mock.as_ref(), &CancellationToken::new()).await,
            Eligibility::Ineligible { .. }
        ));
    }

    #[test]
    fn eligibility_cache_holds_failures_until_invalidated() {
        let cache = EligibilityCache::default();
        let key = EligibilityCache::key("openai", "gpt-x", None);
        assert_eq!(cache.get(&key), None);
        cache.insert(
            key.clone(),
            Eligibility::Ineligible {
                reason: "offline".into(),
            },
        );
        // Cached: a probe failure stays ineligible, no re-probe per call.
        assert!(matches!(
            cache.get(&key),
            Some(Eligibility::Ineligible { .. })
        ));
        assert_ne!(key, EligibilityCache::key("openai", "gpt-y", None));
        assert_ne!(
            key,
            EligibilityCache::key("openai", "gpt-x", Some("https://proxy"))
        );
        cache.invalidate(&key);
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn auto_resolution_matrix() {
        let eligible = Eligibility::Eligible(JudgeParams::default());
        let ineligible = Eligibility::Ineligible { reason: "x".into() };
        assert_eq!(resolve_auto(Some(&eligible), false), AutoResolution::Judged);
        assert_eq!(resolve_auto(Some(&eligible), true), AutoResolution::Judged);
        assert_eq!(
            resolve_auto(Some(&ineligible), true),
            AutoResolution::SandboxPrompts
        );
        assert_eq!(resolve_auto(Some(&ineligible), false), AutoResolution::Ask);
        assert_eq!(resolve_auto(None, false), AutoResolution::Ask);
        assert_eq!(resolve_auto(None, true), AutoResolution::SandboxPrompts);
    }
}
