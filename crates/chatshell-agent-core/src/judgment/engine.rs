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

/// One completion call.
#[derive(Debug, Clone)]
pub struct JudgeCall<'a> {
    pub system: &'a str,
    pub user: &'a str,
    /// `Some(0.0)` by default; `None` for models that reject the
    /// parameter (found by the eligibility probe).
    pub temperature: Option<f64>,
    pub max_tokens: u32,
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
    #[error("cancelled")]
    Cancelled,
}

/// The host's model call. `label` names the model in the decision log.
pub trait JudgeModel: Send + Sync {
    fn label(&self) -> String;
    fn complete<'a>(&'a self, call: JudgeCall<'a>) -> BoxFuture<'a, Result<String, JudgeError>>;
}

/// Call parameters that worked for a model (found by the probe).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JudgeParams {
    pub temperature: Option<f64>,
    pub max_tokens: u32,
    pub attempt_timeout: Duration,
}

impl Default for JudgeParams {
    fn default() -> Self {
        Self {
            temperature: Some(0.0),
            max_tokens: MAX_OUTPUT_TOKENS,
            attempt_timeout: ATTEMPT_TIMEOUT,
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
}

impl ReviewFailure {
    fn detail(&self) -> &str {
        match self {
            ReviewFailure::Unavailable(detail) | ReviewFailure::Unparseable(detail) => detail,
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
    /// Calls made (1, or 2 after a transient failure).
    pub attempts: u32,
    pub duration: Duration,
    pub model: String,
}

/// Run one review.
pub async fn review(
    model: &dyn JudgeModel,
    params: JudgeParams,
    posture: &Posture,
    transcript: &AssembledTranscript,
    request: &JudgmentRequest,
    cancel: &CancellationToken,
) -> Review {
    let started = Instant::now();
    let user = ruleset::render_request(posture, transcript, request);
    let (outcome, attempts) = run(model, params, &user, cancel).await;
    Review {
        outcome,
        attempts,
        duration: started.elapsed(),
        model: model.label(),
    }
}

async fn run(
    model: &dyn JudgeModel,
    params: JudgeParams,
    user: &str,
    cancel: &CancellationToken,
) -> (ReviewOutcome, u32) {
    let mut attempts = 0;
    let mut last_error = String::new();
    // One call plus at most one retry, on transient failures only.
    while attempts < 2 {
        attempts += 1;
        match attempt(model, params, user, cancel).await {
            Ok(text) => {
                let outcome = match verdict::parse_verdict(&text) {
                    Ok(verdict) if verdict.outcome == Outcome::Allow => {
                        ReviewOutcome::Allow { verdict }
                    }
                    Ok(verdict) => ReviewOutcome::Block { verdict },
                    Err(err) => ReviewOutcome::Failed {
                        failure: ReviewFailure::Unparseable(err.to_string()),
                    },
                };
                return (outcome, attempts);
            }
            Err(JudgeError::Cancelled) => return (ReviewOutcome::Cancelled, attempts),
            Err(JudgeError::Permanent(detail)) => {
                return (
                    ReviewOutcome::Failed {
                        failure: ReviewFailure::Unavailable(detail),
                    },
                    attempts,
                );
            }
            Err(JudgeError::Transient(detail)) => last_error = detail,
        }
    }
    (
        ReviewOutcome::Failed {
            failure: ReviewFailure::Unavailable(last_error),
        },
        attempts,
    )
}

async fn attempt(
    model: &dyn JudgeModel,
    params: JudgeParams,
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
        max_tokens: params.max_tokens,
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

/// The probe's canned action: a clearly benign, clearly authorized call
/// whose judgment exercises the whole contract.
fn probe_inputs() -> (Posture, AssembledTranscript, JudgmentRequest) {
    use super::{Evidence, JudgmentSource, TranscriptBudget, TranscriptEntry};
    use crate::permission::GateContext;
    let ctx = GateContext {
        workspace: Some("/workspace/demo".into()),
        sandbox_available: false,
        network_allowed: false,
        scratch_dir: None,
    };
    let args = serde_json::json!({ "command": "ls -la" });
    let request = JudgmentRequest {
        subject: "ls -la".to_string(),
        source: JudgmentSource::UnsandboxedBash,
        forced: None,
        evidence: Evidence::collect("bash", &args, &ctx),
    };
    let transcript = super::assemble(
        &[TranscriptEntry::User {
            text: "List the files in this folder.".to_string(),
        }],
        TranscriptBudget::default(),
    );
    let posture = Posture {
        os: std::env::consts::OS.to_string(),
        workspace: Some("/workspace/demo".to_string()),
        jail: super::JailProfile::Absent,
        mcp_servers: Vec::new(),
    };
    (posture, transcript, request)
}

/// Setup-time capability probe: one judgment of a canned benign action.
/// Eligible when the model returns a parseable verdict. A permanent error
/// with temperature set is retried once without it (some models reject
/// the parameter); the parameters that worked are returned for reuse.
/// Tests capability, not names: any model that passes is eligible.
pub async fn probe(model: &dyn JudgeModel, cancel: &CancellationToken) -> Eligibility {
    let (posture, transcript, request) = probe_inputs();
    let mut params = JudgeParams::default();
    let mut tried_without_temperature = false;
    loop {
        let review = review(model, params, &posture, &transcript, &request, cancel).await;
        match review.outcome {
            ReviewOutcome::Allow { .. } | ReviewOutcome::Block { .. } => {
                return Eligibility::Eligible(params);
            }
            ReviewOutcome::Failed {
                failure: ReviewFailure::Unavailable(_),
            } if params.temperature.is_some() && !tried_without_temperature => {
                tried_without_temperature = true;
                params.temperature = None;
            }
            ReviewOutcome::Failed { failure } => {
                return Eligibility::Ineligible {
                    reason: match failure {
                        ReviewFailure::Unavailable(detail) => {
                            format!("The model could not be reached for review: {detail}")
                        }
                        ReviewFailure::Unparseable(_) => {
                            "The model did not return a structured verdict.".to_string()
                        }
                    },
                };
            }
            ReviewOutcome::Cancelled => {
                return Eligibility::Ineligible {
                    reason: "The capability check was cancelled.".to_string(),
                };
            }
        }
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
            JudgeParams::default(),
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
