//! Host-side pieces of the auto-mode judge every client shares.
//!
//! The engine ([`super::engine`]) is provider-agnostic; this module
//! supplies what each host would otherwise rebuild: a [`JudgeModel`] over
//! the core's own provider routing ([`GenerationJudge`]), provider-error
//! classification, the per-wire thinking controls the capability probe
//! tries, and a single-flight eligibility store. Desktop keeps its own
//! [`JudgeModel`] (its chat agent builder) and reuses the rest; the mobile
//! clients use all of it through the FFI.

use super::engine::{self, Eligibility, EligibilityCache, JudgeCall, JudgeError, JudgeModel};
use crate::types::{
    AgentError, AgentEvent, AgentRequest, ChatMessage, MessageRole, ProviderConfig,
    SamplingParameters,
};
use futures::future::BoxFuture;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Sort a provider error into retry-worthy or not. A reply that ran out
/// of output budget is [`JudgeError::Truncated`]; rate limits, overload,
/// 5xx, timeouts, and network failures are transient; everything else
/// (auth, bad request, unsupported parameter, unknown model) is
/// permanent.
pub fn classify_provider_error(message: &str) -> JudgeError {
    let lower = message.to_ascii_lowercase();
    // rig ends a run that delivered no answer with this finish reason
    // (reasoning used the whole output budget): worth a retry with a
    // larger bound, and never a sign that a request field was rejected.
    if lower.contains("finish_reason=length") {
        return JudgeError::Truncated(message.to_string());
    }
    // Status codes match as whole numbers only, so a limit or a count in
    // a permanent error ("at most 15000 tokens") never reads as a 500.
    let transient_codes = ["429", "500", "502", "503", "504", "529"];
    if lower
        .split(|c: char| !c.is_ascii_digit())
        .any(|token| transient_codes.contains(&token))
    {
        return JudgeError::Transient(message.to_string());
    }
    let transient_markers = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "overloaded",
        "internal server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "timed out",
        "timeout",
        "connection",
        "connect error",
        "dns",
        "network",
        "temporarily",
    ];
    if transient_markers
        .iter()
        .any(|marker| lower.contains(marker))
    {
        JudgeError::Transient(message.to_string())
    } else {
        JudgeError::Permanent(message.to_string())
    }
}

/// Request fields that reduce a judge model's reasoning, most reducing
/// first, for the probe to try in order (see [`engine::probe_with`]).
/// Chosen by wire, never by vendor: which value a model accepts varies by
/// model and endpoint (2026-11-06 eval: Volcengine GLM 5.3 takes `low` but
/// not `none`, its Flash model takes `minimal`, GPT-6 Luna takes `none`
/// and `low` but not `minimal`), so the probe finds out and drops what a
/// provider rejects. Wires without a known control get none; the probe's
/// larger output bound covers them.
pub fn thinking_candidates(provider_type: &str, api_style: Option<&str>) -> Vec<Value> {
    use crate::provider_catalog::{self, Wire};
    const LEVELS: [&str; 3] = ["none", "minimal", "low"];
    let chat = || {
        LEVELS
            .iter()
            .map(|level| serde_json::json!({ "reasoning_effort": level }))
            .collect()
    };
    let responses = || {
        LEVELS
            .iter()
            .map(|level| serde_json::json!({ "reasoning": { "effort": level } }))
            .collect()
    };
    match provider_catalog::resolve(provider_type).map(|route| route.wire) {
        // Custom OpenAI-compatible providers pick their wire by API style.
        Some(Wire::Responses) if api_style == Some("chat_completions") => chat(),
        Some(Wire::Responses) => responses(),
        Some(Wire::OpenAiCompat | Wire::OpenRouter | Wire::Azure) => chat(),
        // `disabled` for models that allow it; `between_tools` is how
        // Claude Sonnet 5.5 turns thinking off (it rejects `disabled`);
        // models whose thinking can't be turned off (Claude Opus 5.5, the
        // Fable line) keep it with no visible output, so the judge never
        // falls through to the chat defaults that request summaries.
        Some(Wire::Anthropic) => vec![
            serde_json::json!({ "thinking": { "type": "disabled" } }),
            serde_json::json!({ "thinking": { "type": "between_tools" } }),
            serde_json::json!({ "thinking": { "type": "adaptive", "display": "omitted" } }),
        ],
        // Ollama's native API; rig lifts `think` out of the extra fields.
        Some(Wire::Ollama) => vec![serde_json::json!({ "think": false })],
        _ => Vec::new(),
    }
}

/// Probe results plus one in-flight capability check per judge, so
/// concurrent callers (the input bar's status and a turn often ask about
/// the same model at once) share a single probe.
#[derive(Debug, Default)]
pub struct EligibilityStore {
    cache: EligibilityCache,
    probes: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl EligibilityStore {
    pub fn cache(&self) -> &EligibilityCache {
        &self.cache
    }

    /// The cached eligibility for `key`, else the result of `probe`, run by
    /// at most one caller at a time per key: a caller that finds a probe in
    /// flight waits for it and reads its cached result instead of probing
    /// again. A cancelled probe is not cached (the next waiter probes); a
    /// caller cancelled while waiting returns without probing.
    pub async fn get_or_probe<F, Fut>(
        &self,
        key: &str,
        cancel: &CancellationToken,
        probe: F,
    ) -> Eligibility
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Eligibility>,
    {
        if let Some(cached) = self.cache.get(key) {
            return cached;
        }
        let lock = self
            .probes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(key.to_string())
            .or_default()
            .clone();
        let _guard = tokio::select! {
            guard = lock.lock() => guard,
            _ = cancel.cancelled() => {
                return Eligibility::Ineligible {
                    reason: "The capability check was cancelled.".to_string(),
                    cause: engine::IneligibleCause::Cancelled,
                };
            }
        };
        // The probe this caller waited for may have settled it.
        if let Some(cached) = self.cache.get(key) {
            return cached;
        }
        let eligibility = probe().await;
        if !cancel.is_cancelled() {
            self.cache.insert(key.to_string(), eligibility.clone());
        }
        eligibility
    }

    /// Keep a larger output bound for a model after a review needed it, so
    /// its later reviews start there instead of running out first.
    pub fn raise_budget(&self, key: &str, max_tokens: u32) -> bool {
        if let Some(Eligibility::Eligible(mut params)) = self.cache.get(key)
            && params.max_tokens < max_tokens
        {
            params.max_tokens = max_tokens;
            self.cache
                .insert(key.to_string(), Eligibility::Eligible(params));
            return true;
        }
        false
    }
}

/// A judge reached through the core's provider routing
/// ([`crate::run_generation`]): one non-tool turn with the ruleset as the
/// system prompt.
#[derive(Debug, Clone)]
pub struct GenerationJudge {
    /// Provider id (`openai`, `custom_anthropic`, ...) for labels, the
    /// cache key, and the thinking candidates.
    pub provider_type: String,
    pub provider: ProviderConfig,
    pub model: String,
}

impl GenerationJudge {
    /// The eligibility cache key for this model.
    pub fn cache_key(&self) -> String {
        EligibilityCache::key(
            &self.provider_type,
            &self.model,
            self.provider.base_url.as_deref(),
        )
    }

    /// The reasoning controls the capability probe tries for this model.
    pub fn thinking_candidates(&self) -> Vec<Value> {
        thinking_candidates(&self.provider_type, self.provider.api_style.as_deref())
    }

    /// The request of one judge call.
    fn request(&self, call: &JudgeCall<'_>) -> AgentRequest {
        let message = |role, content: &str| ChatMessage {
            role,
            content: content.to_string(),
            reasoning_content: None,
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_result_call_id: None,
        };
        AgentRequest {
            provider: self.provider.clone(),
            model: self.model.clone(),
            messages: vec![
                message(MessageRole::System, call.system),
                message(MessageRole::User, call.user),
            ],
            parameters: SamplingParameters {
                temperature: call.temperature,
                max_tokens: Some(u64::from(call.max_tokens)),
                additional_params_json: call.extra.map(Value::to_string),
            },
            tools: Vec::new(),
        }
    }
}

impl JudgeModel for GenerationJudge {
    fn label(&self) -> String {
        format!("{}/{}", self.provider_type, self.model)
    }

    fn complete<'a>(&'a self, call: JudgeCall<'a>) -> BoxFuture<'a, Result<String, JudgeError>> {
        Box::pin(async move {
            let request = self.request(&call);
            let cancel = call.cancel.child_token();
            let (sender, mut receiver) = mpsc::channel(64);
            let generation = crate::run_generation(request, sender, cancel.clone());
            let collect = async {
                let mut text = String::new();
                while let Some(event) = receiver.recv().await {
                    match event {
                        Ok(AgentEvent::TextDelta { text: delta }) => text.push_str(&delta),
                        Ok(AgentEvent::Completed {
                            content,
                            cancelled,
                            output_limited,
                            ..
                        }) => {
                            if cancelled {
                                return Err(JudgeError::Cancelled);
                            }
                            let content = if content.is_empty() { text } else { content };
                            // A reply cut at the bound may hold a cut verdict.
                            if output_limited {
                                return Err(JudgeError::Truncated(format!(
                                    "finish_reason=length after {} bytes",
                                    content.len()
                                )));
                            }
                            return Ok(content);
                        }
                        Ok(_) => {}
                        Err(error) => return Err(judge_error(error)),
                    }
                }
                Err(JudgeError::Transient(
                    "the provider stream ended without a reply".to_string(),
                ))
            };
            let (result, outcome) = tokio::join!(generation, collect);
            if call.cancel.is_cancelled() {
                return Err(JudgeError::Cancelled);
            }
            match (result, outcome) {
                (Err(error), Err(JudgeError::Transient(_))) => Err(judge_error(error)),
                (_, outcome) => outcome,
            }
        })
    }
}

/// What `auto` decided for one tool call.
#[derive(Debug, Clone, PartialEq)]
pub enum JudgedCall {
    /// The gate allows the call without a judgment (read tier, a
    /// human-minted grant).
    Allowed,
    /// The gate refuses the call outright; `reason` goes to the model.
    Denied { reason: String },
    /// The gate wants a human (never produced in `auto` today; kept so a
    /// host fails safe to its prompt should that change).
    Prompt { subject: String },
    /// The judge reviewed the call.
    Reviewed {
        subject: String,
        review: engine::Review,
    },
}

/// Judge one tool call in `auto`: the gate first (its hard denies and
/// silent allows stand), then the judge for what the gate escalates.
/// `history` is the transcript before this call (user messages and tool
/// inputs only, oldest first); the call's own input is never part of it.
#[allow(clippy::too_many_arguments)]
pub async fn judge_call(
    judge: &dyn JudgeModel,
    params: &engine::JudgeParams,
    posture: &super::ruleset::Posture,
    context: &crate::permission::GateContext,
    history: &[super::TranscriptEntry],
    tool_name: &str,
    args: &Value,
    grants: &crate::permission::SessionGrants,
    cancel: &CancellationToken,
) -> JudgedCall {
    use crate::permission::{GateDecision, PermissionMode, resolve_in};
    match resolve_in(PermissionMode::Auto, context, tool_name, args, grants) {
        GateDecision::Allow => JudgedCall::Allowed,
        GateDecision::Deny { reason } => JudgedCall::Denied { reason },
        GateDecision::Prompt { subject, .. } => JudgedCall::Prompt { subject },
        GateDecision::Classify(request) => {
            let transcript =
                super::transcript::assemble(history, super::TranscriptBudget::default());
            let review =
                engine::review(judge, params, posture, &transcript, &request, cancel).await;
            JudgedCall::Reviewed {
                subject: request.subject.clone(),
                review,
            }
        }
    }
}

fn judge_error(error: AgentError) -> JudgeError {
    match error {
        AgentError::Cancelled => JudgeError::Cancelled,
        AgentError::Provider(message) => classify_provider_error(&message),
        other => JudgeError::Permanent(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn provider_errors_are_classified() {
        for transient in [
            "HTTP 429 Too Many Requests",
            "Provider returned 503 Service Unavailable",
            "anthropic: overloaded_error",
            "request timed out",
            "error sending request: connection refused",
            "rate_limit_error: slow down",
        ] {
            assert!(
                matches!(classify_provider_error(transient), JudgeError::Transient(_)),
                "{transient}"
            );
        }
        for permanent in [
            "401 Unauthorized: invalid x-api-key",
            "400 Unsupported value: 'temperature' does not support 0",
            "model not found: gpt-nope",
            "400 max_tokens must be at most 15000",
            "400 prompt is 4500 tokens over the limit for model-5029",
        ] {
            assert!(
                matches!(classify_provider_error(permanent), JudgeError::Permanent(_)),
                "{permanent}"
            );
        }
    }

    #[test]
    fn running_out_of_budget_is_classified_as_truncated() {
        let message = "PromptError: CompletionError: ResponseError: the model produced no \
                       answer and stopped with finish_reason=Length";
        assert!(matches!(
            classify_provider_error(message),
            JudgeError::Truncated(_)
        ));
    }

    #[test]
    fn thinking_candidates_follow_the_wire() {
        let efforts = |field: &dyn Fn(&str) -> Value| -> Vec<Value> {
            ["none", "minimal", "low"]
                .iter()
                .map(|l| field(l))
                .collect()
        };
        let chat = efforts(&|l| json!({ "reasoning_effort": l }));
        let responses = efforts(&|l| json!({ "reasoning": { "effort": l } }));
        assert_eq!(thinking_candidates("minimax_cn", None), chat);
        assert_eq!(thinking_candidates("openrouter", None), chat);
        assert_eq!(
            thinking_candidates("custom_openai", Some("chat_completions")),
            chat
        );
        assert_eq!(
            thinking_candidates("custom_openai", Some("responses")),
            responses
        );
        assert_eq!(thinking_candidates("openai", None), responses);
        assert_eq!(
            thinking_candidates("custom_anthropic", None),
            vec![
                json!({ "thinking": { "type": "disabled" } }),
                json!({ "thinking": { "type": "between_tools" } }),
                json!({ "thinking": { "type": "adaptive", "display": "omitted" } }),
            ]
        );
        assert_eq!(
            thinking_candidates("ollama", None),
            vec![json!({ "think": false })]
        );
        assert!(thinking_candidates("gemini", None).is_empty());
        assert!(thinking_candidates("not-a-provider", None).is_empty());
    }

    #[test]
    fn judge_requests_carry_the_ruleset_and_the_bounds() {
        let judge = GenerationJudge {
            provider_type: "openai".to_string(),
            provider: ProviderConfig {
                kind: crate::types::ProviderKind::OpenAi,
                api_key: Some("k".to_string()),
                base_url: None,
                api_style: None,
            },
            model: "gpt-x".to_string(),
        };
        let extra = json!({ "reasoning": { "effort": "low" } });
        let request = judge.request(&JudgeCall {
            system: "rules",
            user: "action",
            temperature: Some(0.0),
            max_tokens: 256,
            extra: Some(&extra),
            cancel: CancellationToken::new(),
        });
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, MessageRole::System);
        assert_eq!(request.messages[0].content, "rules");
        assert_eq!(request.messages[1].content, "action");
        assert_eq!(request.parameters.temperature, Some(0.0));
        assert_eq!(request.parameters.max_tokens, Some(256));
        assert_eq!(
            request.parameters.additional_params_json.as_deref(),
            Some(extra.to_string().as_str())
        );
        assert!(request.tools.is_empty());
        assert_eq!(judge.label(), "openai/gpt-x");
    }

    struct Fixed(&'static str);

    impl JudgeModel for Fixed {
        fn label(&self) -> String {
            "fixed".to_string()
        }
        fn complete<'a>(
            &'a self,
            _call: JudgeCall<'a>,
        ) -> BoxFuture<'a, Result<String, JudgeError>> {
            Box::pin(async move { Ok(self.0.to_string()) })
        }
    }

    fn mobile_posture() -> super::super::ruleset::Posture {
        super::super::ruleset::Posture {
            os: "ios".to_string(),
            workspace: None,
            jail: super::super::JailProfile::Absent,
            mcp_servers: vec!["notes".to_string()],
        }
    }

    async fn judge(model: &Fixed, tool: &str, args: Value) -> JudgedCall {
        let mut grants = crate::permission::SessionGrants::default();
        if tool == "granted" {
            grants.grant_server_tool("notes", "search");
        }
        judge_call(
            model,
            &engine::JudgeParams::default(),
            &mobile_posture(),
            &crate::permission::GateContext::default(),
            &[super::super::TranscriptEntry::User {
                text: "find my notes about rust".to_string(),
            }],
            if tool == "granted" {
                "mcp_tool_use"
            } else {
                tool
            },
            &args,
            &grants,
            &CancellationToken::new(),
        )
        .await
    }

    #[tokio::test]
    async fn mcp_calls_in_auto_go_to_the_judge() {
        const ALLOW: &str = r#"{"risk_level":"low","user_authorization":"high","outcome":"allow","rationale":"Requested."}"#;
        const DENY: &str = r#"{"risk_level":"high","user_authorization":"unknown","outcome":"deny","rationale":"Deletes notes."}"#;
        let args = json!({ "server": "notes", "tool": "search", "arguments": { "q": "rust" } });
        match judge(&Fixed(ALLOW), "mcp_tool_use", args.clone()).await {
            JudgedCall::Reviewed { review, .. } => {
                assert!(matches!(
                    review.outcome,
                    engine::ReviewOutcome::Allow { .. }
                ))
            }
            other => panic!("unexpected {other:?}"),
        }
        match judge(&Fixed(DENY), "mcp_tool_use", args.clone()).await {
            JudgedCall::Reviewed { review, .. } => {
                assert!(matches!(
                    review.outcome,
                    engine::ReviewOutcome::Block { .. }
                ));
                assert!(
                    review
                        .outcome
                        .model_reason()
                        .unwrap()
                        .contains("Deletes notes.")
                );
            }
            other => panic!("unexpected {other:?}"),
        }
        // A reply without a verdict never allows.
        match judge(&Fixed("sure"), "mcp_tool_use", args).await {
            JudgedCall::Reviewed { review, .. } => {
                assert!(matches!(
                    review.outcome,
                    engine::ReviewOutcome::Failed { .. }
                ))
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[tokio::test]
    async fn read_tier_calls_and_granted_mcp_calls_skip_the_judge() {
        let model = Fixed("never read");
        assert_eq!(
            judge(&model, "web_search", json!({ "query": "rust" })).await,
            JudgedCall::Allowed
        );
        let args = json!({ "server": "notes", "tool": "search", "arguments": {} });
        assert_eq!(judge(&model, "granted", args).await, JudgedCall::Allowed);
    }

    fn eligible() -> Eligibility {
        Eligibility::Eligible(engine::JudgeParams::default())
    }

    #[tokio::test]
    async fn concurrent_callers_share_one_probe() {
        let store = EligibilityStore::default();
        let probes = AtomicUsize::new(0);
        let cancel = CancellationToken::new();
        let probe = || async {
            probes.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            eligible()
        };
        let (a, b) = tokio::join!(
            store.get_or_probe("k", &cancel, probe),
            store.get_or_probe("k", &cancel, probe)
        );
        assert_eq!(a, eligible());
        assert_eq!(b, eligible());
        assert_eq!(probes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_cancelled_probe_is_not_cached() {
        let store = EligibilityStore::default();
        let cancel = CancellationToken::new();
        cancel.cancel();
        store
            .get_or_probe("k", &cancel, || async { eligible() })
            .await;
        assert!(store.cache().get("k").is_none());
    }

    #[test]
    fn a_raised_budget_is_kept_for_later_reviews() {
        let store = EligibilityStore::default();
        store.cache().insert("k".to_string(), eligible());
        assert!(store.raise_budget("k", engine::ESCALATED_OUTPUT_TOKENS));
        assert!(!store.raise_budget("k", engine::MAX_OUTPUT_TOKENS));
        match store.cache().get("k") {
            Some(Eligibility::Eligible(params)) => {
                assert_eq!(params.max_tokens, engine::ESCALATED_OUTPUT_TOKENS)
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
