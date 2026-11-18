//! Auto-mode judge for the mobile clients.
//!
//! The clients own the permission mode and the human prompt; for a call
//! in `auto` they ask [`auto_review_tool_call`], which runs the core gate
//! and, for what the gate escalates, the core judge
//! ([`core::judgment::host::judge_call`]). [`auto_review_eligibility`]
//! tells them how `auto` resolves for a model: judged when the model
//! passes the capability probe, else `ask`.
//!
//! Probe results, raised output bounds, and denial counters live in this
//! process for the run of the app, like desktop's.

use std::sync::LazyLock;

use chatshell_agent_core as core;
use core::judgment::engine::{
    DenialCounters, Eligibility, IneligibleCause, JudgeModel, ReviewOutcome,
};
use core::judgment::host::{EligibilityStore, GenerationJudge, JudgedCall, judge_call};
use core::judgment::ruleset::Posture;
use core::judgment::{JailProfile, TranscriptEntry};
use core::permission::{
    GateContext, GateDecision, McpToolHints, PermissionMode, SessionGrants, resolve_in,
};
use tokio_util::sync::CancellationToken;

use crate::{ProviderConfig, RUNTIME};

static ELIGIBILITY: LazyLock<EligibilityStore> = LazyLock::new(EligibilityStore::default);
static DENIALS: LazyLock<DenialCounters> = LazyLock::new(DenialCounters::default);

/// The model that judges: the dedicated auto-review model when one is
/// configured, else the conversation's model (the client decides).
#[derive(Debug, Clone, uniffi::Record)]
pub struct AutoReviewJudge {
    /// Provider id (`openai`, `custom_anthropic`, ...).
    pub provider_type: String,
    pub provider: ProviderConfig,
    pub model: String,
}

impl From<AutoReviewJudge> for GenerationJudge {
    fn from(value: AutoReviewJudge) -> Self {
        Self {
            provider_type: value.provider_type,
            provider: value.provider.into(),
            model: value.model,
        }
    }
}

/// How `auto` resolves for a judge model.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AutoReviewEligibility {
    /// The model passed the capability probe: `auto` is judged. Otherwise
    /// `auto` behaves as `ask`.
    pub eligible: bool,
    /// Why not, as a stable cause (`unauthorized`, `notFound`,
    /// `rateLimited`, `unreachable`, `rejected`, `outputBudget`,
    /// `noVerdict`, `cancelled`) for the UI to explain.
    pub cause: Option<String>,
    /// The technical detail (provider error text), for diagnostics only.
    pub reason: Option<String>,
    /// The judging model (`provider/model`).
    pub judge: String,
}

/// One entry of the judge transcript, oldest first: each user message's
/// own text and each earlier tool call's input. Never assistant text or
/// tool results.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum AutoReviewTranscriptEntry {
    User { text: String },
    ToolInput { tool: String, input_json: String },
}

impl From<AutoReviewTranscriptEntry> for TranscriptEntry {
    fn from(value: AutoReviewTranscriptEntry) -> Self {
        match value {
            AutoReviewTranscriptEntry::User { text } => TranscriptEntry::User { text },
            AutoReviewTranscriptEntry::ToolInput { tool, input_json } => {
                TranscriptEntry::ToolInput {
                    tool,
                    input: serde_json::from_str(&input_json).unwrap_or(serde_json::Value::Null),
                }
            }
        }
    }
}

/// What an MCP server declares about one of its tools, and whether the
/// user trusts that server's declarations (`trust_annotations`).
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpToolHintsEntry {
    pub server: String,
    pub tool: String,
    pub read_only: bool,
    pub destructive: bool,
    pub trusted: bool,
}

fn gate_context(hints: Vec<McpToolHintsEntry>) -> GateContext {
    GateContext {
        mcp_tool_hints: hints
            .into_iter()
            .filter(|h| h.read_only || h.destructive)
            .map(|h| {
                (
                    McpToolHints::key(&h.server, &h.tool),
                    McpToolHints {
                        read_only: h.read_only,
                        destructive: h.destructive,
                        trusted: h.trusted,
                    },
                )
            })
            .collect(),
        ..GateContext::default()
    }
}

/// The `ask`-mode decision for one MCP call (`mcp_tool_use` or
/// `mcp_resource_read`), from the core gate (the client keeps its own chat
/// grants for plain prompts).
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Enum)]
pub enum McpAskDecision {
    /// Run without asking (a trusted server declared the tool read-only).
    Allow,
    /// Ask as usual; a chat grant may cover it.
    Prompt,
    /// Ask even when a chat grant exists, and offer no grant; `detail`
    /// explains why (the server declared the tool destructive, or a
    /// `file://` resource is a sensitive path).
    ForcedPrompt { detail: String },
}

/// Resolve one MCP call in `ask` against the server-declared hints.
/// `tool_name` is `mcp_tool_use` (`{server, tool, arguments}`) or
/// `mcp_resource_read` (`{server, uri}`).
#[uniffi::export]
pub fn mcp_ask_decision(
    tool_name: String,
    arguments_json: String,
    hints: Vec<McpToolHintsEntry>,
) -> McpAskDecision {
    let args: serde_json::Value =
        serde_json::from_str(&arguments_json).unwrap_or(serde_json::Value::Null);
    match resolve_in(
        PermissionMode::Ask,
        &gate_context(hints),
        &tool_name,
        &args,
        &SessionGrants::default(),
    ) {
        GateDecision::Allow => McpAskDecision::Allow,
        GateDecision::Prompt {
            forced: Some(forced),
            ..
        } => McpAskDecision::ForcedPrompt {
            detail: forced.detail,
        },
        _ => McpAskDecision::Prompt,
    }
}

/// What to do with one call in `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum AutoReviewDecision {
    /// Run the call.
    Allow,
    /// Do not run it; send `model_reason` back as the tool result.
    Deny,
    /// No judgment is available (the model is not eligible): ask the
    /// user as `ask` would.
    Prompt,
}

/// The result of [`auto_review_tool_call`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct AutoReviewResult {
    pub decision: AutoReviewDecision,
    /// Tool-result text for the model when the call is not run.
    pub model_reason: Option<String>,
    /// The judge's one-line rationale, when it gave a verdict.
    pub rationale: Option<String>,
    /// `low` / `medium` / `high` / `critical`, when it gave a verdict.
    pub risk_level: Option<String>,
    /// How clearly the user asked for the action: `unknown` / `low` /
    /// `medium` / `high`, when it gave a verdict.
    pub user_authorization: Option<String>,
    /// The review produced no verdict (provider failure, unparseable or
    /// truncated reply); the call is denied.
    pub failed: bool,
    /// Show the one-time suggestion to switch the conversation to `ask`
    /// (repeated denials).
    pub suggest_ask: bool,
    /// Why `auto` fell back to a prompt (`Prompt` only), as in
    /// [`AutoReviewEligibility::cause`].
    pub ineligible_cause: Option<String>,
    /// The judging model (`provider/model`).
    pub judge: String,
    pub duration_ms: u64,
}

impl AutoReviewResult {
    fn plain(decision: AutoReviewDecision, judge: String) -> Self {
        Self {
            decision,
            model_reason: None,
            rationale: None,
            risk_level: None,
            user_authorization: None,
            failed: false,
            suggest_ask: false,
            ineligible_cause: None,
            judge,
            duration_ms: 0,
        }
    }
}

async fn eligibility(judge: &GenerationJudge, cancel: &CancellationToken) -> Eligibility {
    ELIGIBILITY
        .get_or_probe(&judge.cache_key(), cancel, || async {
            let candidates = judge.thinking_candidates();
            core::judgment::engine::probe_with(
                judge,
                core::judgment::engine::JudgeParams::default(),
                &candidates,
                cancel,
            )
            .await
        })
        .await
}

fn eligibility_record(judge: &GenerationJudge, eligibility: &Eligibility) -> AutoReviewEligibility {
    match eligibility {
        Eligibility::Eligible(_) => AutoReviewEligibility {
            eligible: true,
            cause: None,
            reason: None,
            judge: judge.label(),
        },
        Eligibility::Ineligible { reason, cause } => AutoReviewEligibility {
            eligible: false,
            cause: Some(cause.as_str().to_string()),
            reason: Some(reason.clone()),
            judge: judge.label(),
        },
    }
}

/// How `auto` resolves for `judge`: probed once per model for this run of
/// the app (concurrent callers share the probe). `recheck` drops the
/// cached result first (the user changed models or the judge setting).
#[uniffi::export]
pub async fn auto_review_eligibility(
    judge: AutoReviewJudge,
    recheck: bool,
) -> AutoReviewEligibility {
    let judge: GenerationJudge = judge.into();
    let (tx, rx) = tokio::sync::oneshot::channel();
    RUNTIME.spawn(async move {
        if recheck {
            ELIGIBILITY.cache().invalidate(&judge.cache_key());
        }
        let cancel = CancellationToken::new();
        let eligibility = eligibility(&judge, &cancel).await;
        let _ = tx.send(eligibility_record(&judge, &eligibility));
    });
    rx.await.unwrap_or_else(|_| AutoReviewEligibility {
        eligible: false,
        cause: Some(IneligibleCause::Cancelled.as_str().to_string()),
        reason: Some("The capability check stopped.".to_string()),
        judge: String::new(),
    })
}

/// Decide one tool call in `auto`. `history` is the transcript before this
/// call; `mcp_servers` names the servers enabled for the conversation;
/// `granted_server_tools` lists `server\u{1f}tool` pairs the user allowed
/// for this conversation (a human grant skips the judge); `mcp_tool_hints`
/// carries server-declared annotations (a destructive tool is judged past
/// grants; a trusted read-only one runs without a judgment).
#[uniffi::export]
#[allow(clippy::too_many_arguments)]
pub async fn auto_review_tool_call(
    conversation_id: String,
    judge: AutoReviewJudge,
    mcp_servers: Vec<String>,
    granted_server_tools: Vec<String>,
    mcp_tool_hints: Vec<McpToolHintsEntry>,
    history: Vec<AutoReviewTranscriptEntry>,
    tool_name: String,
    arguments_json: String,
) -> AutoReviewResult {
    let judge: GenerationJudge = judge.into();
    let label = judge.label();
    let (tx, rx) = tokio::sync::oneshot::channel();
    RUNTIME.spawn(async move {
        let result = review(
            &conversation_id,
            &judge,
            mcp_servers,
            &granted_server_tools,
            mcp_tool_hints,
            history,
            &tool_name,
            &arguments_json,
        )
        .await;
        let _ = tx.send(result);
    });
    rx.await.unwrap_or_else(|_| AutoReviewResult {
        model_reason: Some("Auto-review was cancelled.".to_string()),
        ..AutoReviewResult::plain(AutoReviewDecision::Deny, label)
    })
}

/// Forget a conversation's denial counters (conversation deleted).
#[uniffi::export]
pub fn forget_auto_review_conversation(conversation_id: String) {
    DENIALS.forget(&conversation_id);
}

#[allow(clippy::too_many_arguments)]
async fn review(
    conversation_id: &str,
    judge: &GenerationJudge,
    mcp_servers: Vec<String>,
    granted_server_tools: &[String],
    mcp_tool_hints: Vec<McpToolHintsEntry>,
    history: Vec<AutoReviewTranscriptEntry>,
    tool_name: &str,
    arguments_json: &str,
) -> AutoReviewResult {
    let label = judge.label();
    let cancel = CancellationToken::new();
    let params = match eligibility(judge, &cancel).await {
        Eligibility::Eligible(params) => params,
        Eligibility::Ineligible { cause, .. } => {
            return AutoReviewResult {
                ineligible_cause: Some(cause.as_str().to_string()),
                ..AutoReviewResult::plain(AutoReviewDecision::Prompt, label)
            };
        }
    };
    let args: serde_json::Value =
        serde_json::from_str(arguments_json).unwrap_or(serde_json::Value::Null);
    let mut grants = SessionGrants::default();
    for pair in granted_server_tools {
        if let Some((server, tool)) = pair.split_once('\u{1f}') {
            grants.grant_server_tool(server, tool);
        }
    }
    let posture = Posture {
        os: std::env::consts::OS.to_string(),
        workspace: None,
        jail: JailProfile::Absent,
        mcp_servers,
    };
    let history: Vec<TranscriptEntry> = history.into_iter().map(Into::into).collect();
    let judged = judge_call(
        judge,
        &params,
        &posture,
        &gate_context(mcp_tool_hints),
        &history,
        tool_name,
        &args,
        &grants,
        &cancel,
    )
    .await;
    let review = match judged {
        JudgedCall::Allowed => return AutoReviewResult::plain(AutoReviewDecision::Allow, label),
        JudgedCall::Denied { reason } => {
            return AutoReviewResult {
                model_reason: Some(reason),
                ..AutoReviewResult::plain(AutoReviewDecision::Deny, label)
            };
        }
        JudgedCall::Prompt { .. } => {
            return AutoReviewResult::plain(AutoReviewDecision::Prompt, label);
        }
        JudgedCall::Reviewed { review, .. } => review,
    };
    ELIGIBILITY.raise_budget(&judge.cache_key(), review.max_tokens);
    let suggest_ask = DENIALS.record(conversation_id, &review.outcome);
    let verdict = review.outcome.verdict();
    AutoReviewResult {
        decision: match review.outcome {
            ReviewOutcome::Allow { .. } => AutoReviewDecision::Allow,
            _ => AutoReviewDecision::Deny,
        },
        model_reason: review.outcome.model_reason(),
        rationale: verdict.map(|v| v.rationale.clone()),
        risk_level: verdict.and_then(|v| wire_name(v.risk_level)),
        user_authorization: verdict.and_then(|v| wire_name(v.user_authorization)),
        failed: matches!(review.outcome, ReviewOutcome::Failed { .. }),
        suggest_ask,
        ineligible_cause: None,
        judge: review.model,
        duration_ms: u64::try_from(review.duration.as_millis()).unwrap_or(u64::MAX),
    }
}

/// The serde wire name of a verdict enum (`high`, `unknown`, ...).
fn wire_name(value: impl serde::Serialize) -> Option<String> {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
}

#[cfg(test)]
mod mcp_ask_tests {
    use super::*;

    fn hints(read_only: bool, destructive: bool, trusted: bool) -> Vec<McpToolHintsEntry> {
        vec![McpToolHintsEntry {
            server: "srv".into(),
            tool: "t".into(),
            read_only,
            destructive,
            trusted,
        }]
    }

    const USE: &str = "mcp_tool_use";
    const CALL: &str = r#"{"server":"srv","tool":"t","arguments":{}}"#;

    #[test]
    fn resource_reads_run_unless_the_file_is_sensitive() {
        let read = "mcp_resource_read";
        assert_eq!(
            mcp_ask_decision(
                read.into(),
                r#"{"server":"s","uri":"mem://a"}"#.into(),
                vec![]
            ),
            McpAskDecision::Allow
        );
        let home = std::env::var("HOME").unwrap();
        let uri = format!("file://{home}/.zshrc");
        let args = serde_json::json!({"server": "s", "uri": uri}).to_string();
        assert!(matches!(
            mcp_ask_decision(read.into(), args, vec![]),
            McpAskDecision::ForcedPrompt { .. }
        ));
    }

    #[test]
    fn unannotated_prompts() {
        assert_eq!(
            mcp_ask_decision(USE.into(), CALL.into(), vec![]),
            McpAskDecision::Prompt
        );
    }

    #[test]
    fn trusted_read_only_runs() {
        assert_eq!(
            mcp_ask_decision(USE.into(), CALL.into(), hints(true, false, true)),
            McpAskDecision::Allow
        );
    }

    #[test]
    fn untrusted_read_only_prompts() {
        assert_eq!(
            mcp_ask_decision(USE.into(), CALL.into(), hints(true, false, false)),
            McpAskDecision::Prompt
        );
    }

    #[test]
    fn destructive_forces_a_prompt_even_when_trusted() {
        assert!(matches!(
            mcp_ask_decision(USE.into(), CALL.into(), hints(true, true, true)),
            McpAskDecision::ForcedPrompt { .. }
        ));
    }

    #[test]
    fn hints_for_another_tool_do_not_apply() {
        let call = r#"{"server":"srv","tool":"other"}"#;
        assert_eq!(
            mcp_ask_decision(USE.into(), call.into(), hints(true, false, true)),
            McpAskDecision::Prompt
        );
    }
}
