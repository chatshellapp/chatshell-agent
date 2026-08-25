//! Context compaction planning in a protocol-light form: threshold math,
//! the `findCutPoint` boundary walk, and conversation serialization with
//! the reference implementation's truncation budgets. Pure — no I/O, no
//! LLM calls;
//! the host decides methods (soft summary / snapcompact) and persists the
//! result.

use crate::context_pruning::estimate_tokens;

/// Newest tokens that must survive compaction verbatim (upstream
/// `keepRecentTokens` default 20,000).
pub const DEFAULT_KEEP_RECENT_TOKENS: usize = 20_000;

/// Reserve subtracted from the context window before the compaction
/// threshold (upstream `DEFAULT_RESERVE_TOKENS`).
pub const DEFAULT_RESERVE_TOKENS: usize = 16_384;

/// Fraction of the context window reserved when it exceeds the flat floor
/// (upstream: `max(window * 15%, reserve)`).
pub const RESERVE_WINDOW_FRACTION: f64 = 0.15;

/// Hard ceiling on a generated summary, in tokens (upstream `MAX_SUMMARY_TOKENS`).
pub const MAX_SUMMARY_TOKENS: usize = DEFAULT_RESERVE_TOKENS;

/// Fallback context window when the model's capability metadata is unknown.
pub const DEFAULT_CONTEXT_WINDOW: usize = 128_000;

// ---------------------------------------------------------------------------
// Threshold
// ---------------------------------------------------------------------------

/// Context-usage level at which compaction should trigger:
/// `context_window - max(15% of the window, 16,384)`.
pub fn resolve_threshold_tokens(context_window: usize) -> usize {
    let fractional = (context_window as f64 * RESERVE_WINDOW_FRACTION) as usize;
    context_window.saturating_sub(fractional.max(DEFAULT_RESERVE_TOKENS))
}

/// Whether `context_tokens` crosses the compaction threshold for the window.
pub fn should_compact(context_tokens: usize, context_window: usize) -> bool {
    context_tokens >= resolve_threshold_tokens(context_window)
}

// ---------------------------------------------------------------------------
// History model
// ---------------------------------------------------------------------------

/// Role of one chronological history item, reduced to what compaction needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryRole {
    User,
    Assistant,
    /// A tool invocation issued by an assistant (text = serialized args).
    ToolCall,
    /// A tool result following a [`HistoryRole::ToolCall`] (text = output).
    ToolResult,
}

/// One message-scale history item. Tool calls and results are separate items
/// so the cut-point walk can reason about turn structure.
#[derive(Debug, Clone)]
pub struct HistoryItem {
    pub id: String,
    pub role: HistoryRole,
    pub text: String,
    pub tool_name: Option<String>,
}

impl HistoryItem {
    /// Estimated tokens this item contributes to the context.
    pub fn tokens(&self) -> usize {
        estimate_tokens(&self.text)
    }
}

/// Sum of estimated tokens over the items.
pub fn total_tokens(items: &[HistoryItem]) -> usize {
    items.iter().map(HistoryItem::tokens).sum()
}

/// Where a compaction cuts: items `[0, keep_from)` are discarded (archived or
/// summarized), items `[keep_from, ..)` stay verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutPoint {
    pub keep_from: usize,
}

/// Port of upstream `findCutPoint`: walk newest -> oldest accumulating token
/// estimates until `keep_recent_tokens`, then cut at the nearest user-message
/// boundary at-or-before the crossing (older side), so the kept region always
/// covers at least `keep_recent_tokens`. Valid cuts are user starts only:
/// tool calls/results hang off assistant messages, so a user boundary can
/// never split an assistant(tool_calls) -> tool(result) chain.
///
/// Returns `None` when the whole history fits the keep window, when the only
/// user boundary would discard nothing, or when no user boundary exists.
pub fn find_cut_point(items: &[HistoryItem], keep_recent_tokens: usize) -> Option<CutPoint> {
    if items.is_empty() {
        return None;
    }
    let mut accumulated = 0usize;
    let mut crossing = None;
    for index in (0..items.len()).rev() {
        accumulated += items[index].tokens();
        if accumulated >= keep_recent_tokens {
            crossing = Some(index);
            break;
        }
    }
    let crossing = crossing?;
    let keep_from = (0..=crossing)
        .rev()
        .find(|&index| items[index].role == HistoryRole::User)?;
    if keep_from == 0 {
        return None;
    }
    Some(CutPoint { keep_from })
}

// ---------------------------------------------------------------------------
// Serialization (summary input / snapcompact archive source)
// ---------------------------------------------------------------------------

/// Character budgets applied while serializing discarded history, ported
/// from upstream `serializeConversation` defaults.
#[derive(Debug, Clone)]
pub struct SerializeBudgets {
    /// Per-tool-result character cap, split head+tail (errors and test
    /// failures usually land in the tail).
    pub tool_result_max_chars: usize,
    /// Per-argument-value cap inside one serialized tool call.
    pub tool_arg_max_chars: usize,
    /// Cap across one tool call's full serialized argument list.
    pub tool_call_max_chars: usize,
    /// Fraction of a truncation budget spent on the head; the rest keeps the
    /// tail.
    pub head_ratio: f64,
}

impl Default for SerializeBudgets {
    fn default() -> Self {
        Self {
            tool_result_max_chars: 2_000,
            tool_arg_max_chars: 500,
            tool_call_max_chars: 2_000,
            head_ratio: 0.6,
        }
    }
}

/// Keep the head and tail of `text`, eliding the middle beyond `max_chars`
/// (upstream `truncateForSummary`).
pub fn truncate_head_tail(text: &str, max_chars: usize, head_ratio: f64) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let head = (((max_chars as f64) * head_ratio.clamp(0.0, 1.0)) as usize).min(total);
    let tail = max_chars.saturating_sub(head).min(total - head);
    let elided = total - head - tail;
    let mut out: String = text.chars().take(head).collect();
    out.push_str(&format!("[... {elided} chars elided ...]"));
    out.extend(text.chars().skip(head + elided).take(tail));
    out
}

/// One serialized section of the discarded history.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptSection {
    User(String),
    Assistant(String),
    ToolCall { name: String, args: String },
    ToolResult { name: String, output: String },
}

/// Build the truncated transcript sections for the discarded prefix.
pub fn build_transcript(
    items: &[HistoryItem],
    budgets: &SerializeBudgets,
) -> Vec<TranscriptSection> {
    items
        .iter()
        .map(|item| match item.role {
            HistoryRole::User => TranscriptSection::User(item.text.trim().to_string()),
            HistoryRole::Assistant => TranscriptSection::Assistant(item.text.trim().to_string()),
            HistoryRole::ToolCall => TranscriptSection::ToolCall {
                name: item.tool_name.clone().unwrap_or_else(|| "tool".into()),
                args: truncate_head_tail(
                    item.text.trim(),
                    budgets.tool_call_max_chars,
                    budgets.head_ratio,
                ),
            },
            HistoryRole::ToolResult => TranscriptSection::ToolResult {
                name: item.tool_name.clone().unwrap_or_else(|| "tool".into()),
                output: truncate_head_tail(
                    item.text.trim(),
                    budgets.tool_result_max_chars,
                    budgets.head_ratio,
                ),
            },
        })
        .collect()
}

/// Render transcript sections as the plain text fed to a summarizer (or to
/// the snapcompact normalizer). With `dim_tool_noise`, tool call/result
/// sections are wrapped in the renderer's zero-width dim-ink toggles so the
/// bitmap archive prints conversation text louder than tool noise.
pub fn render_transcript(sections: &[TranscriptSection], dim_tool_noise: bool) -> String {
    let mut out = String::new();
    for section in sections {
        match section {
            TranscriptSection::User(text) => {
                out.push_str("User: ");
                out.push_str(text);
            }
            TranscriptSection::Assistant(text) => {
                out.push_str("Assistant: ");
                out.push_str(text);
            }
            TranscriptSection::ToolCall { name, args } => {
                if dim_tool_noise {
                    out.push('\u{000e}');
                }
                out.push_str(&format!("Tool call ({name}): {args}"));
                if dim_tool_noise {
                    out.push('\u{000f}');
                }
            }
            TranscriptSection::ToolResult { name, output } => {
                if dim_tool_noise {
                    out.push('\u{000e}');
                }
                out.push_str(&format!("Tool result ({name}): {output}"));
                if dim_tool_noise {
                    out.push('\u{000f}');
                }
            }
        }
        out.push_str("\n\n");
    }
    out
}

// ---------------------------------------------------------------------------
// Soft-summary prompts (ported from the reference implementation's compaction prompts)
// ---------------------------------------------------------------------------

/// System prompt for the summarization one-shot.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a summarization engine. You read a conversation transcript \
and output exactly one structured handoff summary for another LLM to resume the task. \
You never output anything besides the structured summary.";

/// First-pass instruction (port of upstream `compaction-summary.md`).
pub const SUMMARY_INSTRUCTION: &str = "Summarize the conversation above into a structured handoff summary for another LLM to resume the task.\n\
\n\
IMPORTANT: If the conversation ends with an unanswered question or a request awaiting user response, you MUST preserve that exact question/request.\n\
\n\
Use this format (sections can be omitted if not applicable):\n\
\n\
## Goal\n\
[User goals; list multiple if session covers different tasks.]\n\
\n\
## Constraints & Preferences\n\
- [Constraints or requirements mentioned]\n\
\n\
## Progress\n\
\n\
### Done\n\
- [x] [Completed tasks/changes]\n\
\n\
### In Progress\n\
- [ ] [Current work]\n\
\n\
### Blocked\n\
- [Issues preventing progress]\n\
\n\
## Key Decisions\n\
- **[Decision]**: [Brief rationale]\n\
\n\
## Next Steps\n\
1. [Ordered list of next actions]\n\
\n\
## Critical Context\n\
- [Important data, pending questions, references]\n\
\n\
## Additional Notes\n\
[Anything else important not covered above]\n\
\n\
Output only the structured summary. Keep sections concise. Preserve exact file paths, function names, \
error messages, and relevant tool outputs or command results.";

/// Iterative instruction when a previous summary exists (port of upstream
/// `compaction-update-summary.md`).
pub const UPDATE_SUMMARY_INSTRUCTION: &str = "Update the existing handoff summary in <previous-summary> from the new messages above, for another LLM to resume.\n\
\n\
MUST:\n\
- preserve all previous-summary information; add new progress, decisions, context.\n\
- Progress: move completed \"In Progress\" items to \"Done\".\n\
- update \"Next Steps\" for completed work.\n\
- preserve exact file paths, function names, error messages.\n\
- MAY remove irrelevant content.\n\
- If the new messages end with an unanswered user question/request: add it to Critical Context; replace any previous pending question if answered.\n\
- output only the structured summary; keep sections concise.\n\
- include mentioned repository state changes if any.";

/// Build the user message for the summarization one-shot: optional iterative
/// `<previous-summary>`, then the wrapped conversation and the instruction.
pub fn build_summary_user_prompt(
    conversation_text: &str,
    previous_summary: Option<&str>,
) -> String {
    let mut out = String::new();
    if let Some(previous) = previous_summary {
        out.push_str("<previous-summary>\n");
        out.push_str(previous.trim());
        out.push_str("\n</previous-summary>\n\n");
    }
    out.push_str("<conversation>\n");
    out.push_str(conversation_text.trim());
    out.push_str("\n</conversation>\n\n");
    out.push_str(if previous_summary.is_some() {
        UPDATE_SUMMARY_INSTRUCTION
    } else {
        SUMMARY_INSTRUCTION
    });
    out
}

/// Clamp a generated summary to [`MAX_SUMMARY_TOKENS`] (the LLM call itself
/// cannot carry a max_tokens parameter through every provider path, so the
/// clamp is applied on the produced text).
pub fn clamp_summary(summary: &str) -> String {
    let max_chars = MAX_SUMMARY_TOKENS * 4;
    if summary.chars().count() <= max_chars {
        return summary.to_string();
    }
    let mut out: String = summary.chars().take(max_chars).collect();
    out.push_str("\n[summary clamped]");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, role: HistoryRole, text: &str) -> HistoryItem {
        HistoryItem {
            id: id.to_string(),
            role,
            text: text.to_string(),
            tool_name: Some("grep".into()),
        }
    }

    fn filler(tokens: usize) -> String {
        "a".repeat(tokens * 4)
    }

    #[test]
    fn threshold_takes_the_larger_reserve() {
        // 128k window: 15% = 19,200 > 16,384 floor.
        assert_eq!(resolve_threshold_tokens(128_000), 128_000 - 19_200);
        // Small window: fraction below the floor, flat 16,384 wins.
        assert_eq!(resolve_threshold_tokens(64_000), 64_000 - 16_384);
        // Tiny window: never negative.
        assert_eq!(resolve_threshold_tokens(8_000), 0);
    }

    #[test]
    fn should_compact_boundary() {
        let window = 128_000;
        let threshold = resolve_threshold_tokens(window);
        assert!(!should_compact(threshold - 1, window));
        assert!(should_compact(threshold, window));
    }

    #[test]
    fn cut_point_keeps_recent_window_at_user_boundary() {
        let items = vec![
            item("m0", HistoryRole::User, &filler(12_000)),
            item("m1", HistoryRole::Assistant, &filler(2_000)),
            item("m2", HistoryRole::User, &filler(12_000)),
            item("m3", HistoryRole::Assistant, &filler(2_000)),
            item("m4", HistoryRole::User, &filler(12_000)),
        ];
        let cut = find_cut_point(&items, 20_000).expect("cut exists");
        // Walking back: m4 (12k) + m3 (2k) + m2 (12k) crosses 20k at m2;
        // m2 itself is a user boundary -> keep [m2..].
        assert_eq!(cut.keep_from, 2);
        assert_eq!(items[cut.keep_from].role, HistoryRole::User);
        let kept_tokens: usize = items[cut.keep_from..].iter().map(HistoryItem::tokens).sum();
        assert!(kept_tokens >= 20_000, "kept only {kept_tokens}");
    }

    #[test]
    fn cut_point_none_when_history_fits() {
        let items = vec![
            item("m0", HistoryRole::User, "hi"),
            item("m1", HistoryRole::Assistant, "hello"),
        ];
        assert!(find_cut_point(&items, 20_000).is_none());
    }

    #[test]
    fn cut_point_none_when_only_first_user_exists() {
        // One giant turn: the only user boundary is index 0 -> nothing to
        // discard.
        let items = vec![
            item("m0", HistoryRole::User, &filler(30_000)),
            item("t0", HistoryRole::ToolResult, &filler(30_000)),
            item("m1", HistoryRole::Assistant, "done"),
        ];
        assert!(find_cut_point(&items, 20_000).is_none());
    }

    #[test]
    fn cut_point_never_splits_tool_chain() {
        let items = vec![
            item("u0", HistoryRole::User, &filler(1_000)),
            item("a0", HistoryRole::Assistant, &filler(1_000)),
            item("c0", HistoryRole::ToolCall, "{}"),
            item("r0", HistoryRole::ToolResult, &filler(10_000)),
            item("u1", HistoryRole::User, &filler(1_000)),
            item("a1", HistoryRole::Assistant, &filler(1_000)),
        ];
        // A window that would cross inside the tool result extends the cut
        // back to the turn's user start (index 0) -> nothing to discard.
        assert!(find_cut_point(&items, 8_000).is_none());
        // A window that fits between turns cuts exactly at u1, discarding
        // the complete chain [u0, a0, c0, r0).
        let cut = find_cut_point(&items, 2_000).expect("cut exists");
        assert_eq!(cut.keep_from, 4);
    }

    #[test]
    fn truncation_keeps_head_and_tail() {
        let text = "H".repeat(1_200) + &"T".repeat(800);
        let out = truncate_head_tail(&text, 1_000, 0.6);
        assert!(out.starts_with("HHHH"), "{out:.40}");
        assert!(out.ends_with("TTTT"), "{out:.40}");
        assert!(out.contains("chars elided"), "{out:.60}");
        // Short text untouched.
        assert_eq!(truncate_head_tail("short", 100, 0.6), "short");
    }

    #[test]
    fn transcript_budgets_and_render() {
        let items = vec![
            item("m0", HistoryRole::User, "please fix the bug"),
            item("t0", HistoryRole::ToolCall, &"x".repeat(5_000)),
            item(
                "t1",
                HistoryRole::ToolResult,
                &format!("{}tail", "r".repeat(9_000)),
            ),
            item("m1", HistoryRole::Assistant, "fixed"),
        ];
        let sections = build_transcript(&items, &SerializeBudgets::default());
        let plain = render_transcript(&sections, false);
        assert!(plain.contains("User: please fix the bug"));
        assert!(plain.contains("Tool call (grep):"));
        assert!(plain.contains("Tool result (grep):"));
        assert!(plain.contains("tail"));
        assert!(!plain.contains('\u{000e}'));

        let dimmed = render_transcript(&sections, true);
        assert!(dimmed.contains('\u{000e}'));
        assert!(dimmed.contains('\u{000f}'));
        // Dim toggles hug tool sections only.
        assert!(dimmed.matches('\u{000e}').count() >= 2);

        // Budget enforcement: the tool call args shrink to <= ~2000 chars.
        let call = sections.iter().find_map(|s| match s {
            TranscriptSection::ToolCall { args, .. } => Some(args.chars().count()),
            _ => None,
        });
        assert!(call.unwrap() <= 2_100);
    }

    #[test]
    fn summary_prompt_wraps_conversation() {
        let first = build_summary_user_prompt("hello world", None);
        assert!(first.contains("<conversation>\nhello world\n</conversation>"));
        assert!(first.contains(SUMMARY_INSTRUCTION));

        let update = build_summary_user_prompt("more", Some("old summary"));
        assert!(update.contains("<previous-summary>\nold summary\n</previous-summary>"));
        assert!(update.contains(UPDATE_SUMMARY_INSTRUCTION));
    }

    #[test]
    fn summary_clamp() {
        let short = "fine".to_string();
        assert_eq!(clamp_summary(&short), short);
        let long = "a".repeat(MAX_SUMMARY_TOKENS * 4 + 100);
        assert!(clamp_summary(&long).contains("[summary clamped]"));
    }
}
