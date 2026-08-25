//! Tool-output pruning policy for context management, ported from the
//! reference implementation's pruning semantics: historical tool results inside
//! the protected recent window stay verbatim; older, individually-large
//! results collapse into a short truncation notice — but only when the total
//! saving clears a threshold, so pruning never churns the prompt cache for a
//! negligible gain.

/// Newest tool-output tokens that must never be pruned.
pub const DEFAULT_PROTECT_RECENT_TOKENS: usize = 40_000;
/// Pruning must save at least this many tokens to be worth the cache churn.
pub const DEFAULT_MIN_TOTAL_SAVINGS: usize = 20_000;
/// Never prune a result below this size: the placeholder itself costs
/// [`PLACEHOLDER_TOKENS`] tokens, so pruning smaller results would grow the
/// context.
pub const MIN_PRUNE_TOKENS: usize = 50;
/// Approximate token cost of the truncation placeholder.
pub const PLACEHOLDER_TOKENS: usize = 8;

/// Placeholder shown to the LLM for elided, contextually useless results
/// (zero-match searches, empty fetches). Distinct from truncation: nothing is
/// recoverable because there was nothing there.
pub const UNEVENTFUL_NOTICE: &str = "[Uneventful result elided]";

/// Cheap local token estimate (~4 chars/token). Deliberately not a real
/// tokenizer: pruning decisions only need consistent relative magnitudes.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.chars().count().div_ceil(4)
    }
}

/// The placeholder replacing a pruned tool output.
pub fn truncation_notice(original_tokens: usize) -> String {
    format!("[Output truncated - ~{original_tokens} tokens]")
}

/// Single-line ellipsize for catalog-style text (tool descriptions, skill
/// entries): collapse internal whitespace and cap at `max_chars` including
/// the trailing ellipsis character.
pub fn ellipsize(text: &str, max_chars: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let char_count = collapsed.chars().count();
    if char_count <= max_chars {
        collapsed
    } else if max_chars <= 1 {
        "\u{2026}".to_string()
    } else {
        let kept: String = collapsed.chars().take(max_chars - 1).collect();
        format!("{kept}\u{2026}")
    }
}

/// Whether a tool output is contextually useless - it happened, but it
/// carries nothing the model needs later. Zero-hit searches are the classic
/// case: the fact "we searched and found nothing" is worth one line at most,
/// and only near the turn it happened.
///
/// Errors are NOT uneventful (they surface through a different channel and
/// the model may need to react to them).
pub fn is_uneventful(tool_name: &str, output: &str) -> bool {
    match tool_name {
        // Deterministic zero-hit marker emitted by the desktop web_search tool.
        "web_search" => output.contains("**Results Found:** 0"),
        _ => false,
    }
}

/// How many newest tool outputs keep their uneventful text verbatim. Fresh
/// no-hit results are cheap and give the current turn context; stale ones are
/// noise.
pub const DEFAULT_KEEP_RECENT_UNEVENTFUL: usize = 8;

/// Decide which uneventful entries elide to [`UNEVENTFUL_NOTICE`]. Only the
/// `keep_recent` newest entries survive verbatim; older uneventful entries
/// elide regardless of size (the notice is smaller than any real output).
pub fn plan_elide_uneventful(entries: &[ToolOutputRef], keep_recent: usize) -> Vec<bool> {
    let n = entries.len();
    let cutoff = n.saturating_sub(keep_recent);
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| e.uneventful && i < cutoff)
        .collect()
}

/// Tunable pruning policy.
#[derive(Debug, Clone)]
pub struct PrunePolicy {
    pub protect_recent_tokens: usize,
    pub min_total_savings: usize,
    pub min_result_tokens: usize,
    /// Tool names whose outputs are never pruned (e.g. loaded skill
    /// instructions — the model cannot re-read them from history).
    pub protected_tools: Vec<String>,
}

impl Default for PrunePolicy {
    fn default() -> Self {
        Self {
            protect_recent_tokens: DEFAULT_PROTECT_RECENT_TOKENS,
            min_total_savings: DEFAULT_MIN_TOTAL_SAVINGS,
            min_result_tokens: MIN_PRUNE_TOKENS,
            protected_tools: vec!["skill".to_string()],
        }
    }
}

/// One tool output under consideration, in chronological order.
#[derive(Debug, Clone)]
pub struct ToolOutputRef {
    pub tool_name: String,
    pub tokens: usize,
    /// Contextually useless (zero-hit search, empty result): eligible for
    /// elision via [`plan_elide_uneventful`].
    pub uneventful: bool,
}

/// Decide which entries to truncate. Returns a parallel `Vec<bool>` (`true` =
/// replace with [`truncation_notice`]).
///
/// Algorithm: walk newest → oldest accumulating token estimates until the
/// protected window is filled; everything older is a *candidate* if it is at
/// least `min_result_tokens` and its tool is not protected. Truncation fires
/// only when the candidates' combined savings reach `min_total_savings` —
/// otherwise the plan is all-keep (a partial prune would invalidate the
/// provider prompt cache for almost no gain).
pub fn plan_prune(entries: &[ToolOutputRef], policy: &PrunePolicy) -> Vec<bool> {
    let n = entries.len();
    let mut plan = vec![false; n];
    if n == 0 {
        return plan;
    }

    // Find the protected boundary: the oldest index still inside the window.
    let mut accumulated = 0usize;
    let mut boundary = 0; // entries[..boundary] are candidates (older side)
    for i in (0..n).rev() {
        accumulated += entries[i].tokens;
        if accumulated >= policy.protect_recent_tokens {
            boundary = i;
            break;
        }
        boundary = i;
    }
    if boundary == 0 {
        // Everything fits in the protected window.
        return plan;
    }

    let is_candidate = |entry: &ToolOutputRef| {
        entry.tokens >= policy.min_result_tokens
            && !policy.protected_tools.contains(&entry.tool_name)
    };

    let mut savings = 0usize;
    for entry in entries.iter().take(boundary) {
        if is_candidate(entry) {
            savings += entry.tokens.saturating_sub(PLACEHOLDER_TOKENS);
        }
    }
    if savings < policy.min_total_savings {
        return plan;
    }

    for (i, entry) in entries.iter().enumerate().take(boundary) {
        if is_candidate(entry) {
            plan[i] = true;
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(tool: &str, tokens: usize) -> ToolOutputRef {
        ToolOutputRef {
            tool_name: tool.to_string(),
            tokens,
            uneventful: false,
        }
    }

    fn policy(protect: usize, min_savings: usize) -> PrunePolicy {
        PrunePolicy {
            protect_recent_tokens: protect,
            min_total_savings: min_savings,
            min_result_tokens: MIN_PRUNE_TOKENS,
            protected_tools: vec!["skill".into()],
        }
    }

    #[test]
    fn everything_inside_window_is_kept() {
        let entries = vec![entry("bash", 100), entry("bash", 100)];
        let plan = plan_prune(&entries, &policy(500, 10));
        assert!(plan.iter().all(|p| !p), "all kept: {plan:?}");
    }

    #[test]
    fn old_large_results_truncate_when_savings_clear_threshold() {
        // 10 old results of 1k tokens = ~10k savings; threshold 5k -> prune.
        let mut entries: Vec<_> = (0..10).map(|_| entry("bash", 1_000)).collect();
        // One protected-window result at the end.
        entries.push(entry("bash", 50_000));
        let plan = plan_prune(&entries, &policy(40_000, 5_000));
        assert_eq!(plan.len(), 11);
        assert!(plan[..10].iter().all(|p| *p), "old side pruned");
        assert!(!plan[10], "newest protected");
    }

    #[test]
    fn below_savings_threshold_is_noop() {
        let mut entries: Vec<_> = (0..3).map(|_| entry("bash", 1_000)).collect();
        entries.push(entry("bash", 50_000));
        // ~3k savings < 20k threshold -> keep everything.
        let plan = plan_prune(&entries, &policy(40_000, 20_000));
        assert!(plan.iter().all(|p| !p));
    }

    #[test]
    fn small_results_and_protected_tools_survive() {
        let entries = vec![
            entry("bash", 1_000),  // candidate -> pruned
            entry("bash", 10),     // below MIN_PRUNE_TOKENS -> kept
            entry("skill", 5_000), // protected tool -> kept
            entry("bash", 60_000), // protected window
        ];
        let plan = plan_prune(&entries, &policy(40_000, 500));
        assert!(plan[0]);
        assert!(!plan[1]);
        assert!(!plan[2]);
        assert!(!plan[3]);
    }

    #[test]
    fn ellipsize_single_line_and_bound() {
        assert_eq!(ellipsize("short", 10), "short");
        assert_eq!(ellipsize("  a   b  c  ", 10), "a b c");
        let long = "word ".repeat(50);
        let cut = ellipsize(&long, 20);
        assert_eq!(cut.chars().count(), 20);
        assert!(cut.ends_with('\u{2026}'));
        assert!(!cut.contains('\n'));
    }

    #[test]
    fn uneventful_detection() {
        let zero_hit = "## Search Results for: \"q\"\n\n**Provider:** DuckDuckGo\n**Results Found:** 0\n\nNo results found for this query.\n";
        assert!(is_uneventful("web_search", zero_hit));
        assert!(!is_uneventful(
            "web_search",
            &zero_hit.replace("**Results Found:** 0", "**Results Found:** 3")
        ));
        assert!(!is_uneventful("bash", zero_hit));
    }

    #[test]
    fn elide_plan_keeps_recent_and_eventful() {
        let mut entries: Vec<ToolOutputRef> = (0..12)
            .map(|_| entry("web_search", 20))
            .map(|mut e| {
                e.uneventful = true;
                e
            })
            .collect();
        entries[5].uneventful = false; // an old real result among the noise
        let plan = plan_elide_uneventful(&entries, DEFAULT_KEEP_RECENT_UNEVENTFUL);
        // 12 entries, keep 8 newest -> cutoff 4; old uneventful elide.
        assert!(plan[..4].iter().all(|p| *p));
        assert!(
            !plan[4..].iter().any(|p| *p),
            "recent window stays verbatim"
        );
        assert!(!plan[5]);
    }

    #[test]
    fn estimate_and_notice() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(truncation_notice(1234), "[Output truncated - ~1234 tokens]");
        assert_eq!(UNEVENTFUL_NOTICE, "[Uneventful result elided]");
    }
}
