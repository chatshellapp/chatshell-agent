//! Attachment ingestion and context compaction for hosts without a Rust
//! backend of their own (iOS): PDF text and page images, Office documents,
//! provider image fitting, and the compaction / snapcompact planning the
//! desktop runner uses. Thin wrappers — the logic lives in the core.

use chatshell_agent_core as core;
use core::compaction::{self, HistoryItem, HistoryRole, SerializeBudgets};
use core::snapcompact;

#[derive(Debug, Clone, thiserror::Error, uniffi::Error)]
pub enum DocumentError {
    #[error("{message}")]
    Failed { message: String },
}

fn failed(message: impl ToString) -> DocumentError {
    DocumentError::Failed {
        message: message.to_string(),
    }
}

// ---------------------------------------------------------------------------
// PDF
// ---------------------------------------------------------------------------

/// A PDF's text layer and page count.
#[derive(Debug, Clone, uniffi::Record)]
pub struct PdfText {
    pub text: String,
    pub page_count: u32,
}

/// Text of one 1-based page.
#[derive(Debug, Clone, uniffi::Record)]
pub struct PdfPage {
    pub page: u32,
    pub text: String,
}

/// One scan-sized image of a page.
#[derive(Debug, Clone, uniffi::Record)]
pub struct PdfPageImage {
    pub bytes: Vec<u8>,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
}

/// Extract the whole text layer (empty for scanned PDFs).
#[uniffi::export]
pub fn pdf_text(bytes: Vec<u8>) -> Result<PdfText, DocumentError> {
    let document = core::pdf_text::extract_pdf_text_document(&bytes).map_err(failed)?;
    Ok(PdfText {
        text: document.text,
        page_count: document.page_count as u32,
    })
}

/// Extract the text of specific 1-based pages.
#[uniffi::export]
pub fn pdf_pages(bytes: Vec<u8>, pages: Vec<u32>) -> Result<Vec<PdfPage>, DocumentError> {
    let requested: Vec<usize> = pages.into_iter().map(|page| page as usize).collect();
    let result = core::pdf_text::extract_pdf_pages(&bytes, &requested).map_err(failed)?;
    Ok(result
        .pages
        .into_iter()
        .map(|page| PdfPage {
            page: page.page as u32,
            text: page.text,
        })
        .collect())
}

/// The scan-sized images of one 1-based page, largest first.
#[uniffi::export]
pub fn pdf_page_images(bytes: Vec<u8>, page: u32) -> Result<Vec<PdfPageImage>, DocumentError> {
    let result = core::pdf_images::extract_page_images(&bytes, page as usize).map_err(failed)?;
    Ok(result
        .images
        .into_iter()
        .map(|image| PdfPageImage {
            bytes: image.bytes,
            mime_type: image.mime_type.to_string(),
            width: image.width,
            height: image.height,
        })
        .collect())
}

/// Whether extracted text counts as a text layer (desktop's
/// `has_text_layer`: at least 50 non-whitespace characters).
#[uniffi::export]
pub fn has_text_layer(text: String) -> bool {
    text.chars().filter(|c| !c.is_whitespace()).take(50).count() >= 50
}

// ---------------------------------------------------------------------------
// Office
// ---------------------------------------------------------------------------

/// Read a docx / pptx / xlsx (and macro variants) as text: Markdown for
/// documents and slides, a bounded TSV view for workbooks. `sheet` and
/// `range` narrow a workbook read. Legacy binary formats get the same
/// actionable error the read tool gives.
#[uniffi::export]
pub fn office_text(
    bytes: Vec<u8>,
    extension: String,
    sheet: Option<String>,
    range: Option<String>,
) -> Result<String, DocumentError> {
    let ext = extension.trim_start_matches('.').to_ascii_lowercase();
    if core::office::is_legacy_office_extension(&ext) {
        return Err(failed(core::office::legacy_message(&ext)));
    }
    match ext.as_str() {
        "docx" | "docm" => core::office::read_docx(&bytes).map_err(failed),
        "pptx" | "pptm" => core::office::read_pptx(&bytes).map_err(failed),
        "xlsx" | "xlsm" => {
            core::office::read_xlsx(&bytes, sheet.as_deref(), range.as_deref(), false)
                .map_err(failed)
        }
        other => Err(failed(format!("Not an Office document: .{other}"))),
    }
}

/// Whether the extension is an Office format the core can read.
#[uniffi::export]
pub fn is_office_extension(extension: String) -> bool {
    core::office::is_office_extension(&extension.trim_start_matches('.').to_ascii_lowercase())
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

/// An image brought within provider limits.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FittedImage {
    pub bytes: Vec<u8>,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
    pub changed: bool,
}

/// Desktop's attach-time image fit: longest side <= 2000 px and encoded
/// size under the provider budget, re-encoding as JPEG when needed.
#[uniffi::export]
pub fn fit_image(bytes: Vec<u8>) -> Result<FittedImage, DocumentError> {
    let fitted = core::image_fit::fit_image(&bytes).map_err(failed)?;
    Ok(FittedImage {
        bytes: fitted.bytes,
        mime_type: fitted.mime_type,
        width: fitted.width,
        height: fitted.height,
        changed: fitted.changed,
    })
}

/// The core's token estimate (CJK-aware), shared by budgets on all hosts.
#[uniffi::export]
pub fn estimate_tokens(text: String) -> u64 {
    core::context_pruning::estimate_tokens(&text) as u64
}

// ---------------------------------------------------------------------------
// Compaction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum CompactionRole {
    User,
    Assistant,
    ToolCall,
    ToolResult,
}

/// One chronological history item (desktop `load_history_items`).
#[derive(Debug, Clone, uniffi::Record)]
pub struct CompactionItem {
    pub id: String,
    pub role: CompactionRole,
    pub text: String,
    pub tool_name: Option<String>,
}

fn to_core(items: &[CompactionItem]) -> Vec<HistoryItem> {
    items
        .iter()
        .map(|item| HistoryItem {
            id: item.id.clone(),
            role: match item.role {
                CompactionRole::User => HistoryRole::User,
                CompactionRole::Assistant => HistoryRole::Assistant,
                CompactionRole::ToolCall => HistoryRole::ToolCall,
                CompactionRole::ToolResult => HistoryRole::ToolResult,
            },
            text: item.text.clone(),
            tool_name: item.tool_name.clone(),
        })
        .collect()
}

/// Where to cut, when the context crosses the threshold: items before
/// `keep_from` are compacted.
#[derive(Debug, Clone, uniffi::Record)]
pub struct CompactionPlan {
    pub keep_from: u32,
    pub context_tokens: u64,
}

/// Desktop `run_post_turn_compaction`'s decision: `prior_tokens` is the
/// prior compaction's projection cost, `items` the uncompacted suffix.
/// None when under the threshold or when no user boundary can be cut.
#[uniffi::export]
pub fn plan_compaction(
    items: Vec<CompactionItem>,
    context_window: u64,
    prior_tokens: u64,
) -> Option<CompactionPlan> {
    let core_items = to_core(&items);
    let context_tokens = prior_tokens as usize + compaction::total_tokens(&core_items);
    let window = if context_window == 0 {
        compaction::DEFAULT_CONTEXT_WINDOW
    } else {
        context_window as usize
    };
    if !compaction::should_compact(context_tokens, window) {
        return None;
    }
    let cut = compaction::find_cut_point(&core_items, compaction::DEFAULT_KEEP_RECENT_TOKENS)?;
    Some(CompactionPlan {
        keep_from: cut.keep_from as u32,
        context_tokens: context_tokens as u64,
    })
}

/// The serialized transcript of discarded items (summary input, or the
/// snapcompact source with `dim_tool_noise`).
#[uniffi::export]
pub fn compaction_transcript(items: Vec<CompactionItem>, dim_tool_noise: bool) -> String {
    compaction::render_transcript(
        &compaction::build_transcript(&to_core(&items), &SerializeBudgets::default()),
        dim_tool_noise,
    )
}

/// System and user prompts for the soft-summary one-shot.
#[derive(Debug, Clone, uniffi::Record)]
pub struct SummaryPrompts {
    pub system: String,
    pub user: String,
}

#[uniffi::export]
pub fn compaction_summary_prompts(
    transcript: String,
    previous_summary: Option<String>,
) -> SummaryPrompts {
    SummaryPrompts {
        system: compaction::SUMMARIZATION_SYSTEM_PROMPT.to_string(),
        user: compaction::build_summary_user_prompt(&transcript, previous_summary.as_deref()),
    }
}

#[uniffi::export]
pub fn clamp_compaction_summary(summary: String) -> String {
    compaction::clamp_summary(&summary)
}

/// One rendered snapcompact frame.
#[derive(Debug, Clone, uniffi::Record)]
pub struct SnapcompactFrame {
    pub width: u32,
    pub height: u32,
    pub png_base64: String,
}

/// A snapcompact archive: `shell_json` is what the compaction row stores
/// (frames stripped), `frames` the rendered images.
#[derive(Debug, Clone, uniffi::Record)]
pub struct SnapcompactResult {
    pub shell_json: String,
    pub head_text: String,
    pub tail_text: String,
    pub frames: Vec<SnapcompactFrame>,
    pub chars_dropped: u64,
    /// Billed tokens of the frames plus the text edges.
    pub projection_tokens: u64,
}

fn snapcompact_result(
    archive: snapcompact::SnapcompactArchive,
) -> Result<SnapcompactResult, DocumentError> {
    let family = snapcompact::billing_family(&archive.provider);
    let projection_tokens = snapcompact::archive_token_estimate(&archive, family)
        + core::context_pruning::estimate_tokens(&archive.head_text)
        + core::context_pruning::estimate_tokens(&archive.tail_text);
    let shell_json = serde_json::to_string(&archive.persisted_shell()).map_err(failed)?;
    Ok(SnapcompactResult {
        shell_json,
        head_text: archive.head_text,
        tail_text: archive.tail_text,
        frames: archive
            .frames
            .into_iter()
            .map(|frame| SnapcompactFrame {
                width: frame.width as u32,
                height: frame.height as u32,
                png_base64: frame.png_base64,
            })
            .collect(),
        chars_dropped: archive.chars_dropped as u64,
        projection_tokens: projection_tokens as u64,
    })
}

/// Build a snapcompact archive from a transcript for `provider`. Errors
/// when the transcript is too short to need frames.
#[uniffi::export]
pub fn snapcompact_build(
    transcript: String,
    provider: String,
) -> Result<SnapcompactResult, DocumentError> {
    snapcompact::build_archive(&transcript, &provider)
        .map_err(failed)
        .and_then(snapcompact_result)
}

/// Resolve a stored archive for projection (desktop
/// `archive_for_projection`): legacy rows carry frames inline; shell rows
/// re-render from `source_text` + `provider`.
#[uniffi::export]
pub fn snapcompact_resolve(archive_json: String) -> Result<SnapcompactResult, DocumentError> {
    let parsed: snapcompact::SnapcompactArchive =
        serde_json::from_str(&archive_json).map_err(failed)?;
    if !parsed.frames.is_empty() || parsed.source_text.is_empty() || parsed.provider.is_empty() {
        return snapcompact_result(parsed);
    }
    snapcompact::build_archive(&parsed.source_text, &parsed.provider)
        .map_err(failed)
        .and_then(snapcompact_result)
}

/// The carried source text of a stored archive (re-compaction ages it in
/// ahead of the newly discarded region).
#[uniffi::export]
pub fn snapcompact_source_text(archive_json: String) -> Option<String> {
    let parsed: snapcompact::SnapcompactArchive = serde_json::from_str(&archive_json).ok()?;
    (!parsed.source_text.is_empty()).then_some(parsed.source_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, role: CompactionRole, tokens: usize) -> CompactionItem {
        CompactionItem {
            id: id.into(),
            role,
            text: "a".repeat(tokens * 4),
            tool_name: None,
        }
    }

    #[test]
    fn plan_only_past_the_threshold() {
        let small = vec![item("u1", CompactionRole::User, 10)];
        assert!(plan_compaction(small, 128_000, 0).is_none());

        let mut big = Vec::new();
        for turn in 0..12 {
            big.push(item(&format!("u{turn}"), CompactionRole::User, 5_000));
            big.push(item(&format!("a{turn}"), CompactionRole::Assistant, 5_000));
        }
        let plan = plan_compaction(big.clone(), 100_000, 0).expect("over threshold");
        assert!(plan.keep_from > 0 && (plan.keep_from as usize) < big.len());
        assert!(matches!(
            big[plan.keep_from as usize].role,
            CompactionRole::User
        ));
    }

    #[test]
    fn office_rejects_legacy_and_unknown() {
        assert!(office_text(vec![], "doc".into(), None, None).is_err());
        assert!(office_text(vec![], "txt".into(), None, None).is_err());
        assert!(is_office_extension(".DOCX".into()));
    }

    #[test]
    fn snapcompact_shell_round_trips() {
        let transcript: String = (0..2_000)
            .map(|line| format!("line {line} of an archived transcript with enough text"))
            .collect::<Vec<_>>()
            .join("\n");
        let built = snapcompact_build(transcript, "anthropic".into()).expect("builds");
        assert!(!built.frames.is_empty());
        let shell: serde_json::Value = serde_json::from_str(&built.shell_json).expect("json");
        assert_eq!(
            shell["frames"].as_array().map(Vec::len),
            Some(0),
            "shell has no frames"
        );
        let resolved = snapcompact_resolve(built.shell_json.clone()).expect("re-renders");
        assert_eq!(resolved.frames.len(), built.frames.len());
        assert_eq!(resolved.head_text, built.head_text);
        assert!(snapcompact_source_text(built.shell_json).is_some());
    }
}
