//! PDF text extraction shared by the `read` tool and host attachment
//! ingestion.

use unicode_normalization::UnicodeNormalization;

/// Extract the text layer from PDF bytes.
///
/// Callers that present the result to a model should handle the empty-text
/// case themselves: scanned/image-only PDFs legitimately produce no text.
///
/// Some generators write a broken `startxref`/trailer or an incremental-update
/// chain whose `/Prev` points at a trailer instead of an xref table. That makes
/// `pdf-extract`'s pinned lopdf reject the file ("Invalid file trailer" /
/// "Invalid cross-reference table"). Newer lopdf recovers by scanning indirect
/// objects, so a reserialized copy is retried once before giving up. The
/// reserialized copy must drop `/Prev`/`XRefStm` (their offsets describe the
/// original layout) and the recovered xref root.
pub fn extract_pdf_text(bytes: &[u8]) -> anyhow::Result<String> {
    let text = match pdf_extract::extract_text_from_mem(bytes) {
        Ok(text) => text,
        Err(primary) => {
            let mut document = lopdf::Document::load_mem(bytes)
                .map_err(|repair| anyhow::anyhow!("{primary}; repair failed: {repair}"))?;
            document.trailer.remove(b"Prev");
            document.trailer.remove(b"XRefStm");
            document.xref_start = 0;
            let mut repaired = Vec::new();
            document
                .save_to(&mut repaired)
                .map_err(|error| anyhow::anyhow!("{primary}; repair save failed: {error}"))?;
            pdf_extract::extract_text_from_mem(&repaired).map_err(|error| {
                anyhow::anyhow!("{primary}; repaired extraction failed: {error}")
            })?
        }
    };
    // PDF text layers commonly use CJK compatibility ideographs; fold them to
    // the unified code points models are trained on.
    Ok(text.nfkc().collect())
}
