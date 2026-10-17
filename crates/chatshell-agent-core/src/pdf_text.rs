//! PDF text extraction shared by the `read` tool and host attachment
//! ingestion.

use pdf_extract::{Document, PlainTextOutput, output_doc, output_doc_page};
use unicode_normalization::UnicodeNormalization;

/// Extracted text plus the number of pages in the document.
#[derive(Debug, Clone)]
pub struct PdfTextDocument {
    pub text: String,
    pub page_count: usize,
}

/// Text extracted from one 1-based physical page.
#[derive(Debug, Clone)]
pub struct PdfPageText {
    pub page: usize,
    pub text: String,
}

/// A page selection plus the total page count of the source document.
#[derive(Debug, Clone)]
pub struct PdfPages {
    pub page_count: usize,
    pub pages: Vec<PdfPageText>,
}

/// Extract the text layer from PDF bytes.
///
/// Callers that present the result to a model should handle the empty-text
/// case themselves: scanned/image-only PDFs legitimately produce no text.
pub fn extract_pdf_text(bytes: &[u8]) -> anyhow::Result<String> {
    Ok(extract_pdf_text_document(bytes)?.text)
}

/// Extract the text layer and page count from PDF bytes.
pub fn extract_pdf_text_document(bytes: &[u8]) -> anyhow::Result<PdfTextDocument> {
    let document = load_document(bytes)?;
    let page_count = document.get_pages().len();
    let mut text = String::new();
    {
        let mut output = PlainTextOutput::new(&mut text);
        output_doc(&document, &mut output)?;
    }
    Ok(PdfTextDocument {
        text: normalize(text),
        page_count,
    })
}

/// Extract the text layer from specific 1-based physical pages.
///
/// `requested` may repeat pages and need not be sorted; callers that want a
/// user-facing range should normalize it first. An out-of-range page is an
/// error so the caller can tell the model the document's real page count.
pub fn extract_pdf_pages(bytes: &[u8], requested: &[usize]) -> anyhow::Result<PdfPages> {
    let document = load_document(bytes)?;
    let page_map = document.get_pages();
    let page_numbers: Vec<u32> = page_map.keys().copied().collect();
    let page_count = page_numbers.len();

    let mut pages = Vec::with_capacity(requested.len());
    for &page in requested {
        if page == 0 || page > page_count {
            anyhow::bail!(
                "page {} is out of range (document has {} pages)",
                page,
                page_count
            );
        }
        let page_number = page_numbers[page - 1];
        let mut text = String::new();
        {
            let mut output = PlainTextOutput::new(&mut text);
            output_doc_page(&document, &mut output, page_number)
                .map_err(|error| anyhow::anyhow!("failed to extract page {}: {}", page, error))?;
        }
        pages.push(PdfPageText {
            page,
            text: normalize(text),
        });
    }

    Ok(PdfPages { page_count, pages })
}

/// Load a PDF, repairing a broken xref/trailer once when necessary.
///
/// Some generators write a broken `startxref`/trailer or an incremental-update
/// chain whose `/Prev` points at a trailer instead of an xref table. That makes
/// `pdf-extract`'s pinned lopdf reject the file ("Invalid file trailer" /
/// "Invalid cross-reference table"). Newer lopdf recovers by scanning indirect
/// objects, so a reserialized copy is retried once before giving up. The
/// reserialized copy must drop `/Prev`/`XRefStm` (their offsets describe the
/// original layout) and the recovered xref root.
fn load_document(bytes: &[u8]) -> anyhow::Result<Document> {
    let mut document = match Document::load_mem(bytes) {
        Ok(document) => document,
        Err(primary) => {
            let repaired = repair_document(bytes)
                .map_err(|repair| anyhow::anyhow!("{primary}; repair failed: {repair}"))?;
            Document::load_mem(&repaired)
                .map_err(|error| anyhow::anyhow!("{primary}; repaired load failed: {error}"))?
        }
    };

    if document.is_encrypted() {
        document
            .decrypt("")
            .map_err(|error| anyhow::anyhow!("encrypted PDF requires a password: {error}"))?;
    }

    Ok(document)
}

fn repair_document(bytes: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut document = lopdf::Document::load_mem(bytes)?;
    document.trailer.remove(b"Prev");
    document.trailer.remove(b"XRefStm");
    document.xref_start = 0;
    let mut repaired = Vec::new();
    document.save_to(&mut repaired)?;
    Ok(repaired)
}

/// PDF text layers commonly use CJK compatibility ideographs; fold them to
/// the unified code points models are trained on.
fn normalize(text: String) -> String {
    text.nfkc().collect()
}
