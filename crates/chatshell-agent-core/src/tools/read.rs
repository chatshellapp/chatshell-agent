//! Read tool for LLM agents
//!
//! A general-purpose file reading tool. Supports text files with line numbers,
//! image files (PNG, JPEG, GIF, WebP) with metadata and base64 encoding,
//! and PDF files with page-aware text extraction.

use std::path::PathBuf;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rig::agent::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::image_question::{ImageInput, ImageQuestionResponder};
use crate::path_policy;
use crate::tool_fs::{Fs, StdFs};
use futures::future::BoxFuture;

const DEFAULT_LINE_LIMIT: usize = 2000;
const MAX_LINE_LENGTH: usize = 2000;
const BINARY_CHECK_SIZE: usize = 8192;
const MAX_IMAGE_SIZE: u64 = 20 * 1024 * 1024; // 20 MB
/// Raw-bytes cap for non-image URL reads (text, JSON, XML, …).
const MAX_URL_TEXT_BYTES: usize = 100 * 1024;

#[derive(Debug, Clone, Deserialize)]
pub struct ReadArgs {
    /// Absolute path to the file to read
    pub path: String,
    /// Line number to start reading from. Positive values are 1-indexed from
    /// the start; negative values count backwards from the end (e.g. -10 reads
    /// the last 10 lines). Defaults to 1.
    #[serde(default)]
    pub offset: Option<i64>,
    /// Number of lines to read. Defaults to 2000.
    #[serde(default)]
    pub limit: Option<usize>,
    /// PDF page selection: a single page (`"3"`), a range (`"1-5"`), or a
    /// comma-separated list (`"1,4,7"`). Pages are 1-based physical pages.
    /// Only applies to PDF files; `offset`/`limit` are ignored when set.
    #[serde(default)]
    pub pages: Option<String>,
    /// Optional natural-language question about an image file. When set,
    /// the image is analyzed by a vision-capable model and the answer is
    /// returned as text instead of the image data. Only supported for
    /// image files; `offset`/`limit` are ignored.
    #[serde(default)]
    pub question: Option<String>,
    /// Worksheet name for `.xlsx`/`.xlsm`. Omit to preview the first sheet.
    #[serde(default)]
    pub sheet: Option<String>,
    /// A1-style window for `.xlsx`/`.xlsm` (e.g. `"A1:F40"`). Capped at
    /// 200 rows × 100 columns per read.
    #[serde(default)]
    pub range: Option<String>,
    /// For `.xlsx`/`.xlsm`: show formula text instead of cached values
    /// where a formula exists.
    #[serde(default)]
    pub formulas: bool,
}

#[derive(Debug, thiserror::Error)]
#[error("Read error: {0}")]
pub struct ReadError(String);

/// Bytes fetched from a remote URL by a host-injected [`UrlByteFetcher`].
#[derive(Debug, Clone)]
pub struct UrlBytes {
    pub bytes: Vec<u8>,
    /// Response `Content-Type` (may carry parameters; callers normalize).
    pub mime_type: String,
    /// Final URL after redirects.
    pub final_url: String,
}

/// Host-injected remote URL fetcher for the read tool. The core crate owns
/// no network client, so the host supplies one; injected only when the
/// web_fetch tool is enabled, which doubles as the settings gate.
pub trait UrlByteFetcher: Send + Sync {
    fn fetch(&self, url: &str) -> BoxFuture<'static, Result<UrlBytes, String>>;
}

fn default_fs() -> std::sync::Arc<dyn Fs> {
    std::sync::Arc::new(StdFs)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ReadTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    project_root: Option<PathBuf>,
    #[serde(skip, default = "default_fs")]
    fs: Arc<dyn Fs>,
    /// Host-injected vision-model call for image questions.
    #[serde(skip, default)]
    image_question: Option<Arc<dyn ImageQuestionResponder>>,
    /// Whether the conversation's main model accepts image input. `None`
    /// keeps legacy inline behavior; `Some(false)` returns metadata plus
    /// guidance instead of pixels the main model cannot use.
    #[serde(skip, default)]
    main_model_vision: Option<bool>,
    /// Host-injected remote URL fetcher; `None` disables URL reads (the
    /// gate: injected only when web_fetch is enabled for the conversation).
    #[serde(skip, default)]
    url_fetcher: Option<Arc<dyn UrlByteFetcher>>,
}

impl Default for ReadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadTool {
    pub fn new() -> Self {
        Self {
            project_root: None,
            fs: Arc::new(StdFs),
            image_question: None,
            main_model_vision: None,
            url_fetcher: None,
        }
    }

    pub fn with_project_root(root: PathBuf) -> Self {
        Self {
            project_root: Some(root),
            ..Self::new()
        }
    }

    /// Route IO through a custom [`Fs`] implementation.
    pub fn with_fs(mut self, fs: Arc<dyn Fs>) -> Self {
        self.fs = fs;
        self
    }

    /// Route image questions through a host-provided vision model.
    pub fn with_image_question(mut self, responder: Arc<dyn ImageQuestionResponder>) -> Self {
        self.image_question = Some(responder);
        self
    }

    /// Declare whether the conversation's main model accepts image input.
    /// When `false`, plain image reads return metadata plus guidance
    /// instead of base64 pixels the main model would drop.
    pub fn with_main_model_vision(mut self, supports: bool) -> Self {
        self.main_model_vision = Some(supports);
        self
    }

    /// Enable remote URL reads through a host-injected fetcher. Without
    /// it, URL paths fail with a gate message pointing at web_fetch.
    pub fn with_url_fetcher(mut self, fetcher: Arc<dyn UrlByteFetcher>) -> Self {
        self.url_fetcher = Some(fetcher);
        self
    }

    /// Answer a question about an image via the injected vision responder.
    async fn answer_image_question(
        &self,
        path: &std::path::Path,
        ext: &str,
        question: &str,
    ) -> Result<String, ReadError> {
        let image = load_image(self.fs.as_ref(), path, ext)?;
        let label = path.display().to_string();
        self.answer_image_question_bytes(image, &label, question)
            .await
    }

    /// In-memory core of the image-question path, shared by file and URL
    /// reads. Failures are tool output, not `Err` (see the note in the
    /// match arm below).
    async fn answer_image_question_bytes(
        &self,
        image: LoadedImage,
        label: &str,
        question: &str,
    ) -> Result<String, ReadError> {
        let Some(responder) = self.image_question.as_ref() else {
            return Ok(format!(
                "{}\nNo vision model is configured to answer image questions. \
                 Configure one under Settings → Auxiliary Models, or read this \
                 file again without \"question\".",
                image.header
            ));
        };

        tracing::info!(
            "🔧 [tool-call] read: image question \"{}\" ({})",
            label,
            question
        );
        let input = ImageInput {
            bytes: image.bytes,
            mime_type: image.mime.to_string(),
        };
        match responder.answer(input, question.to_string()).await {
            Ok(answer) => {
                tracing::info!("🔧 [tool-result] read: image question answered");
                Ok(format!(
                    "{} (analyzed by a vision model)\nQuestion: {}\nAnswer:\n{}",
                    image.header, question, answer
                ))
            }
            // Return the failure as tool output, not an Err: host runtimes
            // (e.g. rig) replace tool errors with generic feedback like
            // "the tool failed", hiding the cause and the remedy from the
            // model — the exact strings it needs to stop retrying blindly.
            Err(error) => Ok(format!(
                "{}\nImage question failed: {}. The vision model may be \
                 misconfigured or unreachable; check Settings → Auxiliary \
                 Models. Read the file again without \"question\" for its \
                 normal content.",
                image.header, error
            )),
        }
    }

    /// Read a remote URL: fetch bytes through the injected seam and
    /// dispatch by content type — images take the same paths as local
    /// image files, text-ish content returns capped raw text, anything
    /// else steers to `web_fetch`.
    ///
    /// Every URL-path failure is returned as tool output, not `Err`:
    /// host runtimes (e.g. rig) replace tool errors with generic feedback
    /// like "the tool failed", hiding the cause and the remedy from the
    /// model — the exact strings it needs to stop retrying blindly.
    async fn read_url(&self, args: &ReadArgs) -> Result<String, ReadError> {
        let Some(fetcher) = self.url_fetcher.as_ref() else {
            return Ok(
                "URL reads require the web fetch tool to be enabled for this \
                 conversation. Enable web_fetch in the conversation's tools, \
                 or read a local file path instead."
                    .to_string(),
            );
        };
        let fetched = match fetcher.fetch(&args.path).await {
            Ok(fetched) => fetched,
            Err(error) => {
                return Ok(format!(
                    "URL fetch failed: {error}\n\nThe request itself failed; \
                     HTTP 4xx usually means the URL is invalid or refuses \
                     this client (e.g. a thumbnail size that was never \
                     rendered). Double-check the URL, try the web_fetch \
                     tool, or use bash for resources needing different \
                     headers."
                ));
            }
        };
        let mime = fetched
            .mime_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_lowercase();

        if mime.starts_with("image/") {
            return self.read_url_image(&fetched, &mime, args).await;
        }
        if mime.starts_with("text/")
            || mime == "application/json"
            || mime == "application/xml"
            || mime == "application/javascript"
            || mime.is_empty()
        {
            return Ok(read_url_text(&fetched.final_url, &fetched.bytes));
        }
        Ok(format!(
            "URL content type \"{mime}\" is not readable directly. Use the \
             web_fetch tool for this resource."
        ))
    }

    /// Image URL: the same dispatch as local image files, in memory.
    async fn read_url_image(
        &self,
        fetched: &UrlBytes,
        mime: &str,
        args: &ReadArgs,
    ) -> Result<String, ReadError> {
        let ext = extension_from_mime(mime);
        // The declared MIME is untrusted server data; a mismatch usually
        // means an error page served with an image content type. Local
        // files trust their extension; URL bytes must prove themselves.
        if !signature_matches_ext(&fetched.bytes, ext) {
            return Ok(format!(
                "The URL did not return a valid {ext} image: the declared \
                 content type \"{mime}\" does not match the body signature.\n\n\
                 The server may have answered with an error page despite an \
                 image content type, or the bytes are corrupt. Check the URL \
                 or use the web_fetch tool to inspect what the server \
                 actually returns."
            ));
        }
        let image = build_loaded_image(fetched.bytes.clone(), ext)?;

        // SVG is text-based; return the decoded content.
        if ext == "svg" {
            let svg_text = String::from_utf8_lossy(&image.bytes);
            return Ok(format!(
                "[Image: SVG, {}]\n\n{}",
                format_file_size(image.bytes.len() as u64),
                svg_text
            ));
        }

        let label = &fetched.final_url;
        if let Some(question) = args
            .question
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty())
        {
            return self
                .answer_image_question_bytes(image, label, question)
                .await;
        }
        if self.main_model_vision == Some(false) {
            return Ok(read_image_metadata_text(
                &image,
                self.image_question.is_some(),
            ));
        }
        Ok(read_image_inline(&image, label))
    }
}

// ---------------------------------------------------------------------------
// File-type detection helpers
// ---------------------------------------------------------------------------

fn is_image_extension(ext: &str) -> bool {
    matches!(
        ext,
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "ico" | "svg"
    )
}

fn is_pdf_extension(ext: &str) -> bool {
    ext == "pdf"
}

/// Whether raw bytes carry the magic signature of the claimed image
/// format. Unknown extensions (`bin`) cannot be verified and pass.
fn signature_matches_ext(bytes: &[u8], ext: &str) -> bool {
    match ext {
        "png" => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "jpg" | "jpeg" => bytes.starts_with(b"\xff\xd8\xff"),
        "gif" => bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a"),
        "webp" => bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP",
        "bmp" => bytes.starts_with(b"BM"),
        "ico" => bytes.starts_with(b"\x00\x00\x01\x00"),
        "svg" => {
            let prefix = &bytes[..bytes.len().min(4096)];
            let Ok(text) = std::str::from_utf8(prefix) else {
                return false;
            };
            let normalized = text.trim_start_matches('\u{feff}').trim_start();
            normalized.starts_with("<svg")
                || (normalized.starts_with("<?xml")
                    && normalized.to_ascii_lowercase().contains("<svg"))
        }
        _ => true,
    }
}

fn mime_from_extension(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "ico" => "image/x-icon",
        "svg" => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

/// Parse image dimensions from raw bytes without external dependencies.
fn parse_image_dimensions(bytes: &[u8], ext: &str) -> Option<(u32, u32)> {
    match ext {
        "png" => parse_png_dimensions(bytes),
        "jpg" | "jpeg" => parse_jpeg_dimensions(bytes),
        "gif" => parse_gif_dimensions(bytes),
        "webp" => parse_webp_dimensions(bytes),
        "bmp" => parse_bmp_dimensions(bytes),
        _ => None,
    }
}

fn parse_png_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    // PNG: bytes 16..20 = width, 20..24 = height (big-endian u32 in IHDR)
    if b.len() >= 24 && &b[..8] == b"\x89PNG\r\n\x1a\n" {
        let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
        let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
        Some((w, h))
    } else {
        None
    }
}

fn parse_jpeg_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    if b.len() < 2 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut i = 2;
    while i + 4 < b.len() {
        if b[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = b[i + 1];
        if marker == 0xD9 {
            break;
        }
        // SOF markers: 0xC0..=0xCF except 0xC4 (DHT) and 0xCC (DAC)
        if (0xC0..=0xCF).contains(&marker) && marker != 0xC4 && marker != 0xCC && i + 9 < b.len() {
            let h = u16::from_be_bytes([b[i + 5], b[i + 6]]) as u32;
            let w = u16::from_be_bytes([b[i + 7], b[i + 8]]) as u32;
            return Some((w, h));
        }
        let seg_len = u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize;
        i += 2 + seg_len;
    }
    None
}

fn parse_gif_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    // GIF: bytes 6..8 = width, 8..10 = height (little-endian u16)
    if b.len() >= 10 && (&b[..6] == b"GIF87a" || &b[..6] == b"GIF89a") {
        let w = u16::from_le_bytes([b[6], b[7]]) as u32;
        let h = u16::from_le_bytes([b[8], b[9]]) as u32;
        Some((w, h))
    } else {
        None
    }
}

fn parse_webp_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    if b.len() < 30 || &b[..4] != b"RIFF" || &b[8..12] != b"WEBP" {
        return None;
    }
    let chunk = &b[12..16];
    if chunk == b"VP8 " && b.len() >= 30 {
        // Lossy VP8: dimensions at offset 26..30
        let w = (u16::from_le_bytes([b[26], b[27]]) & 0x3FFF) as u32;
        let h = (u16::from_le_bytes([b[28], b[29]]) & 0x3FFF) as u32;
        Some((w, h))
    } else if chunk == b"VP8L" && b.len() >= 25 {
        // Lossless VP8L: packed bits at offset 21
        let bits = u32::from_le_bytes([b[21], b[22], b[23], b[24]]);
        let w = (bits & 0x3FFF) + 1;
        let h = ((bits >> 14) & 0x3FFF) + 1;
        Some((w, h))
    } else {
        None
    }
}

fn parse_bmp_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    // BMP: bytes 18..22 = width, 22..26 = height (little-endian i32)
    if b.len() >= 26 && &b[..2] == b"BM" {
        let w = i32::from_le_bytes([b[18], b[19], b[20], b[21]]).unsigned_abs();
        let h = i32::from_le_bytes([b[22], b[23], b[24], b[25]]).unsigned_abs();
        Some((w, h))
    } else {
        None
    }
}

fn format_file_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

// ---------------------------------------------------------------------------
// Tool implementation
// ---------------------------------------------------------------------------

impl Tool for ReadTool {
    const NAME: &'static str = "read";

    type Error = ReadError;
    type Args = ReadArgs;
    type Output = String;

    fn description(&self) -> String {
        "Read the contents of a file from the filesystem or a remote URL. \
                Local paths: for text files, returns content with line numbers; \
                for image files (PNG, JPEG, GIF, WebP, BMP), returns metadata and base64-encoded data; \
                for PDF files, extracts and returns the text content; use \"pages\" for a \
                1-based page selection (e.g. \"3\", \"1-5\", \"1,4,7\"); \
                for Office files, .docx/.pptx return Markdown (headings, lists, tables, slide text) \
                and .xlsx/.xlsm return a bounded TSV view with a sheet list; use \"sheet\" and \
                \"range\" (A1 style) to page spreadsheets, and \"formulas\" to show formulas; \
                legacy .doc/.xls/.ppt are not supported. \
                supports line ranges and negative offset for text files (e.g. -20 reads the last 20 lines). \
                URLs (http/https): image URLs return metadata and base64-encoded data (or a \
                vision-model answer when the \"question\" argument is set); text/JSON/XML URLs \
                return the raw text (capped); use web_fetch for article extraction or binary \
                content. For image files or image URLs, the optional \"question\" argument \
                returns a text answer from a vision-capable model instead of the image data."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a local file, or an HTTP/HTTPS URL"
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from. Positive values are 1-indexed from the start. Negative values count from the end (e.g. -10 reads the last 10 lines). Defaults to 1. Only applies to text files."
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read. Defaults to 2000. Only applies to text files."
                },
                "pages": {
                    "type": "string",
                    "description": "PDF page selection, 1-based. Use a single page (\"3\"), a range (\"1-5\"), or a comma-separated list (\"1,4,7\"). Only applies to PDF files; offset/limit are ignored when set. Omit to read the whole document."
                },
                "question": {
                    "type": "string",
                    "description": "Optional natural-language question about an image file or image URL. When set, the image is analyzed by a vision model and the answer is returned as text instead of the image data. Only supported for images; offset/limit are ignored."
                },
                "sheet": {
                    "type": "string",
                    "description": "Worksheet name for .xlsx/.xlsm files. Omit to preview the first sheet and list the available sheets."
                },
                "range": {
                    "type": "string",
                    "description": "A1-style cell window for .xlsx/.xlsm files, e.g. \"A1:F40\" or \"B2\". Capped at 200 rows by 100 columns per read."
                },
                "formulas": {
                    "type": "boolean",
                    "description": "For .xlsx/.xlsm files: show formula text where a formula exists instead of the cached value."
                }
            },
            "required": ["path"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::agent::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        tracing::info!(
            "🔧 [tool-call] read: path=\"{}\" offset={:?} limit={:?} pages={:?} question={:?}",
            args.path,
            args.offset,
            args.limit,
            args.pages,
            args.question
        );

        // Remote URLs bypass local path policy: fetch raw bytes through
        // the injected seam, then dispatch by content type.
        if args.path.starts_with("http://") || args.path.starts_with("https://") {
            if args.pages.is_some() {
                return Ok("The \"pages\" argument only applies to local PDF files. \
                     Read the URL without \"pages\" instead; remote PDFs are \
                     not supported by the read tool."
                    .to_string());
            }
            return self.read_url(&args).await;
        }

        let path = std::path::Path::new(&args.path);

        path_policy::check_read(path, self.project_root.as_deref()).map_err(ReadError)?;
        if !path.exists() {
            return Err(ReadError(format!("File not found: {}", args.path)));
        }
        if path.is_dir() {
            return Err(ReadError(format!(
                "Path is a directory, not a file: {}. \
             Use the glob tool to list files or the bash tool with ls \
             to explore directory contents.",
                args.path
            )));
        }
        if !path.is_file() {
            return Err(ReadError(format!("Not a regular file: {}", args.path)));
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if args.pages.is_some() && !is_pdf_extension(&ext) {
            return Err(ReadError(format!(
                "The \"pages\" argument only supports PDF files, not \"{}\" \
                 files. Use \"offset\"/\"limit\" for text files.",
                ext
            )));
        }

        // Image questions: answer via the vision seam instead of returning
        // pixels. Only images qualify; anything else is an explicit error.
        let question = args
            .question
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty());
        if let Some(question) = question {
            if !is_image_extension(&ext) {
                return Err(ReadError(format!(
                    "The \"question\" argument only supports image files, not \
                     \"{}\" files. Read the file without \"question\" for its \
                     normal content.",
                    ext
                )));
            }
            return self.answer_image_question(path, &ext, question).await;
        }

        // Dispatch by file type
        if is_image_extension(&ext) {
            // SVG source is plain text any model can read; raster pixels are
            // useless to a main model without image input — return metadata
            // and point at the question path instead.
            if ext != "svg" && self.main_model_vision == Some(false) {
                return read_image_metadata(
                    self.fs.as_ref(),
                    path,
                    &ext,
                    self.image_question.is_some(),
                );
            }
            return read_image(self.fs.as_ref(), path, &ext);
        }
        if is_pdf_extension(&ext) {
            return read_pdf(self.fs.as_ref(), path, &args);
        }
        if crate::office::is_legacy_office_extension(&ext) {
            return Err(ReadError(crate::office::legacy_message(&ext)));
        }
        if crate::office::is_office_extension(&ext) {
            return read_office(self.fs.as_ref(), path, &ext, &args);
        }

        read_text(self.fs.as_ref(), path, &args)
    }
}

/// Read a text file with line numbers and offset/limit support.
fn read_text(fs: &dyn Fs, path: &std::path::Path, args: &ReadArgs) -> Result<String, ReadError> {
    let raw_bytes = fs
        .read(path)
        .map_err(|e| ReadError(format!("Failed to read file: {}", e)))?;

    let check_len = raw_bytes.len().min(BINARY_CHECK_SIZE);
    if raw_bytes[..check_len].contains(&0) {
        return Err(ReadError(format!(
            "Binary file detected, cannot read as text: {}",
            args.path
        )));
    }

    let content = String::from_utf8(raw_bytes)
        .map_err(|_| ReadError(format!("File is not valid UTF-8: {}", args.path)))?;

    // Mint the content anchor for the edit tool ([path#TAG] protocol).
    let anchor_header = crate::anchor_store::record_anchor(path, &content)
        .map(|tag| format!("{}\n", crate::edit_anchor::tag_header(&args.path, &tag)));

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let limit = args.limit.unwrap_or(DEFAULT_LINE_LIMIT);

    // Resolve offset (supports negative values)
    let start_idx = match args.offset {
        Some(off) if off < 0 => {
            // Negative: count from end. -1 means last line, -10 means last 10 lines.
            let from_end = (-off) as usize;
            total_lines.saturating_sub(from_end)
        }
        Some(off) => {
            let off = off.max(1) as usize;
            (off - 1).min(total_lines)
        }
        None => 0,
    };

    let end_idx = (start_idx + limit).min(total_lines);
    let selected = &lines[start_idx..end_idx];

    let mut output = String::new();
    if let Some(header) = anchor_header {
        output.push_str(&header);
    }

    for (i, line) in selected.iter().enumerate() {
        let line_num = start_idx + i + 1;
        let display_line = if line.len() > MAX_LINE_LENGTH {
            let mut end = MAX_LINE_LENGTH;
            while end > 0 && !line.is_char_boundary(end) {
                end -= 1;
            }
            format!(
                "{}... (line truncated, {} chars total)",
                &line[..end],
                line.len()
            )
        } else {
            line.to_string()
        };
        output.push_str(&format!("{:>6}\t{}\n", line_num, display_line));
    }

    // Recovery footers: the model must always know what it did not see and
    // the exact next call to fetch it (bare ellipses make models guess
    // selectors and burn a turn).
    if start_idx > 0 && end_idx < total_lines {
        output.push_str(&format!(
            "\n(showing lines {}-{} of {}; {} before, {} after)",
            start_idx + 1,
            end_idx,
            total_lines,
            start_idx,
            total_lines - end_idx
        ));
    }
    if end_idx < total_lines {
        output.push_str(&format!(
            "\n... ({} more lines after line {} — continue with offset={})",
            total_lines - end_idx,
            end_idx,
            end_idx + 1
        ));
    }

    tracing::info!(
        "🔧 [tool-result] read: returned {} lines from \"{}\" (total: {})",
        selected.len(),
        args.path,
        total_lines
    );

    Ok(output)
}

/// Image bytes plus the shared `[Image: …]` metadata header.
struct LoadedImage {
    bytes: Vec<u8>,
    mime: &'static str,
    header: String,
}

/// Build a [`LoadedImage`] from in-memory bytes (size-capped): the shared
/// core of the file and URL image paths.
fn build_loaded_image(bytes: Vec<u8>, ext: &str) -> Result<LoadedImage, ReadError> {
    let file_size = bytes.len() as u64;
    if file_size > MAX_IMAGE_SIZE {
        return Err(ReadError(format!(
            "Image too large ({}, max {})",
            format_file_size(file_size),
            format_file_size(MAX_IMAGE_SIZE)
        )));
    }
    let dims = parse_image_dimensions(&bytes, ext);
    let dim_str = dims
        .map(|(w, h)| format!("{}x{}", w, h))
        .unwrap_or_else(|| "unknown".to_string());
    Ok(LoadedImage {
        mime: mime_from_extension(ext),
        header: format!(
            "[Image: {}, {}, {}]",
            ext.to_uppercase(),
            dim_str,
            format_file_size(file_size)
        ),
        bytes,
    })
}

/// Load an image (size-capped) and render its metadata header.
fn load_image(fs: &dyn Fs, path: &std::path::Path, ext: &str) -> Result<LoadedImage, ReadError> {
    let metadata = fs
        .metadata(path)
        .map_err(|e| ReadError(format!("Failed to read image metadata: {}", e)))?;
    if metadata.len > MAX_IMAGE_SIZE {
        return Err(ReadError(format!(
            "Image too large ({}, max {})",
            format_file_size(metadata.len),
            format_file_size(MAX_IMAGE_SIZE)
        )));
    }
    let raw_bytes = fs
        .read(path)
        .map_err(|e| ReadError(format!("Failed to read image: {}", e)))?;
    build_loaded_image(raw_bytes, ext)
}

/// Read an image file: return metadata and base64-encoded data.
fn read_image(fs: &dyn Fs, path: &std::path::Path, ext: &str) -> Result<String, ReadError> {
    let image = load_image(fs, path, ext)?;

    // SVG files are text-based; return content directly instead of base64
    if ext == "svg" {
        let svg_text = String::from_utf8_lossy(&image.bytes);
        let size = format_file_size(image.bytes.len() as u64);
        tracing::info!(
            "🔧 [tool-result] read: SVG image \"{}\" ({})",
            path.display(),
            size
        );
        return Ok(format!("[Image: SVG, {}]\n\n{}", size, svg_text));
    }

    Ok(read_image_inline(&image, &path.display().to_string()))
}

/// Inline base64 rendering shared by the file and URL image paths.
fn read_image_inline(image: &LoadedImage, label: &str) -> String {
    tracing::info!(
        "🔧 [tool-result] read: image \"{}\" ({})",
        label,
        image.header
    );
    format!(
        "{}\ndata:{};base64,{}",
        image.header,
        image.mime,
        BASE64.encode(&image.bytes)
    )
}

/// Image read for a main model that cannot accept image input: metadata
/// plus guidance instead of pixels.
fn read_image_metadata(
    fs: &dyn Fs,
    path: &std::path::Path,
    ext: &str,
    question_available: bool,
) -> Result<String, ReadError> {
    let image = load_image(fs, path, ext)?;
    tracing::info!(
        "🔧 [tool-result] read: image metadata \"{}\" (main model has no image input)",
        path.display()
    );
    Ok(read_image_metadata_text(&image, question_available))
}

/// Metadata + guidance rendering shared by the file and URL image paths.
fn read_image_metadata_text(image: &LoadedImage, question_available: bool) -> String {
    let guidance = if question_available {
        "To ask about this image, read it again with the \"question\" argument \
         (e.g. {\"path\": \"<this file or URL>\", \"question\": \"what does \
         this screenshot show?\"}) — a vision model answers and the answer \
         is returned as text."
    } else {
        "No vision model is configured, so this image cannot be analyzed. \
         Configure one under Settings → Auxiliary Models to enable image \
         questions."
    };

    format!(
        "{}\nThe current conversation model does not accept image input, so \
         the image content is omitted.\n{}",
        image.header, guidance
    )
}

/// Extension for a fetched image MIME type (inverse of
/// [`mime_from_extension`], restricted to the image set).
fn extension_from_mime(mime: &str) -> &'static str {
    match mime {
        "image/png" | "image/x-png" => "png",
        "image/jpeg" | "image/jpg" | "image/pjpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/bmp" | "image/x-bmp" => "bmp",
        "image/x-icon" | "image/vnd.microsoft.icon" => "ico",
        "image/svg+xml" => "svg",
        _ => "bin",
    }
}

/// Capped raw text rendering for text-ish URL content. A NUL byte in the
/// leading window means the declared MIME lied; steer to `web_fetch`.
fn read_url_text(final_url: &str, bytes: &[u8]) -> String {
    let check_len = bytes.len().min(BINARY_CHECK_SIZE);
    if bytes[..check_len].contains(&0) {
        return format!(
            "[URL: {}]\n\nThe response body is binary. Use the web_fetch tool \
             for this resource.",
            final_url
        );
    }
    let full = String::from_utf8_lossy(bytes);
    if full.len() <= MAX_URL_TEXT_BYTES {
        return format!("[URL: {}]\n\n{}", final_url, full);
    }
    let mut end = MAX_URL_TEXT_BYTES;
    while end > 0 && !full.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "[URL: {}]\n\n{}\n\n[URL content truncated at {} bytes — use \
         web_fetch to extract the full page]",
        final_url,
        &full[..end],
        MAX_URL_TEXT_BYTES
    )
}

/// Maximum pages a single `pages` request may select. Guards against a
/// pathological range (e.g. `1-99999999`) exhausting memory before the
/// out-of-range check runs.
const MAX_PDF_PAGES_PER_READ: usize = 200;

/// Parse a PDF page selection: `"3"`, `"1-5"`, or `"1,4,7"` (combinations
/// such as `"1-3,7"` are accepted). Pages are 1-based; duplicates are
/// dropped while preserving first-seen order.
fn parse_page_selection(raw: &str) -> Result<Vec<usize>, ReadError> {
    let mut pages = Vec::new();

    for segment in raw.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            return Err(ReadError(format!(
                "Invalid \"pages\" value \"{}\": empty page or range.",
                raw
            )));
        }

        let (start, end) = match segment.split_once('-') {
            Some((start, end)) => (parse_page_number(start, raw)?, parse_page_number(end, raw)?),
            None => {
                let page = parse_page_number(segment, raw)?;
                (page, page)
            }
        };

        if start > end {
            return Err(ReadError(format!(
                "Invalid \"pages\" range \"{}\": start page {} is greater than \
                 end page {}.",
                segment, start, end
            )));
        }
        if end - start + 1 > MAX_PDF_PAGES_PER_READ {
            return Err(ReadError(format!(
                "The \"pages\" range \"{}\" selects more than {} pages in one \
                 read. Split it into smaller ranges.",
                segment, MAX_PDF_PAGES_PER_READ
            )));
        }

        for page in start..=end {
            if !pages.contains(&page) {
                pages.push(page);
            }
        }
        if pages.len() > MAX_PDF_PAGES_PER_READ {
            return Err(ReadError(format!(
                "The \"pages\" value \"{}\" selects more than {} pages in one \
                 read. Split it into smaller ranges.",
                raw, MAX_PDF_PAGES_PER_READ
            )));
        }
    }

    if pages.is_empty() {
        return Err(ReadError(format!(
            "Invalid \"pages\" value \"{}\": no pages selected.",
            raw
        )));
    }
    Ok(pages)
}

fn parse_page_number(raw: &str, selection: &str) -> Result<usize, ReadError> {
    raw.trim()
        .parse::<usize>()
        .ok()
        .filter(|page| *page > 0)
        .ok_or_else(|| {
            ReadError(format!(
                "Invalid \"pages\" value \"{}\": \"{}\" is not a positive page \
                 number.",
                selection,
                raw.trim()
            ))
        })
}

/// Render a page list back into compact ranges for the tool header.
fn format_page_selection(pages: &[usize]) -> String {
    let mut parts = Vec::new();
    let mut index = 0;
    while index < pages.len() {
        let start = pages[index];
        let mut end = start;
        while index + 1 < pages.len() && pages[index + 1] == end + 1 {
            index += 1;
            end = pages[index];
        }
        if start == end {
            parts.push(start.to_string());
        } else {
            parts.push(format!("{start}-{end}"));
        }
        index += 1;
    }
    parts.join(", ")
}

/// Read a PDF file by extracting its text content. When `args.pages` is set,
/// only those pages are extracted and returned with page markers.
fn read_pdf(fs: &dyn Fs, path: &std::path::Path, args: &ReadArgs) -> Result<String, ReadError> {
    let bytes = fs
        .read(path)
        .map_err(|e| ReadError(format!("Failed to read PDF: {}", e)))?;

    // Legacy attachment storage wrote extracted PDF text under a `.pdf`
    // name. Treat a non-PDF payload as text so those paths stay readable,
    // but page selection needs the original PDF bytes.
    let header = &bytes[..bytes.len().min(1024)];
    if !header.windows(5).any(|window| window == b"%PDF-") {
        if args.pages.is_some() {
            return Err(ReadError(format!(
                "File has a .pdf extension but contains extracted text, not PDF \
                 bytes: {}. The \"pages\" argument needs the original PDF; use \
                 \"offset\"/\"limit\" for this file.",
                path.display()
            )));
        }
        let text = String::from_utf8(bytes).map_err(|_| {
            ReadError(format!(
                "File has a .pdf extension but is neither a PDF nor UTF-8 text: {}",
                path.display()
            ))
        })?;
        let line_count = text.lines().count();
        return Ok(format!("[PDF: {} lines extracted]\n\n{}", line_count, text));
    }

    if let Some(selection) = args.pages.as_deref() {
        let requested = parse_page_selection(selection)?;
        let extracted = crate::pdf_text::extract_pdf_pages(&bytes, &requested)
            .map_err(|e| ReadError(format!("Failed to extract PDF text: {}", e)))?;

        let shown = format_page_selection(
            &extracted
                .pages
                .iter()
                .map(|page| page.page)
                .collect::<Vec<_>>(),
        );
        let mut output = format!(
            "[PDF: {} pages; showing page{} {}]\n",
            extracted.page_count,
            if extracted.pages.len() == 1 { "" } else { "s" },
            shown
        );
        for page in &extracted.pages {
            output.push_str(&format!("\n--- page {} ---\n", page.page));
            if page.text.trim().is_empty() {
                output.push_str("[No extractable text on this page]\n");
            } else {
                output.push_str(page.text.trim_end());
                output.push('\n');
            }
        }

        tracing::info!(
            "🔧 [tool-result] read: PDF \"{}\" ({} pages extracted, {} requested)",
            path.display(),
            extracted.pages.len(),
            requested.len()
        );
        return Ok(output);
    }

    let extracted = crate::pdf_text::extract_pdf_text_document(&bytes)
        .map_err(|e| ReadError(format!("Failed to extract PDF text: {}", e)))?;
    let text = extracted.text;

    if text.trim().is_empty() {
        return Ok(format!(
            "[PDF] No extractable text found ({} pages; the PDF may contain only images).",
            extracted.page_count
        ));
    }

    let line_count = text.lines().count();

    tracing::info!(
        "🔧 [tool-result] read: PDF \"{}\" ({} pages, {} lines extracted)",
        path.display(),
        extracted.page_count,
        line_count
    );

    Ok(format!(
        "[PDF: {} pages, {} lines extracted]\n\n{}",
        extracted.page_count, line_count, text
    ))
}

/// Read an OOXML office document. Format-specific extraction lives in
/// `crate::office`; this function owns the tool-level contract (which
/// arguments apply, and how text is windowed).
fn read_office(
    fs: &dyn Fs,
    path: &std::path::Path,
    ext: &str,
    args: &ReadArgs,
) -> Result<String, ReadError> {
    let bytes = fs
        .read(path)
        .map_err(|e| ReadError(format!("Failed to read file: {}", e)))?;

    match ext {
        "xlsx" | "xlsm" => {
            if args.offset.is_some() || args.limit.is_some() {
                return Err(ReadError(
                    "offset/limit do not apply to spreadsheets; use \"range\" \
                     (e.g. A1:F40) to page a sheet."
                        .to_string(),
                ));
            }
            crate::office::read_xlsx(
                &bytes,
                args.sheet.as_deref(),
                args.range.as_deref(),
                args.formulas,
            )
            .map_err(|e| ReadError(e.to_string()))
        }
        "docx" | "docm" => {
            reject_spreadsheet_args(ext, args)?;
            let markdown =
                crate::office::read_docx(&bytes).map_err(|e| ReadError(e.to_string()))?;
            Ok(window_markdown("DOCX", &markdown, args))
        }
        "pptx" | "pptm" => {
            reject_spreadsheet_args(ext, args)?;
            let text = crate::office::read_pptx(&bytes).map_err(|e| ReadError(e.to_string()))?;
            Ok(window_markdown("PPTX", &text, args))
        }
        _ => Err(ReadError(format!(
            "Unsupported office extension \".{ext}\"."
        ))),
    }
}

fn reject_spreadsheet_args(ext: &str, args: &ReadArgs) -> Result<(), ReadError> {
    if args.sheet.is_some() || args.range.is_some() || args.formulas {
        return Err(ReadError(format!(
            "\"sheet\", \"range\", and \"formulas\" only apply to .xlsx/.xlsm \
             files, not \".{ext}\"."
        )));
    }
    Ok(())
}

/// Apply `offset`/`limit` to extracted Markdown/PPTX text. Line numbers are
/// omitted so the text stays copy-pasteable as Markdown.
fn window_markdown(kind: &str, text: &str, args: &ReadArgs) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let limit = args.limit.unwrap_or(DEFAULT_LINE_LIMIT).max(1);
    let start = match args.offset {
        Some(offset) if offset < 0 => total.saturating_sub((-offset) as usize),
        Some(offset) => (offset.max(1) as usize - 1).min(total),
        None => 0,
    };
    let end = (start + limit).min(total);

    let mut output = format!("[{kind}: {total} lines]\n\n");
    for line in &lines[start..end] {
        output.push_str(line);
        output.push('\n');
    }
    if start > 0 || end < total {
        output.push_str(&format!(
            "\n(showing lines {}-{} of {}; use offset/limit to page)\n",
            start + 1,
            end,
            total
        ));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    fn read(path: &std::path::Path, extra: serde_json::Value) -> String {
        let mut map = serde_json::Map::new();
        map.insert("path".to_string(), path.to_string_lossy().into());
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                map.insert(k.clone(), v.clone());
            }
        }
        let args: ReadArgs = serde_json::from_value(serde_json::Value::Object(map)).unwrap();
        read_text(&StdFs, path, &args).unwrap()
    }

    #[test]
    fn tail_footer_gives_next_offset() {
        let (_dir, path) = write_temp(&"line\n".repeat(3000));
        let out = read(&path, serde_json::json!({}));

        assert!(out.contains("1000 more lines after line 2000"));
        assert!(out.contains("continue with offset=2001"), "footer: {out}");
    }

    #[test]
    fn mid_file_read_shows_both_sides() {
        let (_dir, path) = write_temp(&"line\n".repeat(100));
        let out = read(&path, serde_json::json!({ "offset": 50, "limit": 10 }));

        assert!(out.contains("showing lines 50-59 of 100"), "header: {out}");
        assert!(out.contains("41 more lines after line 59"));
        assert!(out.contains("continue with offset=60"));
    }

    #[test]
    fn long_line_truncation_is_char_safe_and_annotated() {
        for tail in ["尾", "あ", "한"] {
            let long_line = format!("{}{tail}", "a".repeat(3000));
            let (_dir, path) = write_temp(&long_line);
            let out = read(&path, serde_json::json!({}));

            assert!(
                out.contains("(line truncated, "),
                "annotation present: {}",
                &out[out.len().saturating_sub(200)..]
            );
            assert!(!out.contains('\u{FFFD}'), "no replacement chars");
        }
    }

    #[tokio::test]
    async fn pdf_extension_with_extracted_text_reads_as_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.pdf");
        std::fs::write(&path, "first line\nsecond line\n").unwrap();

        let args = ReadArgs {
            path: path.to_string_lossy().into_owned(),
            offset: None,
            limit: None,
            pages: None,
            question: None,
            sheet: None,
            range: None,
            formulas: false,
        };
        let out = call_read(&ReadTool::new(), args).await.unwrap();

        assert!(out.contains("first line"), "out: {out}");
        assert!(out.contains("second line"), "out: {out}");
    }

    fn write_pdf(pages: &[&str]) -> (tempfile::TempDir, std::path::PathBuf) {
        use lopdf::{Document, Object, Stream, dictionary};

        let mut document = Document::with_version("1.5");
        let pages_id = document.new_object_id();
        let font_id = document.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let resources_id = document.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
        });

        let mut kids = Vec::new();
        for text in pages {
            let contents = if text.is_empty() {
                Vec::new()
            } else {
                format!("BT /F1 24 Tf 72 720 Td ({text}) Tj ET").into_bytes()
            };
            let content_id = document.add_object(Stream::new(dictionary! {}, contents));
            let page_id = document.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "Contents" => content_id,
                "Resources" => resources_id,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            });
            kids.push(Object::Reference(page_id));
        }

        document.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => Object::Array(kids),
                "Count" => pages.len() as i64,
            }),
        );
        let catalog_id = document.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        document.trailer.set("Root", catalog_id);

        let mut bytes = Vec::new();
        document.save_to(&mut bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pdf");
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    fn pdf_args(path: &std::path::Path, pages: Option<&str>) -> ReadArgs {
        ReadArgs {
            path: path.to_string_lossy().into_owned(),
            offset: None,
            limit: None,
            pages: pages.map(str::to_string),
            question: None,
            sheet: None,
            range: None,
            formulas: false,
        }
    }

    #[tokio::test]
    async fn docx_read_dispatches_and_reports_line_count() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.docx");
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("word/document.xml", options).unwrap();
            zip.write_all(
                br#"<w:document xmlns:w="w"><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Title</w:t></w:r></w:p></w:body></w:document>"#,
            )
            .unwrap();
            zip.finish().unwrap();
        }

        let args = ReadArgs {
            path: path.to_string_lossy().into_owned(),
            offset: None,
            limit: None,
            pages: None,
            question: None,
            sheet: None,
            range: None,
            formulas: false,
        };
        let out = call_read(&ReadTool::new(), args).await.unwrap();
        assert!(out.contains("[DOCX:"), "{out}");
        assert!(out.contains("# Title"), "{out}");
    }

    #[tokio::test]
    async fn legacy_office_extension_reports_conversion_hint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.doc");
        std::fs::write(&path, b"\xD0\xCF\x11\xE0").unwrap();

        let args = ReadArgs {
            path: path.to_string_lossy().into_owned(),
            offset: None,
            limit: None,
            pages: None,
            question: None,
            sheet: None,
            range: None,
            formulas: false,
        };
        let error = call_read(&ReadTool::new(), args).await.unwrap_err();
        assert!(
            error.to_string().contains("Legacy binary Office format"),
            "{error}"
        );
    }

    #[test]
    fn page_selection_parser_accepts_single_ranges_and_lists() {
        assert_eq!(parse_page_selection("3").unwrap(), vec![3]);
        assert_eq!(parse_page_selection("1-3").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_page_selection("1,4,7").unwrap(), vec![1, 4, 7]);
        assert_eq!(parse_page_selection("1-3,3,5").unwrap(), vec![1, 2, 3, 5]);
        assert_eq!(format_page_selection(&[1, 2, 3, 5]), "1-3, 5");
    }

    #[test]
    fn page_selection_parser_rejects_invalid_values() {
        for invalid in ["", "0", "-1", "5-1", "1,,2", "a", "1-a", "1-"] {
            assert!(
                parse_page_selection(invalid).is_err(),
                "accepted invalid selection: {invalid}"
            );
        }
        assert!(parse_page_selection("1-201").is_err());
    }

    #[tokio::test]
    async fn pdf_page_selection_returns_only_requested_pages() {
        let (_dir, path) = write_pdf(&[
            "BT /F1 24 Tf 72 720 Td (Alpha) Tj ET",
            "BT /F1 24 Tf 72 720 Td (Beta) Tj ET",
        ]);
        let out = call_read(&ReadTool::new(), pdf_args(&path, Some("2")))
            .await
            .unwrap();

        assert!(out.contains("[PDF: 2 pages; showing page 2]"), "{out}");
        assert!(out.contains("--- page 2 ---"), "{out}");
        assert!(out.contains("Beta"), "{out}");
        assert!(!out.contains("Alpha"), "{out}");
    }

    #[tokio::test]
    async fn pdf_page_range_returns_multiple_marked_pages() {
        let (_dir, path) = write_pdf(&[
            "BT /F1 24 Tf 72 720 Td (Alpha) Tj ET",
            "BT /F1 24 Tf 72 720 Td (Beta) Tj ET",
        ]);
        let out = call_read(&ReadTool::new(), pdf_args(&path, Some("1-2")))
            .await
            .unwrap();

        assert!(out.contains("[PDF: 2 pages; showing pages 1-2]"), "{out}");
        assert!(out.contains("--- page 1 ---"), "{out}");
        assert!(out.contains("Alpha"), "{out}");
        assert!(out.contains("--- page 2 ---"), "{out}");
        assert!(out.contains("Beta"), "{out}");
    }

    #[tokio::test]
    async fn pdf_without_pages_reads_the_whole_document() {
        let (_dir, path) = write_pdf(&[
            "BT /F1 24 Tf 72 720 Td (Alpha) Tj ET",
            "BT /F1 24 Tf 72 720 Td (Beta) Tj ET",
        ]);
        let out = call_read(&ReadTool::new(), pdf_args(&path, None))
            .await
            .unwrap();

        assert!(out.contains("[PDF: 2 pages,"), "{out}");
        assert!(out.contains("Alpha"), "{out}");
        assert!(out.contains("Beta"), "{out}");
    }

    #[tokio::test]
    async fn pdf_page_selection_marks_empty_pages() {
        let (_dir, path) = write_pdf(&["BT /F1 24 Tf 72 720 Td (Alpha) Tj ET", ""]);
        let out = call_read(&ReadTool::new(), pdf_args(&path, Some("2")))
            .await
            .unwrap();

        assert!(out.contains("--- page 2 ---"), "{out}");
        assert!(out.contains("[No extractable text on this page]"), "{out}");
    }

    #[tokio::test]
    async fn pdf_page_selection_rejects_out_of_range_pages() {
        let (_dir, path) = write_pdf(&["BT /F1 24 Tf 72 720 Td (Alpha) Tj ET"]);
        let err = call_read(&ReadTool::new(), pdf_args(&path, Some("2")))
            .await
            .unwrap_err();

        assert!(err.to_string().contains("out of range"), "error: {err}");
        assert!(err.to_string().contains("1 pages"), "error: {err}");
    }

    #[tokio::test]
    async fn pages_argument_on_text_file_is_an_explicit_error() {
        let (_dir, path) = write_temp("plain text\n");
        let err = call_read(&ReadTool::new(), pdf_args(&path, Some("1")))
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("only supports PDF files"),
            "error: {err}"
        );
    }

    #[tokio::test]
    async fn pages_argument_on_legacy_pdf_text_is_an_explicit_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.pdf");
        std::fs::write(&path, "extracted text\n").unwrap();

        let err = call_read(&ReadTool::new(), pdf_args(&path, Some("1")))
            .await
            .unwrap_err();

        assert!(
            err.to_string().contains("needs the original PDF"),
            "error: {err}"
        );
    }

    #[test]
    fn full_read_has_no_footer() {
        let (_dir, path) = write_temp("small\nfile\n");
        let out = read(&path, serde_json::json!({}));

        assert!(!out.contains("more lines"), "no recovery footer: {out}");
        assert!(!out.contains("showing lines"));
    }

    // ---- image question seam ----

    struct MockVisionResponder {
        fail: bool,
    }

    impl super::super::image_question::ImageQuestionResponder for MockVisionResponder {
        fn answer(
            &self,
            _image: super::super::image_question::ImageInput,
            _question: String,
        ) -> futures::future::BoxFuture<'static, Result<String, String>> {
            let fail = self.fail;
            Box::pin(async move {
                if fail {
                    Err("vision call failed".to_string())
                } else {
                    Ok("a sleeping cat".to_string())
                }
            })
        }
    }

    fn write_temp_image() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.png");
        std::fs::write(&path, b"fake-png-bytes").unwrap();
        (dir, path)
    }

    fn image_args(path: &std::path::Path, question: Option<&str>) -> ReadArgs {
        ReadArgs {
            path: path.to_string_lossy().to_string(),
            offset: None,
            limit: None,
            pages: None,
            question: question.map(str::to_string),
            sheet: None,
            range: None,
            formulas: false,
        }
    }

    async fn call_read(tool: &ReadTool, args: ReadArgs) -> Result<String, ReadError> {
        use rig::agent::tool::Tool as _;
        let mut ctx = rig::agent::tool::ToolContext::default();
        tool.call(&mut ctx, args).await
    }

    #[tokio::test]
    async fn question_on_text_file_is_an_explicit_error() {
        let (_dir, path) = write_temp("plain text\n");
        let tool = ReadTool::new()
            .with_image_question(std::sync::Arc::new(MockVisionResponder { fail: false }));

        let err = call_read(&tool, image_args(&path, Some("what is this?")))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("only supports image files"),
            "error: {err}"
        );
    }

    #[tokio::test]
    async fn question_on_image_returns_vision_answer_without_pixels() {
        let (_dir, path) = write_temp_image();
        let tool = ReadTool::new()
            .with_image_question(std::sync::Arc::new(MockVisionResponder { fail: false }));

        let out = call_read(&tool, image_args(&path, Some("what is shown?")))
            .await
            .unwrap();
        assert!(out.contains("a sleeping cat"), "answer: {out}");
        assert!(out.contains("Question: what is shown?"));
        assert!(!out.contains("base64"), "no pixel payload: {out}");
    }

    #[tokio::test]
    async fn question_without_responder_returns_guidance() {
        let (_dir, path) = write_temp_image();
        let tool = ReadTool::new();

        let out = call_read(&tool, image_args(&path, Some("what is shown?")))
            .await
            .unwrap();
        assert!(out.contains("[Image: PNG"), "metadata header: {out}");
        assert!(out.contains("No vision model is configured"));
    }

    #[tokio::test]
    async fn responder_failure_returns_output_with_recovery_hint() {
        let (_dir, path) = write_temp_image();
        let tool = ReadTool::new()
            .with_image_question(std::sync::Arc::new(MockVisionResponder { fail: true }));

        // Failure is tool output, not an Err: host runtimes replace tool
        // errors with generic feedback, hiding the cause from the model.
        let out = call_read(&tool, image_args(&path, Some("what is shown?")))
            .await
            .unwrap();
        assert!(out.contains("vision call failed"), "out: {out}");
        assert!(out.contains("without \"question\""), "out: {out}");
    }

    #[tokio::test]
    async fn text_only_main_model_gets_metadata_and_question_hint() {
        let (_dir, path) = write_temp_image();
        let tool = ReadTool::new()
            .with_main_model_vision(false)
            .with_image_question(std::sync::Arc::new(MockVisionResponder { fail: false }));

        let out = call_read(&tool, image_args(&path, None)).await.unwrap();
        assert!(out.contains("does not accept image input"), "out: {out}");
        assert!(out.contains("\"question\" argument"), "hint: {out}");
        assert!(!out.contains("base64"), "no pixel payload: {out}");
    }

    #[tokio::test]
    async fn text_only_main_model_without_responder_points_at_settings() {
        let (_dir, path) = write_temp_image();
        let tool = ReadTool::new().with_main_model_vision(false);

        let out = call_read(&tool, image_args(&path, None)).await.unwrap();
        assert!(out.contains("No vision model is configured"), "out: {out}");
        assert!(!out.contains("base64"));
    }

    #[tokio::test]
    async fn vision_capable_main_model_still_gets_inline_pixels() {
        let (_dir, path) = write_temp_image();
        let tool = ReadTool::new().with_main_model_vision(true);

        let out = call_read(&tool, image_args(&path, None)).await.unwrap();
        assert!(
            out.contains("data:image/png;base64,"),
            "inline pixels: {out}"
        );
    }

    #[tokio::test]
    async fn legacy_tool_without_vision_flag_keeps_inline_pixels() {
        let (_dir, path) = write_temp_image();
        let tool = ReadTool::new();

        let out = call_read(&tool, image_args(&path, None)).await.unwrap();
        assert!(
            out.contains("data:image/png;base64,"),
            "legacy behavior kept: {out}"
        );
    }

    // ---- remote URL reads ----

    struct MockFetcher {
        mime: &'static str,
        bytes: Vec<u8>,
        error: Option<&'static str>,
    }

    impl super::UrlByteFetcher for MockFetcher {
        fn fetch(
            &self,
            _url: &str,
        ) -> futures::future::BoxFuture<'static, Result<super::UrlBytes, String>> {
            let bytes = self.bytes.clone();
            let mime = self.mime.to_string();
            let error = self.error.map(str::to_string);
            Box::pin(async move {
                if let Some(error) = error {
                    return Err(error);
                }
                Ok(super::UrlBytes {
                    bytes,
                    mime_type: mime,
                    final_url: "https://example.com/final".to_string(),
                })
            })
        }
    }

    fn mock_fetcher(mime: &'static str, bytes: Vec<u8>) -> MockFetcher {
        MockFetcher {
            mime,
            bytes,
            error: None,
        }
    }

    fn url_args(url: &str, question: Option<&str>) -> ReadArgs {
        ReadArgs {
            path: url.to_string(),
            offset: None,
            limit: None,
            pages: None,
            question: question.map(str::to_string),
            sheet: None,
            range: None,
            formulas: false,
        }
    }

    const PNG_BYTES: &[u8] =
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0DIHDR\x00\x00\x00\x64\x00\x00\x00\x64";

    #[tokio::test]
    async fn url_read_without_fetcher_gates_on_web_fetch() {
        let tool = ReadTool::new();
        let out = call_read(&tool, url_args("https://example.com/a.png", None))
            .await
            .unwrap();
        assert!(
            out.contains("web fetch tool to be enabled"),
            "gate message: {out}"
        );
    }

    #[tokio::test]
    async fn url_fetch_failure_is_reported_as_output() {
        let tool = ReadTool::new().with_url_fetcher(std::sync::Arc::new(MockFetcher {
            mime: "image/png",
            bytes: Vec::new(),
            error: Some("HTTP error: 400 Bad Request"),
        }));
        let out = call_read(&tool, url_args("https://example.com/a.png", None))
            .await
            .unwrap();
        assert!(out.contains("URL fetch failed: HTTP error: 400"), "{out}");
        assert!(out.contains("HTTP 4xx"), "remedy hint present: {out}");
    }

    #[tokio::test]
    async fn url_image_with_invalid_bytes_is_reported_as_output() {
        let tool = ReadTool::new().with_url_fetcher(std::sync::Arc::new(mock_fetcher(
            "image/png",
            b"<!DOCTYPE html>not an image".to_vec(),
        )));
        let out = call_read(&tool, url_args("https://example.com/a.png", None))
            .await
            .unwrap();
        assert!(out.contains("valid png image"), "{out}");
        assert!(out.contains("error page despite"), "{out}");
    }

    #[tokio::test]
    async fn url_image_inline_for_vision_capable_main() {
        let tool = ReadTool::new()
            .with_main_model_vision(true)
            .with_url_fetcher(std::sync::Arc::new(mock_fetcher(
                "image/png",
                PNG_BYTES.to_vec(),
            )));
        let out = call_read(&tool, url_args("https://example.com/a.png", None))
            .await
            .unwrap();
        assert!(out.contains("data:image/png;base64,"), "out: {out}");
        assert!(out.contains("100x100"), "dims parsed: {out}");
    }

    #[tokio::test]
    async fn url_image_question_routes_to_vision_seam() {
        let tool = ReadTool::new()
            .with_image_question(std::sync::Arc::new(MockVisionResponder { fail: false }))
            .with_url_fetcher(std::sync::Arc::new(mock_fetcher(
                "image/png",
                PNG_BYTES.to_vec(),
            )));
        let out = call_read(
            &tool,
            url_args("https://example.com/a.png", Some("what is this?")),
        )
        .await
        .unwrap();
        assert!(out.contains("a sleeping cat"), "out: {out}");
        assert!(out.contains("Question: what is this?"), "out: {out}");
    }

    #[tokio::test]
    async fn url_image_metadata_fallback_for_text_only_main() {
        let tool = ReadTool::new()
            .with_main_model_vision(false)
            .with_url_fetcher(std::sync::Arc::new(mock_fetcher(
                "image/png",
                PNG_BYTES.to_vec(),
            )));
        let out = call_read(&tool, url_args("https://example.com/a.png", None))
            .await
            .unwrap();
        assert!(out.contains("does not accept image input"), "out: {out}");
        assert!(!out.contains("base64"));
    }

    #[tokio::test]
    async fn url_svg_returns_decoded_text() {
        let tool = ReadTool::new().with_url_fetcher(std::sync::Arc::new(mock_fetcher(
            "image/svg+xml",
            b"<svg><circle/></svg>".to_vec(),
        )));
        let out = call_read(&tool, url_args("https://example.com/a.svg", None))
            .await
            .unwrap();
        assert!(out.contains("[Image: SVG"), "out: {out}");
        assert!(out.contains("<circle/>"), "out: {out}");
    }

    #[tokio::test]
    async fn url_json_returns_capped_text() {
        let big = "x".repeat(200 * 1024);
        let tool = ReadTool::new().with_url_fetcher(std::sync::Arc::new(mock_fetcher(
            "application/json",
            big.into_bytes(),
        )));
        let out = call_read(&tool, url_args("https://example.com/a.json", None))
            .await
            .unwrap();
        assert!(
            out.contains("[URL: https://example.com/final]"),
            "out: {out}"
        );
        assert!(out.contains("truncated at 102400 bytes"), "out: {out}");
    }

    #[tokio::test]
    async fn url_binary_mime_steers_to_web_fetch() {
        let tool = ReadTool::new().with_url_fetcher(std::sync::Arc::new(mock_fetcher(
            "application/pdf",
            vec![1, 2, 3],
        )));
        let out = call_read(&tool, url_args("https://example.com/a.pdf", None))
            .await
            .unwrap();
        assert!(out.contains("web_fetch"), "out: {out}");
    }
}
