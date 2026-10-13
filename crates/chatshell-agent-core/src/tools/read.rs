//! Read tool for LLM agents
//!
//! A general-purpose file reading tool. Supports text files with line numbers,
//! image files (PNG, JPEG, GIF, WebP) with metadata and base64 encoding,
//! and PDF files with text extraction.

use std::path::PathBuf;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rig::agent::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::image_question::{ImageInput, ImageQuestionResponder};
use crate::path_policy;
use crate::tool_fs::{Fs, StdFs};

const DEFAULT_LINE_LIMIT: usize = 2000;
const MAX_LINE_LENGTH: usize = 2000;
const BINARY_CHECK_SIZE: usize = 8192;
const MAX_IMAGE_SIZE: u64 = 20 * 1024 * 1024; // 20 MB

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
    /// Optional natural-language question about an image file. When set,
    /// the image is analyzed by a vision-capable model and the answer is
    /// returned as text instead of the image data. Only supported for
    /// image files; `offset`/`limit` are ignored.
    #[serde(default)]
    pub question: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("Read error: {0}")]
pub struct ReadError(String);

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

    /// Answer a question about an image via the injected vision responder.
    async fn answer_image_question(
        &self,
        path: &std::path::Path,
        ext: &str,
        question: &str,
    ) -> Result<String, ReadError> {
        let image = load_image(self.fs.as_ref(), path, ext)?;

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
            path.display(),
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
        "Read the contents of a file from the filesystem. \
                For text files, returns content with line numbers. \
                For image files (PNG, JPEG, GIF, WebP, BMP), returns metadata and base64-encoded data. \
                For PDF files, extracts and returns the text content. \
                Supports reading specific line ranges for large text files. \
                Use negative offset to read from the end of a file (e.g. -20 reads the last 20 lines). \
                For image files, an optional \"question\" argument returns a text answer from a \
                vision-capable model instead of the image data — use it when the image's contents \
                should be analyzed or summarized rather than shown, or when reading images for a \
                model that cannot accept image input."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from. Positive values are 1-indexed from the start. Negative values count from the end (e.g. -10 reads the last 10 lines). Defaults to 1. Only applies to text files."
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read. Defaults to 2000. Only applies to text files."
                },
                "question": {
                    "type": "string",
                    "description": "Optional natural-language question about an image file. When set, the image is analyzed by a vision model and the answer is returned as text instead of the image data. Only supported for image files; offset/limit are ignored."
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
            "🔧 [tool-call] read: path=\"{}\" offset={:?} limit={:?} question={:?}",
            args.path,
            args.offset,
            args.limit,
            args.question
        );

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
            return read_pdf(self.fs.as_ref(), path);
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

/// Load an image (size-capped) and render its metadata header.
fn load_image(fs: &dyn Fs, path: &std::path::Path, ext: &str) -> Result<LoadedImage, ReadError> {
    let metadata = fs
        .metadata(path)
        .map_err(|e| ReadError(format!("Failed to read image metadata: {}", e)))?;

    let file_size = metadata.len;
    if file_size > MAX_IMAGE_SIZE {
        return Err(ReadError(format!(
            "Image too large ({}, max {})",
            format_file_size(file_size),
            format_file_size(MAX_IMAGE_SIZE)
        )));
    }

    let raw_bytes = fs
        .read(path)
        .map_err(|e| ReadError(format!("Failed to read image: {}", e)))?;

    let dims = parse_image_dimensions(&raw_bytes, ext);
    let dim_str = dims
        .map(|(w, h)| format!("{}x{}", w, h))
        .unwrap_or_else(|| "unknown".to_string());

    Ok(LoadedImage {
        bytes: raw_bytes,
        mime: mime_from_extension(ext),
        header: format!(
            "[Image: {}, {}, {}]",
            ext.to_uppercase(),
            dim_str,
            format_file_size(file_size)
        ),
    })
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

    let b64 = BASE64.encode(&image.bytes);

    tracing::info!(
        "🔧 [tool-result] read: image \"{}\" ({})",
        path.display(),
        image.header
    );

    Ok(format!(
        "{}\ndata:{};base64,{}",
        image.header, image.mime, b64
    ))
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

    let guidance = if question_available {
        "To ask about this image, read it again with the \"question\" argument \
         (e.g. {\"path\": \"<this file>\", \"question\": \"what does this \
         screenshot show?\"}) — a vision model answers and the answer is \
         returned as text."
    } else {
        "No vision model is configured, so this image cannot be analyzed. \
         Configure one under Settings → Auxiliary Models to enable image \
         questions."
    };

    Ok(format!(
        "{}\nThe current conversation model does not accept image input, so \
         the image content is omitted.\n{}",
        image.header, guidance
    ))
}

/// Read a PDF file by extracting its text content.
fn read_pdf(fs: &dyn Fs, path: &std::path::Path) -> Result<String, ReadError> {
    let bytes = fs
        .read(path)
        .map_err(|e| ReadError(format!("Failed to read PDF: {}", e)))?;
    let text = pdf_extract::extract_text_from_mem(&bytes)
        .map_err(|e| ReadError(format!("Failed to extract PDF text: {}", e)))?;

    if text.trim().is_empty() {
        return Ok("[PDF] No extractable text found (the PDF may contain only images).".into());
    }

    let line_count = text.lines().count();

    tracing::info!(
        "🔧 [tool-result] read: PDF \"{}\" ({} lines extracted)",
        path.display(),
        line_count
    );

    Ok(format!("[PDF: {} lines extracted]\n\n{}", line_count, text))
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
            question: question.map(str::to_string),
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
}
