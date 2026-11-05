//! Streaming capture of bash command output.
//!
//! The model sees a bounded inline view (head plus tail); the complete
//! output streams to a spill file as it arrives, so nothing past the inline
//! cap is lost. The file is capped too: past [`SPILL_HEAD_BYTES`] it stops
//! growing and, when the command finishes, gets an explicit omission marker
//! followed by the last [`SPILL_TAIL_BYTES`] of output. Every notice the
//! model reads states exactly what the file holds.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// Output above this many bytes is truncated inline and spilled.
pub(crate) const MAX_INLINE_BYTES: usize = 20_000;
/// Bytes kept from the start of truncated inline output.
const INLINE_HEAD_BYTES: usize = 5_000;
/// Bytes kept from the end of truncated inline output.
const INLINE_TAIL_BYTES: usize = 10_000;
/// Bytes of the stream start kept in memory (inline head, denial scans).
const MEMORY_HEAD_BYTES: usize = 64 * 1024;
/// Spill file bytes written verbatim from the stream start.
pub(crate) const SPILL_HEAD_BYTES: u64 = 12 * 1024 * 1024;
/// Bytes of the stream end kept in memory and appended after the
/// omission marker when the stream outgrew [`SPILL_HEAD_BYTES`].
pub(crate) const SPILL_TAIL_BYTES: usize = 4 * 1024 * 1024;

/// Where spilled output goes.
#[derive(Debug, Clone)]
pub struct SpillTarget {
    /// Directory for spill files (created on first spill).
    pub dir: PathBuf,
    /// When set, the model is told `scheme://<file>` (resolved by the read
    /// tool's host resolver) instead of the absolute path, and the file is
    /// left for the host to expire. When `None`, the absolute path is used
    /// and the file is registered for cleanup at the end of the generation.
    pub reference_scheme: Option<String>,
}

impl SpillTarget {
    /// The process temp directory, absolute paths, cleaned up per
    /// generation (hosts without a durable store).
    pub fn temp() -> Self {
        Self {
            dir: std::env::temp_dir(),
            reference_scheme: None,
        }
    }
}

/// A spill file that was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillFile {
    pub path: PathBuf,
    /// What the model is told to read (reference or absolute path).
    pub location: String,
    /// Bytes dropped between the file's head and tail parts (0 = the file
    /// holds the complete output).
    pub omitted_bytes: u64,
    /// Whether the host should delete the file at the end of generation.
    pub temporary: bool,
}

/// The finished capture of one command.
#[derive(Debug, Default)]
pub struct Captured {
    /// The complete output when it fit in memory, else the head and tail
    /// joined by a gap line. Used for denial and hint scans.
    pub scan_text: String,
    /// Bounded inline view for the model.
    pub inline: String,
    pub total_bytes: u64,
    pub truncated_inline: bool,
    pub spill: Option<SpillFile>,
    /// Set when spilling was needed but failed.
    pub spill_error: Option<String>,
}

/// Accumulates output lines; see the module docs.
pub(crate) struct OutputCapture {
    target: Option<SpillTarget>,
    head: String,
    /// The head stopped growing (the stream outgrew [`MEMORY_HEAD_BYTES`]).
    head_full: bool,
    tail: String,
    total_bytes: u64,
    total_lines: u64,
    file: Option<OpenSpill>,
    spill_error: Option<String>,
}

struct OpenSpill {
    path: PathBuf,
    name: String,
    writer: BufWriter<File>,
    written: u64,
    capped: bool,
}

impl OutputCapture {
    pub(crate) fn new(target: Option<SpillTarget>) -> Self {
        Self {
            target,
            head: String::new(),
            head_full: false,
            tail: String::new(),
            total_bytes: 0,
            total_lines: 0,
            file: None,
            spill_error: None,
        }
    }

    /// Append one line (including its newline, if any).
    pub(crate) fn push(&mut self, line: &str) {
        self.total_bytes += line.len() as u64;
        self.total_lines += 1;

        if !self.head_full {
            if self.head.len() + line.len() <= MEMORY_HEAD_BYTES {
                self.head.push_str(line);
            } else {
                self.head_full = true;
            }
        }
        self.tail.push_str(line);
        if self.tail.len() > 2 * SPILL_TAIL_BYTES {
            let cut = ceil_char_boundary(&self.tail, self.tail.len() - SPILL_TAIL_BYTES);
            self.tail.drain(..cut);
        }

        if self.file.is_none()
            && self.spill_error.is_none()
            && self.total_bytes > MAX_INLINE_BYTES as u64
        {
            // Everything so far is still in the head (the inline cap is far
            // below the memory head), so the file starts complete.
            self.open_spill();
        } else if let Some(file) = self.file.as_mut() {
            write_capped(file, line);
        }
    }

    fn open_spill(&mut self) {
        let Some(target) = self.target.clone() else {
            return;
        };
        let name = format!("bash-{}.txt", uuid::Uuid::now_v7());
        let path = target.dir.join(&name);
        let opened = std::fs::create_dir_all(&target.dir).and_then(|()| File::create(&path));
        match opened {
            Ok(file) => {
                let mut spill = OpenSpill {
                    path,
                    name,
                    writer: BufWriter::new(file),
                    written: 0,
                    capped: false,
                };
                let head = std::mem::take(&mut self.head);
                write_capped(&mut spill, &head);
                self.head = head;
                self.file = Some(spill);
            }
            Err(e) => {
                tracing::warn!("🖥️ [bash] Could not create spill file: {e}");
                self.spill_error = Some(e.to_string());
            }
        }
    }

    /// Finish the capture: flush and close the spill file, build the
    /// inline view.
    pub(crate) fn finish(mut self) -> Captured {
        let truncated_inline = self.total_bytes > MAX_INLINE_BYTES as u64;
        let complete_in_memory = !self.head_full;

        let scan_text = if complete_in_memory {
            self.head.clone()
        } else {
            format!(
                "{}\n...\n{}",
                self.head,
                &self.tail[ceil_char_boundary(
                    &self.tail,
                    self.tail.len().saturating_sub(MEMORY_HEAD_BYTES)
                )..]
            )
        };

        let inline = if !truncated_inline {
            self.head.trim_end().to_string()
        } else {
            let head_end = floor_char_boundary(&self.head, INLINE_HEAD_BYTES);
            let head = &self.head[..head_end];
            let tail_start = ceil_char_boundary(
                &self.tail,
                self.tail.len().saturating_sub(INLINE_TAIL_BYTES),
            );
            let tail = &self.tail[tail_start..];
            let shown_lines = (head.lines().count() + tail.lines().count()) as u64;
            let omitted_lines = self.total_lines.saturating_sub(shown_lines);
            format!(
                "{}\n\n... [{} lines omitted, {} total] ...\n\n{}",
                head.trim_end(),
                omitted_lines,
                format_bytes(self.total_bytes),
                tail.trim()
            )
        };

        let mut spill_error = self.spill_error.take();
        let spill = match self.file.take() {
            Some(mut file) => match finish_spill(&mut file, &self.tail, self.total_bytes) {
                Ok(omitted_bytes) => {
                    let target = self.target.as_ref();
                    let scheme = target.and_then(|t| t.reference_scheme.as_deref());
                    Some(SpillFile {
                        location: match scheme {
                            Some(scheme) => crate::tools::reference::reference(scheme, &file.name),
                            None => file.path.display().to_string(),
                        },
                        path: file.path,
                        omitted_bytes,
                        temporary: scheme.is_none(),
                    })
                }
                Err(e) => {
                    tracing::warn!("🖥️ [bash] Could not finish spill file: {e}");
                    let _ = std::fs::remove_file(&file.path);
                    spill_error = Some(e.to_string());
                    None
                }
            },
            None => None,
        };
        if truncated_inline && spill.is_none() && spill_error.is_none() {
            spill_error = Some("no spill location is configured".to_string());
        }

        Captured {
            scan_text,
            inline,
            total_bytes: self.total_bytes,
            truncated_inline,
            spill,
            spill_error,
        }
    }
}

fn write_capped(file: &mut OpenSpill, text: &str) {
    if file.capped {
        return;
    }
    let room = SPILL_HEAD_BYTES.saturating_sub(file.written) as usize;
    let chunk = if text.len() <= room {
        text
    } else {
        file.capped = true;
        &text[..floor_char_boundary(text, room)]
    };
    match file.writer.write_all(chunk.as_bytes()) {
        Ok(()) => file.written += chunk.len() as u64,
        Err(e) => {
            tracing::warn!("🖥️ [bash] Spill write failed: {e}");
            file.capped = true;
        }
    }
}

/// Append the omission marker and tail when the file was capped. Returns
/// the omitted byte count.
fn finish_spill(file: &mut OpenSpill, tail: &str, total: u64) -> std::io::Result<u64> {
    let mut omitted = 0;
    if file.capped {
        // The in-memory tail is trimmed lazily (up to twice its budget).
        let tail = &tail[ceil_char_boundary(tail, tail.len().saturating_sub(SPILL_TAIL_BYTES))..];
        let missing = total.saturating_sub(file.written);
        let tail = if (tail.len() as u64) >= missing {
            // The tail overlaps what was written: append only the rest.
            &tail[ceil_char_boundary(tail, tail.len() - missing as usize)..]
        } else {
            omitted = missing - tail.len() as u64;
            tail
        };
        if omitted > 0 {
            write!(
                file.writer,
                "\n\n[... {omitted} bytes omitted (spill file cap reached); the last {} follow ...]\n\n",
                format_bytes(tail.len() as u64)
            )?;
        }
        file.writer.write_all(tail.as_bytes())?;
    }
    file.writer.flush()?;
    Ok(omitted)
}

/// The notice appended to the inline view.
pub(crate) fn spill_notice(captured: &Captured) -> String {
    if !captured.truncated_inline {
        return String::new();
    }
    match (&captured.spill, &captured.spill_error) {
        (Some(spill), _) if spill.omitted_bytes == 0 => format!(
            "\n[full output ({}) saved to {} -- use the read tool with offset/limit to view it]",
            format_bytes(captured.total_bytes),
            spill.location
        ),
        (Some(spill), _) => format!(
            "\n[output ({}) saved to {}, except {} in the middle that exceeded the file cap \
             (marked in the file) -- use the read tool with offset/limit to view it]",
            format_bytes(captured.total_bytes),
            spill.location,
            format_bytes(spill.omitted_bytes)
        ),
        (None, Some(error)) => format!(
            "\n[the full output could not be saved ({error}); only the lines above are \
             available -- re-run with a filter such as grep, head, or tail]"
        ),
        (None, None) => String::new(),
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn floor_char_boundary(s: &str, pos: usize) -> usize {
    let mut i = pos.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, pos: usize) -> usize {
    let mut i = pos.min(s.len());
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture_lines(target: Option<SpillTarget>, lines: impl Iterator<Item = String>) -> Captured {
        let mut capture = OutputCapture::new(target);
        for line in lines {
            capture.push(&line);
        }
        capture.finish()
    }

    fn target(dir: &std::path::Path, scheme: Option<&str>) -> SpillTarget {
        SpillTarget {
            dir: dir.to_path_buf(),
            reference_scheme: scheme.map(str::to_string),
        }
    }

    #[test]
    fn short_output_is_inline_and_not_spilled() {
        let dir = tempfile::tempdir().unwrap();
        let captured = capture_lines(
            Some(target(dir.path(), None)),
            ["a\n".to_string(), "b\n".to_string()].into_iter(),
        );
        assert_eq!(captured.inline, "a\nb");
        assert!(captured.spill.is_none());
        assert!(spill_notice(&captured).is_empty());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn long_output_streams_completely_to_the_spill_file() {
        let dir = tempfile::tempdir().unwrap();
        // ~1.3 MB: far beyond the old 100 KB capture ceiling.
        let lines: Vec<String> = (0..100_000).map(|i| format!("line {i}\n")).collect();
        let expected: String = lines.concat();
        let captured = capture_lines(
            Some(target(dir.path(), Some("tool-output"))),
            lines.into_iter(),
        );

        assert!(captured.truncated_inline);
        assert!(captured.inline.contains("line 0"));
        assert!(captured.inline.contains("line 99999"));
        assert!(captured.inline.contains("lines omitted"));
        let spill = captured.spill.as_ref().unwrap();
        assert_eq!(spill.omitted_bytes, 0);
        assert!(!spill.temporary);
        assert!(
            spill.location.starts_with("tool-output://bash-"),
            "{}",
            spill.location
        );
        assert_eq!(std::fs::read_to_string(&spill.path).unwrap(), expected);
        let notice = spill_notice(&captured);
        assert!(notice.contains("full output"), "{notice}");
        assert!(notice.contains(&spill.location), "{notice}");
    }

    #[test]
    fn oversized_output_keeps_head_and_tail_with_an_explicit_marker() {
        let dir = tempfile::tempdir().unwrap();
        let line = format!("{}\n", "x".repeat(1023));
        let count = ((SPILL_HEAD_BYTES as usize + 2 * SPILL_TAIL_BYTES) / 1024) + 10;
        let lines = (0..count).map(|i| {
            if i == count - 1 {
                "LAST LINE\n".to_string()
            } else {
                line.clone()
            }
        });
        let captured = capture_lines(Some(target(dir.path(), None)), lines);

        let spill = captured.spill.as_ref().unwrap();
        assert!(
            spill.temporary,
            "absolute-path spills are cleaned per generation"
        );
        assert!(spill.omitted_bytes > 0);
        let content = std::fs::read_to_string(&spill.path).unwrap();
        assert!(content.contains("bytes omitted (spill file cap reached)"));
        assert!(content.ends_with("LAST LINE\n"));
        assert!(content.len() as u64 <= SPILL_HEAD_BYTES + SPILL_TAIL_BYTES as u64 + 200);
        let notice = spill_notice(&captured);
        assert!(notice.contains("exceeded the file cap"), "{notice}");
    }

    #[test]
    fn missing_spill_target_says_the_output_was_not_saved() {
        let lines = (0..5_000).map(|i| format!("line {i}\n"));
        let captured = capture_lines(None, lines);
        assert!(captured.truncated_inline);
        assert!(captured.spill.is_none());
        assert!(spill_notice(&captured).contains("could not be saved"));
    }

    #[test]
    fn multibyte_output_keeps_char_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        for piece in ["你好世界\n", "こんにちは世界\n", "안녕하세요 세계\n"] {
            let captured = capture_lines(
                Some(target(dir.path(), None)),
                std::iter::repeat_n(piece.to_string(), 10_000),
            );
            assert!(captured.truncated_inline);
            assert!(!captured.inline.contains('\u{FFFD}'));
            let spill = captured.spill.unwrap();
            assert_eq!(
                std::fs::read_to_string(&spill.path).unwrap(),
                piece.repeat(10_000)
            );
        }
    }
}
