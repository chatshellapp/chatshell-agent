//! Read-only extraction for OOXML office documents.
//!
//! `docx`/`pptx` are ZIP + XML; `xlsx` is parsed by calamine. Everything here
//! is read-only and bounded: entry counts, uncompressed sizes, and output
//! characters are capped so a hostile or huge file cannot exhaust memory or
//! the model's context window.
//!
//! The Markdown/TSV output is deliberately lossy — it is a reading view, not
//! an editing model. In particular `docx` text is flattened across runs and
//! tables are emitted as GFM tables; nothing here round-trips back to OOXML.

use std::io::{Cursor, Read};

use calamine::{Data, ExcelDateTime, Reader, Xlsx};
use quick_xml::Reader as XmlReader;
use quick_xml::escape::{resolve_predefined_entity, unescape};
use quick_xml::events::{BytesStart, Event};
use zip::ZipArchive;

/// OOXML formats this module can read.
pub const OFFICE_EXTENSIONS: &[&str] = &["docx", "docm", "xlsx", "xlsm", "pptx", "pptm"];
/// Legacy binary formats: not OOXML, and no reliable pure-Rust reader.
pub const LEGACY_EXTENSIONS: &[&str] = &["doc", "xls", "ppt"];

/// ZIP central-directory entry cap. OOXML documents carry a few dozen parts;
/// this is generous while still rejecting archive bombs.
const MAX_ZIP_ENTRIES: usize = 4096;
/// Declared uncompressed size cap for the whole package.
const MAX_TOTAL_UNCOMPRESSED: u64 = 256 * 1024 * 1024;
/// Declared uncompressed size cap for one part.
const MAX_ENTRY_UNCOMPRESSED: u64 = 64 * 1024 * 1024;
/// Cap on extracted document text handed to the model.
const MAX_OFFICE_CHARS: usize = 400_000;
/// Spreadsheet window caps per read.
const MAX_SHEET_ROWS: usize = 200;
const MAX_SHEET_COLS: usize = 100;
/// Preview size when no sheet was named.
const PREVIEW_ROWS: usize = 20;
const PREVIEW_COLS: usize = 26;
/// Per-cell cap; a single cell must not dominate the output.
const MAX_CELL_CHARS: usize = 500;

const OLE_MAGIC: [u8; 8] = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1];

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct OfficeError(String);

fn err(message: impl Into<String>) -> OfficeError {
    OfficeError(message.into())
}

pub fn is_office_extension(ext: &str) -> bool {
    OFFICE_EXTENSIONS.contains(&ext)
}

pub fn is_legacy_office_extension(ext: &str) -> bool {
    LEGACY_EXTENSIONS.contains(&ext)
}

pub fn legacy_message(ext: &str) -> String {
    format!(
        "Legacy binary Office format \".{ext}\" is not supported. \
         Save it as .docx/.xlsx/.pptx (or convert with LibreOffice) and read again."
    )
}

/// Validate the container before any format-specific parsing: OOXML must be a
/// ZIP, and encrypted OOXML (which is an OLE compound file despite the
/// extension) gets a precise error instead of a confusing parse failure.
pub fn validate_container(bytes: &[u8]) -> Result<(), OfficeError> {
    if bytes.starts_with(&OLE_MAGIC) {
        return Err(err(
            "File is an OLE compound document: it is either password-protected \
             (encrypted OOXML) or a legacy binary Office format. Decrypt it or \
             save as .docx/.xlsx/.pptx first.",
        ));
    }
    if !is_zip(bytes) {
        return Err(err(
            "File does not have a ZIP container; it is not a valid OOXML document.",
        ));
    }

    let mut archive = ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| err(format!("Invalid ZIP container: {e}")))?;
    if archive.len() > MAX_ZIP_ENTRIES {
        return Err(err(format!(
            "ZIP container has {} entries (limit {MAX_ZIP_ENTRIES}).",
            archive.len()
        )));
    }

    let mut total = 0u64;
    for index in 0..archive.len() {
        let file = archive
            .by_index(index)
            .map_err(|e| err(format!("Invalid ZIP entry {index}: {e}")))?;
        if file.size() > MAX_ENTRY_UNCOMPRESSED {
            let name = file
                .name()
                .map(|name| name.into_owned())
                .unwrap_or_else(|_| "<invalid>".to_string());
            return Err(err(format!(
                "ZIP entry \"{name}\" declares {} bytes (limit {}).",
                file.size(),
                MAX_ENTRY_UNCOMPRESSED
            )));
        }
        total = total.saturating_add(file.size());
    }
    if total > MAX_TOTAL_UNCOMPRESSED {
        return Err(err(format!(
            "ZIP package declares {total} uncompressed bytes (limit {MAX_TOTAL_UNCOMPRESSED})."
        )));
    }
    Ok(())
}

fn is_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04")
        || bytes.starts_with(b"PK\x05\x06")
        || bytes.starts_with(b"PK\x07\x08")
}

fn open_checked(bytes: &[u8]) -> Result<ZipArchive<Cursor<&[u8]>>, OfficeError> {
    validate_container(bytes)?;
    ZipArchive::new(Cursor::new(bytes)).map_err(|e| err(format!("Invalid ZIP container: {e}")))
}

fn read_entry(archive: &mut ZipArchive<Cursor<&[u8]>>, name: &str) -> Result<String, OfficeError> {
    let mut file = archive
        .by_name(name)
        .map_err(|_| err(format!("Missing OOXML part \"{name}\".")))?;
    if file.size() > MAX_ENTRY_UNCOMPRESSED {
        return Err(err(format!(
            "Part \"{name}\" declares {} bytes (limit {MAX_ENTRY_UNCOMPRESSED}).",
            file.size()
        )));
    }
    let mut buffer = Vec::new();
    file.by_ref()
        .take(MAX_ENTRY_UNCOMPRESSED + 1)
        .read_to_end(&mut buffer)
        .map_err(|e| err(format!("Failed to read \"{name}\": {e}")))?;
    if buffer.len() as u64 > MAX_ENTRY_UNCOMPRESSED {
        return Err(err(format!(
            "Part \"{name}\" exceeds the uncompressed size limit."
        )));
    }
    String::from_utf8(buffer).map_err(|_| err(format!("Part \"{name}\" is not valid UTF-8.")))
}

fn contains_dtd(xml: &str) -> bool {
    xml.contains("<!DOCTYPE") || xml.contains("<!doctype")
}

fn attr_value(element: &BytesStart<'_>, name: &str) -> Option<String> {
    for attribute in element.attributes().flatten() {
        if attribute.key.local_name().as_ref() == name {
            // Style names, list levels, and similar OOXML attributes are
            // plain identifiers; raw values are sufficient and avoid the
            // deprecated unescape helper.
            return Some(attribute.value.to_string());
        }
    }
    None
}

/// Resolve a general entity reference (`&amp;`, `&#x41;`). Unknown entities
/// are preserved literally: dropping them would silently corrupt text.
fn push_reference(out: &mut String, reference: &quick_xml::events::BytesRef<'_>) {
    if let Ok(Some(ch)) = reference.resolve_char_ref() {
        out.push(ch);
        return;
    }
    let name = reference.as_ref();
    match resolve_predefined_entity(name) {
        Some(value) => out.push_str(value),
        None => {
            out.push('&');
            out.push_str(name);
            out.push(';');
        }
    }
}

// ---------------------------------------------------------------------------
// DOCX
// ---------------------------------------------------------------------------

/// Extract `word/document.xml` as Markdown: headings from paragraph styles,
/// list items from numbering, table cells as GFM rows.
pub fn read_docx(bytes: &[u8]) -> Result<String, OfficeError> {
    let mut archive = open_checked(bytes)?;
    let xml = read_entry(&mut archive, "word/document.xml")?;
    if contains_dtd(&xml) {
        return Err(err("DOCTYPE declarations are not allowed in OOXML parts."));
    }

    let mut reader = XmlReader::from_str(&xml);
    reader.config_mut().trim_text(false);

    let mut out = String::new();
    let mut text = String::new();
    let mut in_text = 0usize;
    let mut paragraph_style: Option<String> = None;
    let mut paragraph_is_list = false;
    let mut paragraph_level = 0usize;

    let mut table_depth = 0usize;
    let mut cell_depth = 0usize;
    let mut cell = String::new();
    let mut row: Vec<String> = Vec::new();
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut truncated = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(element)) => match element.local_name().as_ref() {
                "p" => {
                    text.clear();
                    paragraph_style = None;
                    paragraph_is_list = false;
                    paragraph_level = 0;
                }
                "tbl" => {
                    table_depth += 1;
                    if table_depth == 1 {
                        table_rows.clear();
                    }
                }
                "tr" if table_depth == 1 => row.clear(),
                "tc" if table_depth == 1 => {
                    cell_depth += 1;
                    cell.clear();
                }
                "t" => in_text += 1,
                "pStyle" => paragraph_style = attr_value(&element, "val"),
                "numPr" => paragraph_is_list = true,
                "ilvl" => {
                    paragraph_level = attr_value(&element, "val")
                        .and_then(|value| value.parse().ok())
                        .unwrap_or(0)
                }
                _ => {}
            },
            Ok(Event::Empty(element)) => match element.local_name().as_ref() {
                "tab" => text.push('\t'),
                "br" | "cr" => text.push('\n'),
                "pStyle" => paragraph_style = attr_value(&element, "val"),
                "numPr" => paragraph_is_list = true,
                "ilvl" => {
                    paragraph_level = attr_value(&element, "val")
                        .and_then(|value| value.parse().ok())
                        .unwrap_or(0)
                }
                _ => {}
            },
            Ok(Event::Text(chunk)) => {
                if in_text > 0 {
                    match unescape(chunk.as_ref()) {
                        Ok(value) => text.push_str(&value),
                        Err(e) => return Err(err(format!("Invalid XML text escape: {e}"))),
                    }
                }
            }
            Ok(Event::GeneralRef(reference)) => {
                if in_text > 0 {
                    push_reference(&mut text, &reference);
                }
            }
            Ok(Event::CData(chunk)) => {
                if in_text > 0 {
                    text.push_str(chunk.as_ref());
                }
            }
            Ok(Event::End(element)) => match element.local_name().as_ref() {
                "t" => in_text = in_text.saturating_sub(1),
                "p" => {
                    let paragraph = text.trim();
                    if cell_depth > 0 {
                        if !paragraph.is_empty() {
                            if !cell.is_empty() {
                                cell.push(' ');
                            }
                            cell.push_str(paragraph);
                        }
                    } else if !paragraph.is_empty() {
                        emit_paragraph(
                            &mut out,
                            paragraph,
                            paragraph_style.as_deref(),
                            paragraph_is_list,
                            paragraph_level,
                        );
                    }
                }
                "tc" if table_depth == 1 && cell_depth > 0 => {
                    cell_depth -= 1;
                    row.push(cell.trim().to_string());
                }
                "tr" if table_depth == 1 => table_rows.push(std::mem::take(&mut row)),
                "tbl" => {
                    table_depth = table_depth.saturating_sub(1);
                    if table_depth == 0 {
                        emit_table(&mut out, &mut table_rows);
                    }
                }
                _ => {}
            },
            Ok(Event::DocType(_)) => {
                return Err(err("DOCTYPE declarations are not allowed in OOXML parts."));
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(err(format!("Failed to parse word/document.xml: {e}"))),
        }

        // Cheap byte-based growth guard; the exact character cap is applied
        // once after the parse (counting chars per event would be quadratic).
        if out.len() > MAX_OFFICE_CHARS * 4 {
            truncated = true;
            break;
        }
    }

    let (mut out, cut) = truncate_chars(out, MAX_OFFICE_CHARS);
    if truncated || cut {
        out.push_str("\n\n[truncated: document text exceeds the read limit]");
    }
    if out.trim().is_empty() {
        Ok("[DOCX] No extractable text found (the document may contain only images).".to_string())
    } else {
        Ok(out)
    }
}

/// Truncate on a character boundary. Returns the text and whether it was cut.
fn truncate_chars(mut text: String, limit: usize) -> (String, bool) {
    if text.chars().count() <= limit {
        return (text, false);
    }
    let cut = text
        .char_indices()
        .nth(limit)
        .map(|(index, _)| index)
        .unwrap_or(text.len());
    text.truncate(cut);
    (text, true)
}

fn emit_paragraph(out: &mut String, text: &str, style: Option<&str>, is_list: bool, level: usize) {
    if let Some(level) = style.and_then(heading_level) {
        out.push_str(&"#".repeat(level));
        out.push(' ');
        out.push_str(text);
        out.push_str("\n\n");
        return;
    }
    if is_list {
        out.push_str(&"  ".repeat(level.min(6)));
        out.push_str("- ");
        out.push_str(text);
        out.push('\n');
        return;
    }
    out.push_str(text);
    out.push_str("\n\n");
}

fn heading_level(style: &str) -> Option<usize> {
    let normalized = style.replace([' ', '-'], "").to_ascii_lowercase();
    if normalized == "title" {
        return Some(1);
    }
    if normalized == "subtitle" {
        return Some(2);
    }
    normalized
        .strip_prefix("heading")
        .and_then(|rest| rest.parse::<usize>().ok())
        .map(|level| level.clamp(1, 6))
}

fn emit_table(out: &mut String, rows: &mut Vec<Vec<String>>) {
    let rows = std::mem::take(rows);
    if rows.is_empty() {
        return;
    }
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    for (index, row) in rows.iter().enumerate() {
        out.push('|');
        for column in 0..columns {
            out.push(' ');
            out.push_str(&escape_table_cell(
                row.get(column).map(String::as_str).unwrap_or(""),
            ));
            out.push_str(" |");
        }
        out.push('\n');
        if index == 0 {
            out.push('|');
            for _ in 0..columns {
                out.push_str(" --- |");
            }
            out.push('\n');
        }
    }
    out.push('\n');
}

fn escape_table_cell(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

// ---------------------------------------------------------------------------
// PPTX
// ---------------------------------------------------------------------------

/// Extract slide text in slide-number order. Notes are appended when present.
pub fn read_pptx(bytes: &[u8]) -> Result<String, OfficeError> {
    let mut archive = open_checked(bytes)?;

    let mut slides: Vec<(u32, String)> = Vec::new();
    let mut notes: Vec<(u32, String)> = Vec::new();

    for index in 0..archive.len() {
        let name = {
            let file = archive
                .by_index(index)
                .map_err(|e| err(format!("Invalid ZIP entry {index}: {e}")))?;
            file.name()
                .map_err(|e| err(format!("Invalid ZIP entry name: {e}")))?
                .to_string()
        };
        if let Some(number) = numbered_part(&name, "ppt/slides/slide") {
            slides.push((number, name));
        } else if let Some(number) = numbered_part(&name, "ppt/notesSlides/notesSlide") {
            notes.push((number, name));
        }
    }

    if slides.is_empty() {
        return Err(err("No slides found in ppt/slides."));
    }
    slides.sort_by_key(|(number, _)| *number);
    notes.sort_by_key(|(number, _)| *number);

    let mut out = format!("[PPTX: {} slides]\n\n", slides.len());
    for (position, (number, name)) in slides.iter().enumerate() {
        let xml = read_entry(&mut archive, name)?;
        if contains_dtd(&xml) {
            return Err(err("DOCTYPE declarations are not allowed in OOXML parts."));
        }
        let text = collect_pptx_text(&xml)?;
        out.push_str(&format!("--- Slide {number} ---\n"));
        if text.trim().is_empty() {
            out.push_str("[No extractable text]\n");
        } else {
            out.push_str(text.trim());
            out.push('\n');
        }

        if let Some((_, notes_name)) = notes.iter().find(|(note, _)| note == number) {
            let notes_xml = read_entry(&mut archive, notes_name)?;
            let notes_text = collect_pptx_text(&notes_xml)?;
            if !notes_text.trim().is_empty() {
                out.push_str("\n[Notes]\n");
                out.push_str(notes_text.trim());
                out.push('\n');
            }
        }
        out.push('\n');

        if position + 1 < slides.len() && out.len() > MAX_OFFICE_CHARS * 4 {
            break;
        }
    }

    let (mut out, cut) = truncate_chars(out, MAX_OFFICE_CHARS);
    if cut {
        out.push_str("\n[truncated: presentation text exceeds the read limit]\n");
    }
    Ok(out)
}

fn numbered_part(name: &str, prefix: &str) -> Option<u32> {
    name.strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(".xml"))
        .and_then(|number| number.parse().ok())
}

fn collect_pptx_text(xml: &str) -> Result<String, OfficeError> {
    let mut reader = XmlReader::from_str(xml);
    reader.config_mut().trim_text(false);

    let mut out = String::new();
    let mut in_text = 0usize;
    loop {
        match reader.read_event() {
            Ok(Event::Start(element)) => {
                if element.local_name().as_ref() == "t" {
                    in_text += 1;
                }
            }
            Ok(Event::Empty(element)) => {
                if element.local_name().as_ref() == "br" {
                    out.push('\n');
                }
            }
            Ok(Event::Text(chunk)) => {
                if in_text > 0 {
                    match unescape(chunk.as_ref()) {
                        Ok(value) => out.push_str(&value),
                        Err(e) => return Err(err(format!("Invalid XML text escape: {e}"))),
                    }
                }
            }
            Ok(Event::GeneralRef(reference)) => {
                if in_text > 0 {
                    push_reference(&mut out, &reference);
                }
            }
            Ok(Event::End(element)) => match element.local_name().as_ref() {
                "t" => in_text = in_text.saturating_sub(1),
                "p" => out.push('\n'),
                _ => {}
            },
            Ok(Event::DocType(_)) => {
                return Err(err("DOCTYPE declarations are not allowed in OOXML parts."));
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(err(format!("Failed to parse slide XML: {e}"))),
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// XLSX
// ---------------------------------------------------------------------------

/// A bounded TSV view of a workbook.
///
/// Without `sheet`, only the first sheet is previewed (`PREVIEW_ROWS`), and
/// the sheet list is printed. With `sheet`, up to `MAX_SHEET_ROWS` rows are
/// shown; `range` (A1 style) narrows the window. Cell addresses are absolute,
/// so a later edit tool can address the same cells.
pub fn read_xlsx(
    bytes: &[u8],
    sheet: Option<&str>,
    range: Option<&str>,
    formulas: bool,
) -> Result<String, OfficeError> {
    validate_container(bytes)?;

    let mut workbook = Xlsx::new(Cursor::new(bytes))
        .map_err(|e| err(format!("Failed to open spreadsheet: {e}")))?;

    let names = workbook.sheet_names().to_owned();
    if names.is_empty() {
        return Ok("[XLSX] No worksheets.".to_string());
    }

    let sheet_name = match sheet {
        Some(requested) => {
            if !names.iter().any(|name| name == requested) {
                return Err(err(format!(
                    "Sheet \"{requested}\" not found. Sheets: {}",
                    names.join(", ")
                )));
            }
            requested.to_string()
        }
        None => names[0].clone(),
    };

    let data = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| err(format!("Failed to read sheet \"{sheet_name}\": {e}")))?;
    let formula_range = if formulas {
        workbook.worksheet_formula(&sheet_name).ok()
    } else {
        None
    };

    // Requested absolute window (0-based, inclusive).
    let preview = sheet.is_none() && range.is_none();
    let (start_row, start_col, mut end_row, mut end_col) = match range {
        Some(spec) => {
            let parsed = parse_a1_range(spec)?;
            (
                parsed.start_row,
                parsed.start_col,
                parsed.end_row,
                parsed.end_col,
            )
        }
        None => {
            let rows = if preview {
                PREVIEW_ROWS
            } else {
                MAX_SHEET_ROWS
            };
            let cols = if preview { PREVIEW_COLS } else { 26 };
            (0, 0, (rows - 1) as u32, (cols - 1) as u32)
        }
    };

    let row_count = end_row.saturating_sub(start_row) + 1;
    let col_count = end_col.saturating_sub(start_col) + 1;
    if row_count as usize > MAX_SHEET_ROWS || col_count as usize > MAX_SHEET_COLS {
        return Err(err(format!(
            "Requested range is {row_count} rows × {col_count} columns; \
             the limit is {MAX_SHEET_ROWS} × {MAX_SHEET_COLS}. Narrow the range."
        )));
    }

    // Trim the preview to the used area so empty sheets stay quiet.
    if preview || range.is_none() {
        if let Some((last_row, last_col)) = last_used(&data) {
            end_row = end_row.min(last_row);
            end_col = end_col.min(last_col);
        } else {
            return Ok(format!(
                "[XLSX: {} sheets: {}]\n\nSheet \"{}\": empty.",
                names.len(),
                names.join(", "),
                sheet_name
            ));
        }
    }
    if end_row < start_row || end_col < start_col {
        return Ok(format!(
            "[XLSX: {} sheets: {}]\n\nSheet \"{}\": requested range is empty.",
            names.len(),
            names.join(", "),
            sheet_name
        ));
    }

    let used = match (data.start(), data.end()) {
        (Some(start), Some(end)) => format!("{}:{}", cell_ref(start), cell_ref(end)),
        _ => "empty".to_string(),
    };

    let mut out = format!(
        "[XLSX: {} sheets: {}]\n\n--- Sheet \"{}\": {}:{}, used {} ---\n",
        names.len(),
        names.join(", "),
        sheet_name,
        cell_ref((start_row, start_col)),
        cell_ref((end_row, end_col)),
        used
    );
    if preview {
        out.push_str("(preview only; pass \"sheet\" to choose a sheet and \"range\" to page)\n");
    }
    if formulas {
        out.push_str("(formulas shown where present)\n");
    }
    out.push('\n');

    out.push_str("row");
    for column in start_col..=end_col {
        out.push('\t');
        out.push_str(&col_letters(column));
    }
    out.push('\n');

    for row in start_row..=end_row {
        out.push_str(&(row + 1).to_string());
        for column in start_col..=end_col {
            out.push('\t');
            let value = data.get_value((row, column));
            let formula = formula_range
                .as_ref()
                .and_then(|range| range.get_value((row, column)));
            out.push_str(&render_cell(value, formula, formulas));
        }
        out.push('\n');
    }

    Ok(out)
}

fn last_used(data: &calamine::Range<Data>) -> Option<(u32, u32)> {
    let (_, end) = (data.start()?, data.end()?);
    Some(end)
}

fn render_cell(value: Option<&Data>, formula: Option<&String>, use_formulas: bool) -> String {
    if use_formulas && let Some(formula) = formula.filter(|formula| !formula.is_empty()) {
        return sanitize_cell(&format!("={formula}"));
    }
    let text = match value {
        None | Some(Data::Empty) => String::new(),
        Some(Data::String(value)) => value.clone(),
        Some(Data::Int(value)) => value.to_string(),
        Some(Data::Float(value)) => value.to_string(),
        Some(Data::Bool(value)) => value.to_string(),
        Some(Data::DateTime(value)) => format_excel_datetime(value),
        Some(Data::DateTimeIso(value)) | Some(Data::DurationIso(value)) => value.clone(),
        Some(Data::Error(value)) => format!("{value}"),
    };
    sanitize_cell(&text)
}

fn sanitize_cell(text: &str) -> String {
    let mut cleaned: String = text
        .chars()
        .map(|ch| {
            if ch == '\t' || ch == '\r' || ch == '\n' {
                ' '
            } else {
                ch
            }
        })
        .collect();
    if cleaned.chars().count() > MAX_CELL_CHARS {
        cleaned = cleaned.chars().take(MAX_CELL_CHARS).collect();
        cleaned.push('…');
    }
    cleaned
}

fn format_excel_datetime(value: &ExcelDateTime) -> String {
    if value.is_duration() {
        return value.as_f64().to_string();
    }
    let (year, month, day, hour, minute, second, milli) = value.to_ymd_hms_milli();
    if hour == 0 && minute == 0 && second == 0 && milli == 0 {
        format!("{year:04}-{month:02}-{day:02}")
    } else if milli > 0 {
        format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}.{milli:03}")
    } else {
        format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}")
    }
}

fn cell_ref((row, column): (u32, u32)) -> String {
    format!("{}{}", col_letters(column), row + 1)
}

fn col_letters(mut column: u32) -> String {
    let mut letters = String::new();
    loop {
        letters.insert(0, (b'A' + (column % 26) as u8) as char);
        if column < 26 {
            break;
        }
        column = column / 26 - 1;
    }
    letters
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct A1Range {
    start_row: u32,
    start_col: u32,
    end_row: u32,
    end_col: u32,
}

fn parse_a1_range(spec: &str) -> Result<A1Range, OfficeError> {
    let spec = spec.trim().to_ascii_uppercase();
    let (start, end) = spec.split_once(':').unwrap_or((&spec, &spec));
    let (start_row, start_col) = parse_cell_ref(start)?;
    let (end_row, end_col) = parse_cell_ref(end)?;
    if end_row < start_row || end_col < start_col {
        return Err(err(format!(
            "Range \"{spec}\" ends before it starts; use the form A1:F40."
        )));
    }
    Ok(A1Range {
        start_row,
        start_col,
        end_row,
        end_col,
    })
}

fn parse_cell_ref(spec: &str) -> Result<(u32, u32), OfficeError> {
    let letters: String = spec
        .chars()
        .take_while(|ch| ch.is_ascii_alphabetic())
        .collect();
    let digits: String = spec
        .chars()
        .skip_while(|ch| ch.is_ascii_alphabetic())
        .collect();
    if letters.is_empty() || digits.is_empty() {
        return Err(err(format!(
            "Invalid cell reference \"{spec}\"; expected A1 style (e.g. A1:F40)."
        )));
    }
    let mut column = 0u32;
    for letter in letters.bytes() {
        column = column
            .checked_mul(26)
            .and_then(|value| value.checked_add((letter - b'A' + 1) as u32))
            .ok_or_else(|| err(format!("Column in \"{spec}\" is out of range.")))?;
    }
    let column = column - 1;
    let row: u32 = digits
        .parse()
        .map_err(|_| err(format!("Invalid row number in \"{spec}\".")))?;
    if row == 0 {
        return Err(err(format!("\"{spec}\" uses a 0 row; rows start at 1.")));
    }
    if column > 16_383 || row > 1_048_576 {
        return Err(err(format!(
            "\"{spec}\" is outside the Excel sheet bounds."
        )));
    }
    Ok((row - 1, column))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use zip::CompressionMethod;
    use zip::write::SimpleFileOptions;

    fn zip_bytes(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options =
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            for (name, content) in parts {
                writer.start_file(*name, options).unwrap();
                writer.write_all(content.as_bytes()).unwrap();
            }
            writer.finish().unwrap();
        }
        cursor.into_inner()
    }

    fn docx_bytes(document: &str) -> Vec<u8> {
        zip_bytes(&[("word/document.xml", document)])
    }

    fn xlsx_bytes(sheet_xml: &str) -> Vec<u8> {
        let content_types = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"#;
        let root_rels = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#;
        let workbook = r#"<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>"#;
        let workbook_rels = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#;
        zip_bytes(&[
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", root_rels),
            ("xl/workbook.xml", workbook),
            ("xl/_rels/workbook.xml.rels", workbook_rels),
            ("xl/worksheets/sheet1.xml", sheet_xml),
        ])
    }

    #[test]
    fn docx_extracts_headings_runs_lists_and_tables() {
        let document = r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Report</w:t></w:r></w:p>
<w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>world &amp; friends</w:t></w:r></w:p>
<w:p><w:pPr><w:numPr><w:ilvl w:val="1"/></w:numPr></w:pPr><w:r><w:t>Item</w:t></w:r></w:p>
<w:tbl>
<w:tr><w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Qty</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>Apple</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>3</w:t></w:r></w:p></w:tc></w:tr>
</w:tbl>
</w:body>
</w:document>"#;
        let out = read_docx(&docx_bytes(document)).unwrap();

        assert!(out.contains("# Report"), "{out}");
        assert!(out.contains("Hello world & friends"), "{out}");
        assert!(out.contains("  - Item"), "{out}");
        assert!(out.contains("| Name | Qty |"), "{out}");
        assert!(out.contains("| Apple | 3 |"), "{out}");
    }

    #[test]
    fn docx_rejects_dtd() {
        let document = r#"<!DOCTYPE foo [<!ENTITY bar "baz">]><w:document xmlns:w="x"><w:body><w:p><w:r><w:t>&bar;</w:t></w:r></w:p></w:body></w:document>"#;
        let error = read_docx(&docx_bytes(document)).unwrap_err();
        assert!(error.to_string().contains("DOCTYPE"), "{error}");
    }

    #[test]
    fn pptx_extracts_slide_text_in_numeric_order() {
        let parts = [
            (
                "ppt/slides/slide10.xml",
                r#"<p:sld xmlns:a="a"><a:p><a:r><a:t>Last</a:t></a:r></a:p></p:sld>"#,
            ),
            (
                "ppt/slides/slide2.xml",
                r#"<p:sld xmlns:a="a"><a:p><a:r><a:t>Second</a:t></a:r></a:p></p:sld>"#,
            ),
        ];
        let out = read_pptx(&zip_bytes(&parts)).unwrap();
        let second = out.find("Second").unwrap();
        let last = out.find("Last").unwrap();
        assert!(second < last, "{out}");
        assert!(out.contains("[PPTX: 2 slides]"), "{out}");
    }

    #[test]
    fn xlsx_previews_cells_and_formulas() {
        let sheet = r#"<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1">
<c r="A1" t="inlineStr"><is><t>Name</t></is></c>
<c r="B1"><v>42</v></c>
<c r="C1"><f>SUM(B1:B1)</f><v>42</v></c>
</row>
<row r="2">
<c r="A2" t="inlineStr"><is><t>Apple</t></is></c>
<c r="B2"><v>3.5</v></c>
</row>
</sheetData>
</worksheet>"#;
        let bytes = xlsx_bytes(sheet);

        let preview = read_xlsx(&bytes, None, None, false).unwrap();
        assert!(preview.contains("Sheet1"), "{preview}");
        assert!(preview.contains("Name\t42\t42"), "{preview}");
        assert!(preview.contains("Apple\t3.5"), "{preview}");

        let formulas = read_xlsx(&bytes, Some("Sheet1"), Some("A1:C2"), true).unwrap();
        assert!(formulas.contains("=SUM(B1:B1)"), "{formulas}");
    }

    #[test]
    fn xlsx_rejects_oversized_windows() {
        let bytes = xlsx_bytes(
            r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData/></worksheet>"#,
        );
        let error = read_xlsx(&bytes, Some("Sheet1"), Some("A1:ZZ999"), false).unwrap_err();
        assert!(error.to_string().contains("limit"), "{error}");
    }

    #[test]
    fn encrypted_and_legacy_containers_are_rejected() {
        let error = validate_container(&OLE_MAGIC).unwrap_err();
        assert!(error.to_string().contains("password-protected"), "{error}");

        let error = validate_container(b"not a zip").unwrap_err();
        assert!(error.to_string().contains("ZIP"), "{error}");
    }

    #[test]
    fn zip_entry_limit_is_enforced() {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut cursor);
            let options =
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            for index in 0..=MAX_ZIP_ENTRIES {
                writer.start_file(format!("part{index}"), options).unwrap();
            }
            writer.finish().unwrap();
        }
        let error = validate_container(&cursor.into_inner()).unwrap_err();
        assert!(error.to_string().contains("entries"), "{error}");
    }

    #[test]
    fn a1_references_round_trip() {
        assert_eq!(parse_cell_ref("A1").unwrap(), (0, 0));
        assert_eq!(parse_cell_ref("F40").unwrap(), (39, 5));
        assert_eq!(parse_cell_ref("AA10").unwrap(), (9, 26));
        assert_eq!(col_letters(0), "A");
        assert_eq!(col_letters(25), "Z");
        assert_eq!(col_letters(26), "AA");
        assert_eq!(cell_ref((39, 5)), "F40");

        let range = parse_a1_range("B2:D5").unwrap();
        assert_eq!(
            range,
            A1Range {
                start_row: 1,
                start_col: 1,
                end_row: 4,
                end_col: 3
            }
        );
        assert!(parse_a1_range("A0:B2").is_err());
        assert!(parse_a1_range("B2:A1").is_err());
    }

    #[test]
    fn excel_datetime_formats_date_only_and_timestamps() {
        use calamine::ExcelDateTimeType;

        let date = ExcelDateTime::new(45000.0, ExcelDateTimeType::DateTime, false);
        assert_eq!(format_excel_datetime(&date), "2023-03-15");

        let stamp = ExcelDateTime::new(45000.5, ExcelDateTimeType::DateTime, false);
        assert_eq!(format_excel_datetime(&stamp), "2023-03-15 12:00:00");
    }

    #[test]
    fn cell_text_is_sanitized_and_bounded() {
        let long = "x".repeat(MAX_CELL_CHARS + 10);
        let cleaned = sanitize_cell(&format!("a\tb\nc{long}"));
        assert!(!cleaned.contains('\t'));
        assert!(!cleaned.contains('\n'));
        assert!(cleaned.ends_with('…'));
    }
}
