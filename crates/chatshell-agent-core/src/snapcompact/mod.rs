//! Snapcompact compaction: archive conversation history as dense pixel-font
//! bitmap images the model reads back near-verbatim at a fraction of the
//! input-token price (see the snapcompact technique writeup). This port
//! keeps the eval-winning black-ink shapes and drops hue cycling, line
//! repetition, foveation, and Lanczos stretching.
//!
//! Font assets are bundled under their own upstream licenses: the X.org
//! misc 8x13 BDF bitmap font (Markus Kuhn's ucs-fonts) and a Silver
//! TrueType fallback.

pub mod fonts;
pub mod normalize;
pub mod render;

use std::sync::LazyLock;

use ab_glyph::{Font as _, FontArc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::snapcompact::fonts::{BDF_8X13, BitmapFont, SILVER_TTF, parse_bdf};
use crate::snapcompact::normalize::wrap;

/// Hard cap on archive frames per compaction (v1; upstream defaults to 80 but
/// budgets against billing and payload size — ChatShell frames ride the
/// snapshot sync for now, so stay conservative).
pub const MAX_FRAMES: usize = 8;

/// Minimum ratio of renderable characters before the 8x13 bitmap shape is
/// considered readable for the text (below: fall back to the Silver
/// TrueType shape).
const MIN_RENDERABLE_RATIO: f64 = 0.95;

/// CJK-density thresholds for choosing the Silver shape up front (upstream
/// `isCjkHeavyText`).
const CJK_HEAVY_MIN_CHARS: usize = 8;
const CJK_HEAVY_RATIO: f64 = 0.25;

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// Which bundled font a shape renders with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontKind {
    /// X.org misc 8x13 bitmap font (Latin/Greek/Cyrillic).
    Bdf8x13,
    /// Silver TrueType, one square cell per character (CJK and fallback).
    Silver,
}

/// Geometry half of a shape: font, cell pitch, and frame edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeGeometry {
    pub font: FontKind,
    pub cell_width: usize,
    pub cell_height: usize,
    pub frame_size: usize,
}

/// 8x13 glyphs on an 11px advance (extra tracking). Tool-result legibility
/// winner on Anthropic readers (f1 .806 vs .755 for plain 8on16).
pub const GEOM_11ON16: ShapeGeometry = ShapeGeometry {
    font: FontKind::Bdf8x13,
    cell_width: 11,
    cell_height: 16,
    frame_size: 1568,
};

/// 8x13 glyphs on a 22px pitch (extra leading). Winner on Google and OpenAI
/// readers (line spacing keeps line numbers legible).
pub const GEOM_8ON22: ShapeGeometry = ShapeGeometry {
    font: FontKind::Bdf8x13,
    cell_width: 8,
    cell_height: 22,
    frame_size: 1568,
};

/// 8x13 glyphs on the natural 8x16 pitch.
pub const GEOM_8ON16: ShapeGeometry = ShapeGeometry {
    font: FontKind::Bdf8x13,
    cell_width: 8,
    cell_height: 16,
    frame_size: 1568,
};

/// Silver TrueType on a 16px square cell: the CJK shape.
pub const GEOM_SILVER16: ShapeGeometry = ShapeGeometry {
    font: FontKind::Silver,
    cell_width: 16,
    cell_height: 16,
    frame_size: 1568,
};

/// Provider families with distinct image billing (formulas verified against
/// live bills in the reference implementation's resolution benchmarks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BillingFamily {
    Anthropic,
    Google,
    OpenAi,
    Unknown,
}

/// Classify a provider type string into a billing family.
pub fn billing_family(provider: &str) -> BillingFamily {
    let provider = provider.to_ascii_lowercase();
    if provider.contains("anthropic") || provider.contains("bedrock") {
        BillingFamily::Anthropic
    } else if provider.contains("gemini")
        || provider.contains("google")
        || provider.contains("vertex")
    {
        BillingFamily::Google
    } else if provider.contains("openai")
        || provider.contains("azure")
        || provider.contains("codex")
    {
        BillingFamily::OpenAi
    } else {
        // Unknown families share Anthropic's pixel-area pricing as the
        // safe ceiling.
        BillingFamily::Unknown
    }
}

/// Per-frame billed-token estimate for a square frame of `frame_size`:
/// - Anthropic: 28px patches capped at 4,784 visual tokens, +5% margin.
/// - Google: fixed `media_resolution` budget per image (1,120 at HIGH).
/// - OpenAI: 32px patches x 1.2 flagship multiplier, 10,000-patch budget.
pub fn frame_token_estimate(family: BillingFamily, frame_size: usize) -> usize {
    match family {
        BillingFamily::Google => 1_120,
        BillingFamily::OpenAi => {
            let patches = (frame_size
                .div_ceil(32)
                .saturating_mul(frame_size.div_ceil(32)))
            .min(10_000);
            (patches * 6).div_ceil(5) // ceil(patches * 1.2)
        }
        BillingFamily::Anthropic | BillingFamily::Unknown => {
            let patches = (frame_size
                .div_ceil(28)
                .saturating_mul(frame_size.div_ceil(28)))
            .min(4_784);
            (patches * 21).div_ceil(20) // ceil(patches * 1.05)
        }
    }
}

/// Whether `text` is CJK-heavy enough to prefer the Silver shape outright.
pub fn cjk_heavy(text: &str) -> bool {
    let sample: String = text.chars().take(4_096).collect();
    let wide = sample
        .chars()
        .filter(|ch| {
            let cp = *ch as u32;
            (0x1100..=0x115F).contains(&cp)
                || (0x3040..=0x33FF).contains(&cp)
                || (0x3400..=0x9FFF).contains(&cp)
                || (0xAC00..=0xD7A3).contains(&cp)
                || (0xF900..=0xFAFF).contains(&cp)
                || (0xFF00..=0xFF60).contains(&cp)
                || (0x20000..=0x3FFFD).contains(&cp)
        })
        .count();
    wide >= CJK_HEAVY_MIN_CHARS && wide as f64 / sample.chars().count() as f64 >= CJK_HEAVY_RATIO
}

static BDF_FONT: LazyLock<BitmapFont> = LazyLock::new(|| parse_bdf(BDF_8X13));
static SILVER: LazyLock<FontArc> =
    LazyLock::new(|| FontArc::try_from_slice(SILVER_TTF).expect("embedded font"));

/// Pick the frame shape for a transcript: Silver for CJK-heavy text or when
/// the 8x13 font cannot render enough of it; otherwise the billing family's
/// eval-winning grid shape.
pub fn resolve_shape(provider: &str, text: &str) -> ShapeGeometry {
    if cjk_heavy(text) {
        return GEOM_SILVER16;
    }
    let geometry = match billing_family(provider) {
        BillingFamily::Anthropic => GEOM_11ON16,
        BillingFamily::Google | BillingFamily::OpenAi | BillingFamily::Unknown => GEOM_8ON22,
    };
    let sample: String = text.chars().take(4_096).collect();
    let graphic = sample.chars().filter(|ch| !ch.is_whitespace()).count();
    if graphic == 0 {
        return geometry;
    }
    let covered = sample
        .chars()
        .filter(|ch| ch.is_whitespace() || BDF_FONT.glyphs.contains_key(ch))
        .count();
    let coverage_ratio = (covered as f64) / (sample.chars().count() as f64);
    if coverage_ratio < MIN_RENDERABLE_RATIO {
        GEOM_SILVER16
    } else {
        geometry
    }
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

/// One rendered frame: PNG bytes (base64) plus its pixel dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PngFrame {
    pub width: usize,
    pub height: usize,
    pub png_base64: String,
}

/// The persisted snapcompact archive (stored as JSON on the compaction row):
/// verbatim text at both chronological edges plus the imaged middle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapcompactArchive {
    /// Shape name, for diagnostics and future re-rendering.
    pub shape: String,
    /// Oldest slice kept as plain text (one frame's worth of lines).
    pub head_text: String,
    /// Newest slice kept as plain text (one frame's worth of lines).
    pub tail_text: String,
    /// Imaged middle, oldest first.
    pub frames: Vec<PngFrame>,
    /// Characters of the middle dropped by the frame cap.
    pub chars_dropped: usize,
}

/// Failures of archive construction.
#[derive(Debug, Error)]
pub enum SnapcompactError {
    /// The transcript fits in the two text edges; nothing to image.
    #[error("transcript fits the text edges; snapcompact not needed")]
    NotNeeded,
    #[error("render failure: {0}")]
    Render(String),
}

/// Estimated billed tokens for an archive under `family` (budgeting aid).
pub fn archive_token_estimate(archive: &SnapcompactArchive, family: BillingFamily) -> usize {
    archive
        .frames
        .iter()
        .map(|frame| frame_token_estimate(family, frame.width.max(frame.height)))
        .sum()
}

/// Build the archive from a serialized discarded transcript: normalize ->
/// wrap -> paginate; one page of lines stays verbatim as text at each
/// chronological edge, the middle renders into frames capped at
/// [`MAX_FRAMES`].
pub fn build_archive(
    transcript: &str,
    provider: &str,
) -> Result<SnapcompactArchive, SnapcompactError> {
    let geometry = resolve_shape(provider, transcript);
    let renderable = |ch: char| -> bool {
        match geometry.font {
            FontKind::Bdf8x13 => {
                BDF_FONT.glyphs.contains_key(&ch) || SILVER.glyph_id(ch) != ab_glyph::GlyphId(0)
            }
            FontKind::Silver => {
                SILVER.glyph_id(ch) != ab_glyph::GlyphId(0) || BDF_FONT.glyphs.contains_key(&ch)
            }
        }
    };
    let normalized = normalize::normalize(transcript, renderable);
    let cols = geometry.frame_size / geometry.cell_width.max(1);
    let rows = geometry.frame_size / geometry.cell_height.max(1);
    let lines = wrap(&normalized, cols);

    if lines.len() <= rows * 2 {
        return Err(SnapcompactError::NotNeeded);
    }

    let head_lines = &lines[..rows];
    let tail_lines = &lines[lines.len() - rows..];
    let middle = &lines[rows..lines.len() - rows];

    let mut frames = Vec::new();
    for chunk in middle.chunks(rows).take(MAX_FRAMES) {
        let spec = render::GridSpec {
            cell_width: geometry.cell_width,
            cell_height: geometry.cell_height,
            width: geometry.frame_size,
            max_height: geometry.frame_size,
        };
        let (pixels, width, height, indexed) = match geometry.font {
            FontKind::Bdf8x13 => {
                let (pixels, height) = render::render_bdf_frame(chunk, &BDF_FONT, spec);
                (pixels, geometry.frame_size, height, true)
            }
            FontKind::Silver => {
                let (pixels, height) = render::render_silver_frame(chunk, &SILVER, spec);
                (pixels, geometry.frame_size, height, false)
            }
        };
        let png = if indexed {
            render::encode_indexed_png(&pixels, width, height)
        } else {
            render::encode_rgb_png(&pixels, width, height)
        };
        frames.push(PngFrame {
            width,
            height,
            png_base64: render::png_base64(&png),
        });
    }
    let kept_lines = (middle.len().div_ceil(rows)).min(MAX_FRAMES) * rows;
    let chars_dropped = middle
        .get(kept_lines..)
        .map(|dropped| dropped.iter().map(|line| line.chars().count()).sum())
        .unwrap_or(0);

    Ok(SnapcompactArchive {
        shape: shape_name(geometry),
        head_text: head_lines.join("\n"),
        tail_text: tail_lines.join("\n"),
        frames,
        chars_dropped,
    })
}

fn shape_name(geometry: ShapeGeometry) -> String {
    match geometry {
        GEOM_11ON16 => "11on16-bw".into(),
        GEOM_8ON22 => "8on22-bw".into(),
        GEOM_8ON16 => "8on16-bw".into(),
        GEOM_SILVER16 => "silver16-bw".into(),
        _ => "custom".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latin_transcript(pages: usize) -> String {
        (0..pages)
            .map(|page| {
                (0..40)
                    .map(|line| {
                        format!("page {page} line {line} of the archived conversation transcript")
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn billing_families_map() {
        assert_eq!(billing_family("anthropic"), BillingFamily::Anthropic);
        assert_eq!(billing_family("custom_anthropic"), BillingFamily::Anthropic);
        assert_eq!(billing_family("gemini"), BillingFamily::Google);
        assert_eq!(billing_family("custom_openai"), BillingFamily::OpenAi);
        assert_eq!(billing_family("mystery"), BillingFamily::Unknown);
    }

    #[test]
    fn billing_estimates_match_formulas() {
        // Anthropic: ceil((1568/28)^2 capped 4784 * 1.05) = ceil(3136*1.05)
        assert_eq!(frame_token_estimate(BillingFamily::Anthropic, 1568), 3293);
        assert_eq!(frame_token_estimate(BillingFamily::Google, 1568), 1_120);
        // OpenAI: ceil((1568/32)^2 * 1.2) = ceil(2401*1.2)
        assert_eq!(frame_token_estimate(BillingFamily::OpenAi, 1568), 2_882);
    }

    #[test]
    fn cjk_heavy_detection() {
        assert!(cjk_heavy(&"\u{4f60}\u{597d}\u{4e16}\u{754c}".repeat(4)));
        assert!(!cjk_heavy("plain english transcript"));
    }

    #[test]
    fn shape_resolution_prefers_silver_for_cjk() {
        let text = "\u{4f60}\u{597d}\u{4e16}\u{754c}".repeat(10);
        assert_eq!(resolve_shape("anthropic", &text), GEOM_SILVER16);
        assert_eq!(resolve_shape("anthropic", "latin text"), GEOM_11ON16);
        assert_eq!(resolve_shape("gemini", "latin text"), GEOM_8ON22);
    }

    #[test]
    fn archive_builds_frames_with_text_edges() {
        let transcript = latin_transcript(6);
        let archive = build_archive(&transcript, "custom_openai").expect("archive builds");
        assert!(!archive.frames.is_empty());
        assert!(archive.frames.len() <= MAX_FRAMES);
        assert!(archive.head_text.contains("page 0"));
        assert!(archive.tail_text.contains("page 5"));
        for frame in &archive.frames {
            assert!(frame.width > 0);
            assert!(frame.height > 0);
            // Decodes as a real PNG.
            let bytes = render::png_base64_decode(&frame.png_base64);
            assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']));
        }
        // 6 pages of 40 lines fit inside the 8-frame budget: nothing drops.
        assert_eq!(archive.chars_dropped, 0);
    }

    #[test]
    fn archive_caps_frames_and_counts_dropped_chars() {
        let transcript = latin_transcript(40);
        let archive = build_archive(&transcript, "custom_openai").expect("archive builds");
        assert_eq!(archive.frames.len(), MAX_FRAMES);
        assert!(archive.chars_dropped > 0);
    }

    #[test]
    fn small_transcript_not_needed() {
        let err = build_archive("short transcript", "custom_openai").unwrap_err();
        assert!(matches!(err, SnapcompactError::NotNeeded));
    }

    #[test]
    fn cjk_archive_uses_silver_shape() {
        let cjk: String = (0..200)
            .map(|n| format!("\u{7b2c}{n}\u{884c}\u{4f60}\u{597d}\u{4e16}\u{754c}\u{6d4b}\u{8bd5}\u{6587}\u{672c}\u{5185}\u{5bb9}"))
            .collect::<Vec<_>>()
            .join("\n");
        let archive = build_archive(&cjk, "anthropic").expect("archive builds");
        assert_eq!(archive.shape, "silver16-bw");
        assert!(!archive.frames.is_empty());
    }
}
