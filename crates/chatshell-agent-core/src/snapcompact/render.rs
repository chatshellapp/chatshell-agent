//! Frame compositing and PNG encoding for snapcompact. The v1 renderer
//! covers the eval-winning shapes — X.org 8x13 bitmap glyphs on widened
//! cell pitches (black ink, dim tool-noise spans) and the Silver TrueType
//! fallback for CJK — as palette-indexed / RGB PNGs. Sentence-hue cycling,
//! line repetition, and Lanczos stretching belong to shapes v1 does not
//! ship.

use std::collections::HashMap;

use ab_glyph::{Font as _, FontArc, GlyphId, PxScale};
use base64::Engine as _;
use png::Encoder;

use super::fonts::{BitmapFont, Glyph};
use super::normalize::{DIM_OFF, DIM_ON, NEWLINE_GLYPH, cell_units};

/// Palette slots shared with the reference implementation's renderer: 0 background, 1-6 reserved
/// sentence hues, 7 black ink, 8 repeat band, 9 dim gray ink.
pub const BG: u8 = 0;
pub const INK_BLACK: u8 = 7;
pub const INK_DIM: u8 = 9;

/// Minimum rendered frame height: some vision backends reject images of
/// 32px or smaller (short final archive frames), so hug-the-rows never
/// goes below this floor (clamped to `max_height`).
pub const MIN_FRAME_HEIGHT: usize = 64;

fn palette() -> Vec<[u8; 3]> {
    vec![
        [0xFF, 0xFF, 0xFF], // 0 background
        [0x00, 0x00, 0x00], // 1 hue slot (reserved)
        [0x00, 0x00, 0x00], // 2 hue slot (reserved)
        [0x00, 0x00, 0x00], // 3 hue slot (reserved)
        [0x00, 0x00, 0x00], // 4 hue slot (reserved)
        [0x00, 0x00, 0x00], // 5 hue slot (reserved)
        [0x00, 0x00, 0x00], // 6 hue slot (reserved)
        [0x00, 0x00, 0x00], // 7 black ink
        [0xF0, 0xF0, 0xF0], // 8 repeat band (reserved)
        [0xB8, 0xB8, 0xB8], // 9 dim gray ink
    ]
}

/// Blit one glyph's bitmask rows at (`left`, `top`), clipped to the canvas.
fn blit_glyph(
    pixels: &mut [u8],
    spec: &GridSpec,
    glyph: &Glyph,
    ascent: usize,
    left: usize,
    row: usize,
    ink: u8,
) {
    let width = spec.width;
    // The canvas height is exact: derive it from the pixel buffer.
    let height = pixels.len() / width;
    let cell_height = spec.cell_height;
    let top = row * cell_height + glyph.top_offset(ascent).max(0) as usize;
    for glyph_row in 0..glyph.height {
        let y = top + glyph_row;
        if y >= height {
            break;
        }
        let bytes_per_row = glyph.width.div_ceil(8);
        for glyph_col in 0..glyph.width {
            let byte = glyph.bitmap[glyph_row * bytes_per_row + glyph_col / 8];
            if byte & (0x80 >> (glyph_col % 8)) == 0 {
                continue;
            }
            let x = left + glyph.dx.max(0) as usize + glyph_col;
            if x < width {
                pixels[y * width + x] = ink;
            }
        }
    }
}

/// Grid geometry for one frame render: cell pitch, canvas width, and the
/// height cap (the produced height hugs the rows actually used).
#[derive(Debug, Clone, Copy)]
pub struct GridSpec {
    pub cell_width: usize,
    pub cell_height: usize,
    pub width: usize,
    pub max_height: usize,
}

impl GridSpec {
    fn cols(&self) -> usize {
        self.width / self.cell_width.max(1)
    }

    fn used_height(&self, lines: usize) -> usize {
        let used_rows = lines.min(self.max_height / self.cell_height.max(1));
        (used_rows * self.cell_height).max(MIN_FRAME_HEIGHT.min(self.max_height))
    }
}

/// Rasterize pre-wrapped lines onto a cell grid as palette indices, row
/// major with no wrapping (the caller paginates). Dim toggles switch ink to
/// [`INK_DIM`] without consuming a cell; [`NEWLINE_GLYPH`] fills its whole
/// cell with black. Code points missing from the bitmap font draw through
/// the embedded Silver TrueType fallback when it has a glyph (wide CJK
/// spans two cells); only glyphs in neither font fall back to '?'. Frame
/// height hugs the rows actually used, capped at the spec.
pub fn render_bdf_frame(
    lines: &[String],
    font: &BitmapFont,
    silver: &FontArc,
    spec: GridSpec,
) -> (Vec<u8>, usize) {
    let cell_width = spec.cell_width;
    let cell_height = spec.cell_height;
    let width = spec.width;
    let cols = spec.cols();
    let height = spec.used_height(lines.len());
    let used_rows = lines.len().min(spec.max_height / spec.cell_height.max(1));
    let mut pixels = vec![BG; width * height];

    let question_mark = font.glyphs.get(&'?');
    let mut silver_cache: HashMap<(char, usize), Option<Rasterized>> = HashMap::new();
    for (row, line) in lines.iter().take(used_rows).enumerate() {
        let mut ink = INK_BLACK;
        let mut col = 0usize;
        for ch in line.chars() {
            match ch {
                DIM_ON => ink = INK_DIM,
                DIM_OFF => ink = INK_BLACK,
                NEWLINE_GLYPH => {
                    if col < cols {
                        for y in row * cell_height..(row + 1) * cell_height {
                            for x in col * cell_width..(col + 1) * cell_width {
                                if y < height {
                                    pixels[y * width + x] = INK_BLACK;
                                }
                            }
                        }
                    }
                    col += 1;
                }
                ch => {
                    let units = cell_units(ch);
                    if col + units > cols {
                        break; // a CJK wide glyph never straddles the right edge
                    }
                    if let Some(glyph) = font.glyphs.get(&ch) {
                        blit_glyph(
                            &mut pixels,
                            &spec,
                            glyph,
                            font.ascent,
                            col * cell_width,
                            row,
                            ink,
                        );
                    } else if let Some(glyph) = silver_fallback(
                        &mut silver_cache,
                        silver,
                        ch,
                        units,
                        cell_width,
                        cell_height,
                    ) {
                        let box_left = col * cell_width;
                        let box_top = row * cell_height;
                        let box_w = units * cell_width;
                        let center_x =
                            box_left + box_w.saturating_sub(glyph.width.max(0) as usize) / 2;
                        let center_y =
                            box_top + cell_height.saturating_sub(glyph.height.max(0) as usize) / 2;
                        let offset_x = center_x as i32 - glyph.min_x;
                        let offset_y = center_y as i32 - glyph.min_y;
                        for (x, y, coverage) in &glyph.samples {
                            if *coverage < 0.5 {
                                continue; // quantize anti-aliasing to bitmap ink
                            }
                            let px = (*x + offset_x).max(0) as usize;
                            let py = (*y + offset_y).max(0) as usize;
                            if px < width && py < height {
                                pixels[py * width + px] = ink;
                            }
                        }
                    } else if let Some(glyph) = question_mark {
                        blit_glyph(
                            &mut pixels,
                            &spec,
                            glyph,
                            font.ascent,
                            col * cell_width,
                            row,
                            ink,
                        );
                    }
                    col += units;
                }
            }
        }
    }
    (pixels, height)
}

/// Rasterize `ch` with Silver sized to a `units`-cell box (the bitmap
/// renderer's per-character fallback for glyphs its font lacks).
#[allow(clippy::too_many_arguments)]
fn silver_fallback(
    cache: &mut HashMap<(char, usize), Option<Rasterized>>,
    silver: &FontArc,
    ch: char,
    units: usize,
    cell_width: usize,
    cell_height: usize,
) -> Option<Rasterized> {
    let key = (ch, units);
    cache
        .entry(key)
        .or_insert_with(|| {
            let glyph_id = silver.glyph_id(ch);
            if glyph_id == GlyphId(0) {
                None
            } else {
                let box_px = (units.max(1) * cell_width).min(cell_height.max(cell_width));
                rasterize_ttf_glyph(silver, glyph_id, box_px as f32)
            }
        })
        .clone()
}

/// One rasterized TrueType glyph: coverage samples at glyph-space offsets.
#[derive(Clone)]
struct Rasterized {
    samples: Vec<(i32, i32, f32)>,
    min_x: i32,
    min_y: i32,
    width: i32,
    height: i32,
}

fn rasterize_ttf_glyph(font: &FontArc, glyph_id: GlyphId, scale: f32) -> Option<Rasterized> {
    let glyph = ab_glyph::Glyph {
        id: glyph_id,
        scale: PxScale::from(scale),
        position: ab_glyph::Point { x: 0.0, y: 0.0 },
    };
    let outlined = font.outline_glyph(glyph)?;
    let mut samples = Vec::new();
    outlined.draw(|x, y, coverage| {
        if coverage > 0.02 {
            samples.push((x as i32, y as i32, coverage));
        }
    });
    if samples.is_empty() {
        return None;
    }
    // Derive the glyph's extent from the samples themselves so placement is
    // independent of the draw callback's coordinate convention.
    let min_x = samples.iter().map(|(x, _, _)| *x).min()?;
    let min_y = samples.iter().map(|(_, y, _)| *y).min()?;
    let max_x = samples.iter().map(|(x, _, _)| *x).max()?;
    let max_y = samples.iter().map(|(_, y, _)| *y).max()?;
    Some(Rasterized {
        samples,
        min_x,
        min_y,
        width: max_x - min_x + 1,
        height: max_y - min_y + 1,
    })
}

/// Rasterize pre-wrapped lines with the Silver TrueType font into an RGB8
/// buffer: one cell per character, glyphs vertically centered in the cell
/// and horizontally centered when narrower. Dim spans render in gray ink.
pub fn render_silver_frame(lines: &[String], font: &FontArc, spec: GridSpec) -> (Vec<u8>, usize) {
    let cell_width = spec.cell_width;
    let cell_height = spec.cell_height;
    let width = spec.width;
    let cols = spec.cols();
    let height = spec.used_height(lines.len());
    // White background, interleaved RGB.
    let mut pixels = vec![0xFFu8; width * height * 3];
    // Upstream sizes Silver at its design size (the cell height): the
    // glyph fills the cell instead of floating at 82% of it.
    let scale = cell_height as f32;

    let mut cache: HashMap<char, Option<Rasterized>> = HashMap::new();
    let put_pixel = |pixels: &mut [u8], x: usize, y: usize, coverage: f32, dim: bool| {
        if x >= width || y >= height {
            return;
        }
        let index = (y * width + x) * 3;
        let delta = (coverage * if dim { 71.0 } else { 255.0 }) as u8;
        pixels[index] = pixels[index].saturating_sub(delta);
        pixels[index + 1] = pixels[index + 1].saturating_sub(delta);
        pixels[index + 2] = pixels[index + 2].saturating_sub(delta);
    };

    let used_rows = lines.len().min(spec.max_height / spec.cell_height.max(1));
    for (row, line) in lines.iter().take(used_rows).enumerate() {
        let mut dim = false;
        let mut col = 0usize;
        for ch in line.chars() {
            match ch {
                DIM_ON => dim = true,
                DIM_OFF => dim = false,
                NEWLINE_GLYPH => {
                    if col < cols {
                        for y in row * cell_height..(row + 1) * cell_height {
                            for x in col * cell_width..(col + 1) * cell_width {
                                if y < height {
                                    let index = (y * width + x) * 3;
                                    pixels[index..index + 3].fill(0);
                                }
                            }
                        }
                    }
                    col += 1;
                }
                ch => {
                    if col >= cols {
                        break;
                    }
                    let rasterized = cache
                        .entry(ch)
                        .or_insert_with(|| {
                            let glyph_id = font.glyph_id(ch);
                            if glyph_id == GlyphId(0) {
                                None // .notdef: no glyph for this character
                            } else {
                                rasterize_ttf_glyph(font, glyph_id, scale)
                            }
                        })
                        .clone();
                    if let Some(glyph) = rasterized {
                        let center_x = col * cell_width
                            + (cell_width.saturating_sub(glyph.width.max(0) as usize)) / 2;
                        let center_y = row * cell_height
                            + (cell_height.saturating_sub(glyph.height.max(0) as usize)) / 2;
                        let offset_x = center_x as i32 - glyph.min_x;
                        let offset_y = center_y as i32 - glyph.min_y;
                        for (x, y, coverage) in glyph.samples {
                            put_pixel(
                                &mut pixels,
                                (x + offset_x).max(0) as usize,
                                (y + offset_y).max(0) as usize,
                                coverage,
                                dim,
                            );
                        }
                    }
                    col += 1;
                }
            }
        }
    }
    (pixels, height)
}

/// Encode a palette-indexed bitmap as an indexed PNG (no row filtering: the
/// glyph bitmap is minimal-entropy already).
pub fn encode_indexed_png(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = Encoder::new(&mut out, width as u32, height as u32);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::Eight);
        let palette: Vec<u8> = palette()
            .iter()
            .flat_map(|rgb| rgb.iter().copied())
            .collect();
        encoder.set_palette(&palette);
        let mut writer = encoder.write_header().expect("png header");
        writer.set_filter(png::FilterType::NoFilter);
        writer.write_image_data(pixels).expect("png data");
    }
    out
}

/// Encode an interleaved RGB8 buffer as PNG (adaptive filtering suits the
/// continuous-tone TrueType coverage).
pub fn encode_rgb_png(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = Encoder::new(&mut out, width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(pixels).expect("png data");
    }
    out
}

/// PNG bytes as a data-URL-ready base64 string.
pub fn png_base64(png_bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(png_bytes)
}

/// Decode a [`png_base64`] string back to PNG bytes (verification paths).
pub fn png_base64_decode(encoded: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapcompact::fonts::{BDF_8X13, SILVER_TTF, parse_bdf};
    use std::sync::LazyLock;

    static BDF: LazyLock<BitmapFont> = LazyLock::new(|| parse_bdf(BDF_8X13));
    static SILVER: LazyLock<FontArc> =
        LazyLock::new(|| FontArc::try_from_slice(SILVER_TTF).expect("silver loads"));

    #[test]
    fn bdf_frame_hugs_height_with_min_floor() {
        // A single short line never goes below MIN_FRAME_HEIGHT: vision
        // backends reject tiny images.
        let (_, height) = render_bdf_frame(
            &["Hello".to_string()],
            &BDF,
            &SILVER,
            GridSpec {
                cell_width: 8,
                cell_height: 13,
                width: 80,
                max_height: 1568,
            },
        );
        assert_eq!(height, MIN_FRAME_HEIGHT);

        // Several lines still hug the rows actually printed.
        let lines: Vec<String> = (0..10).map(|n| format!("line {n}")).collect();
        let (pixels, height) = render_bdf_frame(
            &lines,
            &BDF,
            &SILVER,
            GridSpec {
                cell_width: 8,
                cell_height: 13,
                width: 80,
                max_height: 1568,
            },
        );
        assert_eq!(height, 10 * 13);
        assert!(pixels.iter().any(|p| *p != BG), "no ink drawn");
        // Blank leading rows stay background above the glyph tops only
        // within the cell; count distinct ink columns to sanity-check width.
        let ink_columns = (0..80 / 8)
            .filter(|col| {
                (0..13).any(|y| {
                    pixels[y * 80 + col * 8..(y * 80 + (col + 1) * 8)].contains(&INK_BLACK)
                })
            })
            .count();
        assert!(
            ink_columns >= 4,
            "expected ink in ~10 cells, got {ink_columns}"
        );
    }

    #[test]
    fn bdf_frame_draws_cjk_through_silver_fallback() {
        let spec = GridSpec {
            cell_width: 11,
            cell_height: 16,
            width: 88,
            max_height: 1568,
        };
        let (cjk_pixels, _) =
            render_bdf_frame(&["\u{4f60}\u{597d} ok".to_string()], &BDF, &SILVER, spec);
        let (qm_pixels, _) = render_bdf_frame(&["?? ok".to_string()], &BDF, &SILVER, spec);
        assert!(cjk_pixels.iter().any(|p| *p != BG), "CJK ink drawn");
        assert_ne!(
            cjk_pixels, qm_pixels,
            "CJK must render as Silver glyphs, not question marks"
        );

        // CJK wide glyphs occupy two cells: the first glyph's ink spans
        // columns 0-1 and the ASCII tail sits at columns 4-5.
        let ink_columns = |pixels: &[u8]| {
            (0..88 / 11)
                .filter(|col| {
                    (0..16).any(|y| pixels[y * 88 + col * 11..y * 88 + (col + 1) * 11] != [BG])
                })
                .collect::<Vec<_>>()
        };
        let cjk_cols = ink_columns(&cjk_pixels);
        assert!(
            cjk_cols.contains(&0) && cjk_cols.contains(&1),
            "CJK wide glyph spans two cells"
        );
        assert!(
            cjk_cols.contains(&4) && cjk_cols.contains(&5),
            "ok at cols 4-5"
        );
    }

    #[test]
    fn bdf_frame_respects_dim_and_full_block() {
        let lines = vec![format!("{DIM_ON}ab{DIM_OFF}{NEWLINE_GLYPH}")];
        let (pixels, _height) = render_bdf_frame(
            &lines,
            &BDF,
            &SILVER,
            GridSpec {
                cell_width: 8,
                cell_height: 13,
                width: 40,
                max_height: 1568,
            },
        );
        assert!(pixels.contains(&INK_DIM), "dim ink drawn");
        assert!(pixels.contains(&INK_BLACK), "full block drawn");
    }

    #[test]
    fn bdf_frame_caps_height() {
        let lines: Vec<String> = (0..400).map(|n| format!("line{n}")).collect();
        let (_pixels, height) = render_bdf_frame(
            &lines,
            &BDF,
            &SILVER,
            GridSpec {
                cell_width: 8,
                cell_height: 13,
                width: 80,
                max_height: 130,
            },
        );
        assert_eq!(height, 130);
    }

    #[test]
    fn silver_frame_renders_cjk() {
        let lines = vec!["\u{4f60}\u{597d}".to_string()];
        let (pixels, height) = render_silver_frame(
            &lines,
            &SILVER,
            GridSpec {
                cell_width: 16,
                cell_height: 16,
                width: 64,
                max_height: 1568,
            },
        );
        assert_eq!(height, MIN_FRAME_HEIGHT);
        let dark = pixels.chunks_exact(3).filter(|px| px[0] < 128).count();
        assert!(dark > 4, "expected CJK ink, got {dark} dark pixels");
    }

    #[test]
    fn indexed_png_round_trips() {
        let lines = vec!["Test 123".to_string()];
        let (pixels, height) = render_bdf_frame(
            &lines,
            &BDF,
            &SILVER,
            GridSpec {
                cell_width: 8,
                cell_height: 13,
                width: 80,
                max_height: 1568,
            },
        );
        let png = encode_indexed_png(&pixels, 80, height);
        assert!(png.starts_with(&[0x89, b'P', b'N', b'G']));
        let decoder = png::Decoder::new(png.as_slice());
        let mut reader = decoder.read_info().expect("decode");
        assert_eq!(reader.info().width, 80);
        assert_eq!(reader.info().height as usize, height);
        let mut buffer = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut buffer).expect("frame");
        assert_eq!(buffer, pixels);
    }

    #[test]
    fn rgb_png_round_trips() {
        let lines = vec!["ok".to_string()];
        let (pixels, height) = render_silver_frame(
            &lines,
            &SILVER,
            GridSpec {
                cell_width: 16,
                cell_height: 16,
                width: 64,
                max_height: 1568,
            },
        );
        let png = encode_rgb_png(&pixels, 64, height);
        let decoder = png::Decoder::new(png.as_slice());
        let mut reader = decoder.read_info().expect("decode");
        let mut buffer = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut buffer).expect("frame");
        assert_eq!(buffer, pixels);
    }

    #[test]
    fn base64_encodes() {
        assert_eq!(png_base64(b"hi"), "aGk=");
    }
}
