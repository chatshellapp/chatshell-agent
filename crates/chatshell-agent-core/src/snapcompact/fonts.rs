//! Pixel-font parsing for the snapcompact renderer: X.org BDF bitmap fonts
//! (Latin and common diacritics) and the embedded Silver TrueType fallback
//! for CJK and other non-Latin text.
//!

use std::collections::HashMap;

use ab_glyph::FontRef;

/// The X.org misc 8x13 bitmap font (ISO10646-1; Latin, Greek, Cyrillic and
/// common symbols). Attribution: Markus Kuhn's ucs-fonts.
pub const BDF_8X13: &str = include_str!("fonts/8x13.bdf");

/// The bundled Silver TrueType font, used for CJK and other text the bitmap
/// font cannot render.
pub const SILVER_TTF: &[u8] = include_bytes!("fonts/Silver.ttf");

/// One bitmap glyph: rows of set/unset pixels with its BDF placement.
#[derive(Debug, Clone)]
pub struct Glyph {
    pub bitmap: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// X displacement of the bitmap's left edge from the cell origin.
    pub dx: i32,
    /// Y displacement of the bitmap's bottom edge from the baseline.
    pub dy: i32,
}

impl Glyph {
    /// Row index (from the cell top) this glyph's bitmap starts at, given
    /// the font's ascent.
    pub fn top_offset(&self, ascent: usize) -> i32 {
        ascent as i32 - self.dy - self.height as i32
    }
}

/// A parsed bitmap font.
#[derive(Debug, Clone)]
pub struct BitmapFont {
    pub glyphs: HashMap<char, Glyph>,
    pub ascent: usize,
}

/// Parse a BDF font, keeping every glyph with a usable encoding.
pub fn parse_bdf(text: &str) -> BitmapFont {
    let mut glyphs = HashMap::new();
    let mut ascent = 11usize;

    let mut lines = text.lines().map(str::trim_end);
    while let Some(line) = lines.next() {
        if let Some(value) = line.strip_prefix("FONT_ASCENT ") {
            ascent = value.trim().parse().unwrap_or(11);
        } else if line.starts_with("STARTCHAR ") {
            let mut encoding: Option<u32> = None;
            let mut width = 0usize;
            let mut height = 0usize;
            let mut dx = 0i32;
            let mut dy = 0i32;
            let mut bitmap: Vec<u8> = Vec::new();
            let mut in_bitmap = false;
            for inner in lines.by_ref() {
                if in_bitmap {
                    if inner.starts_with("ENDCHAR") {
                        break;
                    }
                    if !inner.is_empty() {
                        // Two hex digits per pixel column, row-major.
                        let byte_count = inner.len() / 2;
                        for column in 0..byte_count {
                            let Ok(byte) =
                                u8::from_str_radix(&inner[column * 2..column * 2 + 2], 16)
                            else {
                                continue;
                            };
                            bitmap.push(byte);
                        }
                    }
                } else if let Some(value) = inner.strip_prefix("ENCODING ") {
                    encoding = value.trim().parse().ok();
                } else if let Some(value) = inner.strip_prefix("BBX ") {
                    let mut parts = value.split_whitespace();
                    if let (Some(w), Some(h), Some(x), Some(y)) =
                        (parts.next(), parts.next(), parts.next(), parts.next())
                    {
                        width = w.parse().unwrap_or(0);
                        height = h.parse().unwrap_or(0);
                        dx = x.parse().unwrap_or(0);
                        dy = y.parse().unwrap_or(0);
                    }
                } else if inner.starts_with("BITMAP") {
                    in_bitmap = true;
                }
            }
            if let Some(code) = encoding
                && let Some(ch) = char::from_u32(code)
                && width > 0
                && height > 0
                && bitmap.len() == width.div_ceil(8) * height
            {
                glyphs.insert(
                    ch,
                    Glyph {
                        bitmap,
                        width,
                        height,
                        dx,
                        dy,
                    },
                );
            }
        }
    }
    BitmapFont { glyphs, ascent }
}

/// The parsed Silver TrueType font (panics only on a corrupt embedded asset).
pub fn silver_font() -> FontRef<'static> {
    FontRef::try_from_slice(SILVER_TTF).expect("embedded Silver.ttf is valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ab_glyph::Font as _;

    #[test]
    fn parses_the_bundled_bdf() {
        let font = parse_bdf(BDF_8X13);
        for expected in ['A', 'a', '0', ' ', '{', '~'] {
            assert!(font.glyphs.contains_key(&expected), "missing {expected:?}");
        }
        // Latin-1 coverage (NFKD-folded text lands here).
        assert!(font.glyphs.contains_key(&'\u{00e9}'));
        // Control codes are never encoded as glyphs.
        assert!(!font.glyphs.contains_key(&'\u{000e}'));
        // Glyph placement sanity: a full-cell glyph sits at the top.
        let a = &font.glyphs[&'A'];
        assert!(
            a.top_offset(font.ascent) <= 0,
            "A at {}",
            a.top_offset(font.ascent)
        );
        assert!(a.bitmap.len() == a.width.div_ceil(8) * a.height);
        // Some ink in the bitmap.
        assert!(a.bitmap.iter().any(|b| *b != 0));
        // CJK is not covered by this font (Silver handles it).
        assert!(!font.glyphs.contains_key(&'\u{4f60}'));
    }

    #[test]
    fn silver_ttf_loads() {
        let font = silver_font();
        assert_ne!(font.glyph_id('\u{4f60}').0, 0, "Silver covers CJK");
        assert_ne!(font.glyph_id('A').0, 0);
    }
}
