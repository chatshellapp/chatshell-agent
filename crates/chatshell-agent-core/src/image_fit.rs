//! Provider image limits: one resize ladder for every image the app sends.
//!
//! Providers cap image input (Anthropic: 5 MB base64 per image, 8000 px,
//! and down-samples anything above ~1568 px anyway; OpenAI and Gemini have
//! similar budgets) and accept only PNG, JPEG, GIF, and WebP. Images that
//! already fit pass through byte-for-byte. Everything else is decoded,
//! scaled to fit [`MAX_SIDE`], and re-encoded: PNG when the source has
//! transparency and the PNG fits, otherwise JPEG at decreasing quality,
//! then smaller dimensions, until it fits [`MAX_BASE64_BYTES`].
//!
//! Used by the read tool for inline image results and by hosts for
//! attached images. The desktop frontend applies the same limits at attach
//! time (`src/lib/image-fit.ts`) so oversized images never cross IPC.

use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader};
use std::io::Cursor;

/// Longest side sent to a provider.
pub const MAX_SIDE: u32 = 2000;
/// Base64 budget per image (Anthropic's per-image limit).
pub const MAX_BASE64_BYTES: usize = 5 * 1024 * 1024;
/// Encoded byte budget matching [`MAX_BASE64_BYTES`].
pub const MAX_ENCODED_BYTES: usize = MAX_BASE64_BYTES / 4 * 3;
/// JPEG qualities tried at each size.
const JPEG_QUALITIES: [u8; 4] = [85, 70, 55, 40];
/// Each further size step scales by this factor.
const SCALE_STEP: f64 = 0.75;
/// Stop shrinking below this longest side.
const MIN_SIDE: u32 = 200;
/// Fewest total pixels a provider accepts (xAI rejects images below 512
/// pixels, e.g. a 20x20 icon); smaller images are upscaled.
pub const MIN_PIXELS: u64 = 512;

/// An image ready for a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FittedImage {
    pub bytes: Vec<u8>,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
    /// False when the input passed through unchanged.
    pub changed: bool,
}

fn provider_format(format: ImageFormat) -> Option<&'static str> {
    match format {
        ImageFormat::Png => Some("image/png"),
        ImageFormat::Jpeg => Some("image/jpeg"),
        ImageFormat::Gif => Some("image/gif"),
        ImageFormat::WebP => Some("image/webp"),
        _ => None,
    }
}

/// [`fit_image`] for images from sources we do not control (MCP servers).
///
/// `fit_image` passes an image that already fits through byte-for-byte
/// after reading only its header. Tolerant decoders (ours included) accept
/// files that strict provider decoders reject, so an untrusted image is
/// fully decoded, and a PNG is re-encoded: a defect such as a bad
/// ancillary-chunk CRC (seen in the MCP reference server's `getTinyImage`,
/// rejected by xAI as "Invalid PNG image") passes every decoder here but
/// fails the whole provider request. PNG re-encoding is lossless; other
/// formats pass through once they decode.
pub fn fit_untrusted_image(bytes: &[u8]) -> Result<FittedImage, String> {
    let result = fit_image(bytes)?;
    if result.changed {
        // Already decoded and re-encoded by `fit_image`.
        return Ok(result);
    }
    let image = image::load_from_memory(&result.bytes)
        .map_err(|e| format!("could not decode image: {e}"))?;
    if result.mime_type == "image/png" {
        return fit_image(&encode_png(&image)?).map(|clean| FittedImage {
            changed: true,
            ..clean
        });
    }
    Ok(result)
}

/// Fit `bytes` to the provider limits. Errors when the bytes are not a
/// decodable image or cannot be brought under the budget.
pub fn fit_image(bytes: &[u8]) -> Result<FittedImage, String> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("unreadable image: {e}"))?;
    let format = reader
        .format()
        .ok_or_else(|| "unknown image format".to_string())?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|e| format!("unreadable image: {e}"))?;

    if (width as u64) * (height as u64) < MIN_PIXELS {
        return upscale_tiny(bytes, format, width, height);
    }

    if let Some(mime) = provider_format(format)
        && width.max(height) <= MAX_SIDE
        && bytes.len() <= MAX_ENCODED_BYTES
    {
        return Ok(FittedImage {
            bytes: bytes.to_vec(),
            mime_type: mime.to_string(),
            width,
            height,
            changed: false,
        });
    }

    let image = image::load_from_memory_with_format(bytes, format)
        .map_err(|e| format!("could not decode image: {e}"))?;
    let has_alpha = image.color().has_alpha();

    let mut side = width.max(height).min(MAX_SIDE);
    loop {
        let scaled = scale_to(&image, side);
        let (w, h) = scaled.dimensions();
        if has_alpha {
            let png = encode_png(&scaled)?;
            if png.len() <= MAX_ENCODED_BYTES {
                return Ok(fitted(png, "image/png", w, h));
            }
        }
        let rgb = DynamicImage::ImageRgb8(scaled.to_rgb8());
        for quality in JPEG_QUALITIES {
            let jpeg = encode_jpeg(&rgb, quality)?;
            if jpeg.len() <= MAX_ENCODED_BYTES {
                return Ok(fitted(jpeg, "image/jpeg", w, h));
            }
        }
        let next = (side as f64 * SCALE_STEP) as u32;
        if next < MIN_SIDE {
            return Err(format!(
                "image is too large to send even at {w}x{h} (limit {} MB)",
                MAX_BASE64_BYTES / (1024 * 1024)
            ));
        }
        side = next;
    }
}

/// Upscale an image below [`MIN_PIXELS`] by the smallest whole factor that
/// reaches it. Nearest-neighbor keeps icons and pixel art crisp; PNG keeps
/// it lossless (the result is tiny either way).
fn upscale_tiny(
    bytes: &[u8],
    format: ImageFormat,
    width: u32,
    height: u32,
) -> Result<FittedImage, String> {
    let pixels = (width as u64) * (height as u64);
    if pixels == 0 {
        return Err("image has no pixels".to_string());
    }
    let image = image::load_from_memory_with_format(bytes, format)
        .map_err(|e| format!("could not decode image: {e}"))?;
    let mut factor: u32 = 2;
    while pixels * (factor as u64).pow(2) < MIN_PIXELS {
        factor += 1;
    }
    let (w, h) = (width * factor, height * factor);
    let scaled = image.resize_exact(w, h, image::imageops::FilterType::Nearest);
    Ok(fitted(encode_png(&scaled)?, "image/png", w, h))
}

fn fitted(bytes: Vec<u8>, mime: &str, width: u32, height: u32) -> FittedImage {
    FittedImage {
        bytes,
        mime_type: mime.to_string(),
        width,
        height,
        changed: true,
    }
}

fn scale_to(image: &DynamicImage, side: u32) -> DynamicImage {
    let (w, h) = image.dimensions();
    if w.max(h) <= side {
        return image.clone();
    }
    image.resize(side, side, image::imageops::FilterType::Lanczos3)
}

fn encode_png(image: &DynamicImage) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .map_err(|e| format!("PNG encode failed: {e}"))?;
    Ok(out)
}

fn encode_jpeg(image: &DynamicImage, quality: u8) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
    image
        .write_with_encoder(encoder)
        .map_err(|e| format!("JPEG encode failed: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage, Rgba, RgbaImage};

    fn encode(image: DynamicImage, format: ImageFormat) -> Vec<u8> {
        let mut out = Vec::new();
        image.write_to(&mut Cursor::new(&mut out), format).unwrap();
        out
    }

    /// A valid PNG with an ancillary `tEXt` chunk whose CRC is wrong.
    fn png_with_bad_ancillary_crc() -> Vec<u8> {
        let png = encode(
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(32, 32, Rgba([1, 2, 3, 128]))),
            ImageFormat::Png,
        );
        // Insert after the 8-byte signature and 25-byte IHDR chunk.
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&5u32.to_be_bytes());
        chunk.extend_from_slice(b"tEXtk\0abc");
        chunk.extend_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        let mut out = png[..33].to_vec();
        out.extend_from_slice(&chunk);
        out.extend_from_slice(&png[33..]);
        out
    }

    #[test]
    fn tiny_images_are_upscaled_to_the_pixel_minimum() {
        let tiny = encode(
            DynamicImage::ImageRgb8(RgbImage::from_pixel(20, 20, Rgb([200, 0, 0]))),
            ImageFormat::Jpeg,
        );
        let fitted = fit_image(&tiny).unwrap();
        assert!(fitted.changed);
        assert_eq!((fitted.width, fitted.height), (40, 40));
        assert_eq!(fitted.mime_type, "image/png");
        let decoded = image::load_from_memory(&fitted.bytes).unwrap();
        assert_eq!(decoded.dimensions(), (40, 40));

        let strip = encode(
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 10, Rgba([0, 0, 0, 0]))),
            ImageFormat::Png,
        );
        let fitted = fit_image(&strip).unwrap();
        assert_eq!((fitted.width, fitted.height), (8, 80));
        assert!(u64::from(fitted.width * fitted.height) >= MIN_PIXELS);
    }

    #[test]
    fn untrusted_png_with_a_bad_chunk_is_reencoded_clean() {
        let bad = png_with_bad_ancillary_crc();
        assert!(
            !fit_image(&bad).unwrap().changed,
            "fit_image alone passes the defect through"
        );
        let clean = fit_untrusted_image(&bad).unwrap();
        assert!(clean.changed);
        assert_eq!(clean.mime_type, "image/png");
        assert_eq!((clean.width, clean.height), (32, 32));
        assert!(!clean.bytes.windows(4).any(|w| w == b"tEXt"));
        let decoded = image::load_from_memory(&clean.bytes).unwrap();
        assert!(decoded.color().has_alpha());
    }

    #[test]
    fn untrusted_jpeg_passes_through_and_garbage_errors() {
        let jpeg = encode(
            DynamicImage::ImageRgb8(RgbImage::from_pixel(32, 32, Rgb([9, 9, 9]))),
            ImageFormat::Jpeg,
        );
        let fitted = fit_untrusted_image(&jpeg).unwrap();
        assert!(!fitted.changed);
        assert_eq!(fitted.bytes, jpeg);
        assert!(fit_untrusted_image(b"not an image").is_err());
    }

    /// Deterministic noise: incompressible enough to exercise the ladder.
    fn noise(width: u32, height: u32) -> RgbImage {
        let mut state: u32 = 0x1234_5678;
        RgbImage::from_fn(width, height, |_, _| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let [a, b, c, _] = state.to_le_bytes();
            Rgb([a, b, c])
        })
    }

    #[test]
    fn small_supported_images_pass_through_unchanged() {
        let png = encode(
            DynamicImage::ImageRgb8(RgbImage::new(64, 32)),
            ImageFormat::Png,
        );
        let fitted = fit_image(&png).unwrap();
        assert!(!fitted.changed);
        assert_eq!(fitted.bytes, png);
        assert_eq!((fitted.width, fitted.height), (64, 32));
        assert_eq!(fitted.mime_type, "image/png");
    }

    #[test]
    fn oversized_dimensions_scale_to_the_max_side() {
        let png = encode(
            DynamicImage::ImageRgb8(RgbImage::new(2400, 600)),
            ImageFormat::Png,
        );
        let fitted = fit_image(&png).unwrap();
        assert!(fitted.changed);
        assert_eq!((fitted.width, fitted.height), (2000, 500));
    }

    #[test]
    fn transparent_images_stay_png_when_they_fit() {
        let rgba = RgbaImage::from_pixel(2400, 2400, Rgba([10, 20, 30, 128]));
        let png = encode(DynamicImage::ImageRgba8(rgba), ImageFormat::Png);
        let fitted = fit_image(&png).unwrap();
        assert_eq!(fitted.mime_type, "image/png");
        assert_eq!(fitted.width, 2000);
    }

    #[test]
    fn heavy_images_fall_back_to_jpeg_under_the_budget() {
        // 1200x1200 noise PNG is ~4.3 MB: over the budget at the max side.
        let png = encode(DynamicImage::ImageRgb8(noise(1200, 1200)), ImageFormat::Png);
        assert!(png.len() > MAX_ENCODED_BYTES);
        let fitted = fit_image(&png).unwrap();
        assert!(fitted.changed);
        assert_eq!(fitted.mime_type, "image/jpeg");
        assert!(fitted.bytes.len() <= MAX_ENCODED_BYTES);
    }

    #[test]
    fn bmp_is_converted_because_providers_reject_it() {
        let bmp = encode(
            DynamicImage::ImageRgb8(RgbImage::new(32, 32)),
            ImageFormat::Bmp,
        );
        let fitted = fit_image(&bmp).unwrap();
        assert!(fitted.changed);
        assert_eq!(fitted.mime_type, "image/jpeg");
    }

    #[test]
    fn garbage_is_an_error() {
        assert!(fit_image(b"not an image").is_err());
    }
}
