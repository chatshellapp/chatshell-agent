//! Page images of scanned PDFs.
//!
//! A scanned PDF has no text layer: each page is one embedded raster image.
//! Rendering PDFs needs a full rasterizer, but reading a scan does not:
//! the page's image XObjects are the content. This module extracts them
//! (JPEG streams as-is; uncompressed or Flate-compressed 8-bit RGB/gray
//! and 1-bit gray rasters re-encoded as PNG) so the read tool can hand
//! them to a vision model. Formats without a pure-Rust decoder here
//! (JPEG 2000, CCITT fax, JBIG2) are reported as unsupported.

use std::io::Cursor;

use image::{DynamicImage, GrayImage, ImageFormat, RgbImage};

/// Images smaller than this on both sides are decorations, not page scans.
const MIN_SCAN_SIDE: i64 = 200;
/// Decompressed raster ceiling per image.
const MAX_RASTER_BYTES: usize = 128 * 1024 * 1024;

/// One extracted page image.
#[derive(Debug, Clone)]
pub struct PageImage {
    pub bytes: Vec<u8>,
    pub mime_type: &'static str,
    pub width: u32,
    pub height: u32,
}

/// The result for one page.
#[derive(Debug, Clone, Default)]
pub struct PageImages {
    pub images: Vec<PageImage>,
    /// Image encodings present on the page that could not be extracted.
    pub unsupported: Vec<String>,
}

/// Extract the scan-sized images of the 1-based `page`, largest first.
pub fn extract_page_images(bytes: &[u8], page: usize) -> anyhow::Result<PageImages> {
    let mut document = lopdf::Document::load_mem(bytes)?;
    if document.is_encrypted() {
        document
            .decrypt("")
            .map_err(|e| anyhow::anyhow!("encrypted PDF requires a password: {e}"))?;
    }
    let pages = document.get_pages();
    let page_count = pages.len();
    let page_id = *pages.values().nth(page.wrapping_sub(1)).ok_or_else(|| {
        anyhow::anyhow!("page {page} is out of range (document has {page_count} pages)")
    })?;

    let mut result = PageImages::default();
    let mut found: Vec<(i64, PageImage)> = Vec::new();
    for image in document.get_page_images(page_id).unwrap_or_default() {
        if image.width < MIN_SCAN_SIDE && image.height < MIN_SCAN_SIDE {
            continue;
        }
        let filters = image.filters.clone().unwrap_or_default();
        let decoded = match filters.as_slice() {
            [only] if only == "DCTDecode" => Some(PageImage {
                bytes: image.content.to_vec(),
                mime_type: "image/jpeg",
                width: image.width as u32,
                height: image.height as u32,
            }),
            [] | [_] if filters.iter().all(|f| f == "FlateDecode") => {
                let raw = match document
                    .get_object(image.id)
                    .and_then(|object| object.as_stream())
                {
                    Ok(stream) if filters.is_empty() => Some(stream.content.clone()),
                    Ok(stream) => stream
                        .decompressed_content_with_limit(MAX_RASTER_BYTES)
                        .ok(),
                    Err(_) => None,
                };
                raw.and_then(|raw| {
                    raster_to_png(
                        &raw,
                        image.width,
                        image.height,
                        image.color_space.as_deref(),
                        image.bits_per_component,
                    )
                })
            }
            _ => None,
        };
        match decoded {
            Some(decoded) => found.push((image.width * image.height, decoded)),
            None => {
                let label = if filters.is_empty() {
                    format!(
                        "raw {} {}-bit",
                        image.color_space.as_deref().unwrap_or("unknown"),
                        image.bits_per_component.unwrap_or(0)
                    )
                } else {
                    filters.join("+")
                };
                if !result.unsupported.contains(&label) {
                    result.unsupported.push(label);
                }
            }
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));
    result.images = found.into_iter().map(|(_, image)| image).collect();
    Ok(result)
}

/// Encode an unfiltered raster as PNG. Supports 8-bit DeviceRGB and
/// DeviceGray, and 1-bit DeviceGray (bilevel scans).
fn raster_to_png(
    raw: &[u8],
    width: i64,
    height: i64,
    color_space: Option<&str>,
    bits: Option<i64>,
) -> Option<PageImage> {
    let (w, h) = (u32::try_from(width).ok()?, u32::try_from(height).ok()?);
    let pixels = (w as usize).checked_mul(h as usize)?;
    let image = match (color_space?, bits?) {
        ("DeviceRGB", 8) => {
            let data = raw.get(..pixels.checked_mul(3)?)?.to_vec();
            DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, data)?)
        }
        ("DeviceGray", 8) => {
            let data = raw.get(..pixels)?.to_vec();
            DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, data)?)
        }
        ("DeviceGray", 1) => {
            let row_bytes = (w as usize).div_ceil(8);
            let packed = raw.get(..row_bytes.checked_mul(h as usize)?)?;
            let mut data = Vec::with_capacity(pixels);
            for row in packed.chunks(row_bytes) {
                for x in 0..w as usize {
                    let bit = (row[x / 8] >> (7 - (x % 8))) & 1;
                    data.push(if bit == 1 { 255 } else { 0 });
                }
            }
            DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, data)?)
        }
        _ => return None,
    };
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .ok()?;
    Some(PageImage {
        bytes,
        mime_type: "image/png",
        width: w,
        height: h,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use lopdf::{Document, Object, Stream, dictionary};

    /// A PDF whose pages each show one image XObject (no text layer).
    pub(crate) fn scanned_pdf(images: Vec<(Stream, &str)>) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let mut kids = Vec::new();
        for (stream, _) in images {
            let image_id = doc.add_object(stream);
            let content = Stream::new(dictionary! {}, b"q 612 0 0 792 0 0 cm /Im0 Do Q".to_vec());
            let content_id = doc.add_object(content);
            let page_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                "Contents" => content_id,
                "Resources" => dictionary! {
                    "XObject" => dictionary! { "Im0" => image_id },
                },
            });
            kids.push(Object::Reference(page_id));
        }
        let count = kids.len() as i64;
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => kids,
                "Count" => count,
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut out = Vec::new();
        doc.save_to(&mut out).unwrap();
        out
    }

    pub(crate) fn jpeg_image_stream(width: u32, height: u32) -> Stream {
        let mut jpeg = Vec::new();
        DynamicImage::ImageRgb8(RgbImage::from_pixel(
            width,
            height,
            image::Rgb([200, 10, 10]),
        ))
        .write_to(&mut Cursor::new(&mut jpeg), ImageFormat::Jpeg)
        .unwrap();
        Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => width as i64,
                "Height" => height as i64,
                "ColorSpace" => "DeviceRGB",
                "BitsPerComponent" => 8,
                "Filter" => "DCTDecode",
            },
            jpeg,
        )
    }

    fn gray_flate_stream(width: u32, height: u32) -> Stream {
        let mut stream = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => width as i64,
                "Height" => height as i64,
                "ColorSpace" => "DeviceGray",
                "BitsPerComponent" => 8,
            },
            vec![128u8; (width * height) as usize],
        );
        stream.compress().unwrap();
        stream
    }

    #[test]
    fn jpeg_scans_come_out_verbatim() {
        let pdf = scanned_pdf(vec![(jpeg_image_stream(300, 400), "jpeg")]);
        let page = extract_page_images(&pdf, 1).unwrap();
        assert_eq!(page.images.len(), 1);
        assert_eq!(page.images[0].mime_type, "image/jpeg");
        assert!(page.images[0].bytes.starts_with(b"\xff\xd8\xff"));
        assert_eq!((page.images[0].width, page.images[0].height), (300, 400));
    }

    #[test]
    fn flate_gray_rasters_become_png() {
        let pdf = scanned_pdf(vec![
            (jpeg_image_stream(300, 300), "jpeg"),
            (gray_flate_stream(250, 260), "gray"),
        ]);
        let page = extract_page_images(&pdf, 2).unwrap();
        assert_eq!(page.images.len(), 1);
        assert_eq!(page.images[0].mime_type, "image/png");
        assert_eq!((page.images[0].width, page.images[0].height), (250, 260));
        assert!(extract_page_images(&pdf, 3).is_err());
    }

    #[test]
    fn unsupported_encodings_are_named() {
        let stream = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 1700,
                "Height" => 2200,
                "ColorSpace" => "DeviceGray",
                "BitsPerComponent" => 1,
                "Filter" => "CCITTFaxDecode",
            },
            vec![0u8; 16],
        );
        let pdf = scanned_pdf(vec![(stream, "fax")]);
        let page = extract_page_images(&pdf, 1).unwrap();
        assert!(page.images.is_empty());
        assert_eq!(page.unsupported, vec!["CCITTFaxDecode".to_string()]);
    }

    #[test]
    fn bilevel_rows_unpack_msb_first() {
        // 9 px wide -> 2 bytes per row; first pixel white, rest black.
        let raw = [0b1000_0000u8, 0, 0b1000_0000, 0];
        let png = raster_to_png(&raw, 9, 2, Some("DeviceGray"), Some(1)).unwrap();
        let decoded = image::load_from_memory(&png.bytes).unwrap().to_luma8();
        assert_eq!(decoded.get_pixel(0, 0).0, [255]);
        assert_eq!(decoded.get_pixel(1, 0).0, [0]);
        assert_eq!(decoded.dimensions(), (9, 2));
    }
}
