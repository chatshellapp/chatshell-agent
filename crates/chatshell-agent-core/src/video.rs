//! Video attachments through an optional ffmpeg.
//!
//! The model never receives video bytes. Reading a video returns ffprobe
//! metadata plus a contact sheet: six frames spread over the duration,
//! tiled 3x2 at 320 px wide, as one PNG. A timestamp selects a single
//! full-size frame instead. Without ffmpeg/ffprobe the read says how to
//! enable it. Hosts that cannot spawn processes (iOS) never reach this.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Duration;

use image::{DynamicImage, GenericImage, GenericImageView, ImageFormat, Rgba, RgbaImage};

pub const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "m4v", "webm", "mkv", "avi"];
/// Frames on a contact sheet.
const SHEET_FRAMES: usize = 6;
const SHEET_COLUMNS: u32 = 3;
const TILE_WIDTH: u32 = 320;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);

pub fn is_video_extension(ext: &str) -> bool {
    VIDEO_EXTENSIONS.contains(&ext)
}

/// ffprobe facts shown to the model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VideoInfo {
    pub duration_secs: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f64>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub container: Option<String>,
}

impl VideoInfo {
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(container) = &self.container {
            parts.push(container.clone());
        }
        if let Some(duration) = self.duration_secs {
            parts.push(format_timestamp(duration));
        }
        if let (Some(w), Some(h)) = (self.width, self.height) {
            parts.push(format!("{w}x{h}"));
        }
        if let Some(fps) = self.fps {
            parts.push(format!("{fps:.2} fps"));
        }
        if let Some(codec) = &self.video_codec {
            parts.push(format!("video {codec}"));
        }
        match &self.audio_codec {
            Some(codec) => parts.push(format!("audio {codec}")),
            None => parts.push("no audio".to_string()),
        }
        parts.join(", ")
    }
}

/// Locate a tool binary: `PATH`, then the usual package-manager prefixes
/// (GUI apps on macOS start with a minimal `PATH`).
pub fn find_tool(name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    let path_dirs = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
        .unwrap_or_default();
    let extra = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/opt/local/bin",
    ]
    .iter()
    .map(PathBuf::from);
    path_dirs
        .into_iter()
        .chain(extra)
        .map(|dir| dir.join(&exe))
        .find(|candidate| candidate.is_file())
}

/// The model-facing note when ffmpeg is missing.
pub fn missing_tools_message() -> String {
    "Video files need ffmpeg and ffprobe on this computer, which were not found. \
     Ask the user to install ffmpeg (macOS: `brew install ffmpeg`; Windows: \
     `winget install ffmpeg`; Linux: the distribution's ffmpeg package) to let \
     you look at videos."
        .to_string()
}

/// Parse `"75"`, `"75.5"`, `"1:15"`, or `"1:02:03"` into seconds.
pub fn parse_timestamp(text: &str) -> Option<f64> {
    let parts: Vec<&str> = text.trim().split(':').collect();
    if parts.is_empty() || parts.len() > 3 {
        return None;
    }
    let mut seconds = 0f64;
    for part in &parts {
        let value: f64 = part.trim().parse().ok()?;
        if value < 0.0 {
            return None;
        }
        seconds = seconds * 60.0 + value;
    }
    Some(seconds)
}

pub fn format_timestamp(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// Evenly spread frame times: the middle of each of `count` segments.
pub fn sheet_times(duration: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| duration * (i as f64 + 0.5) / count as f64)
        .collect()
}

/// Parse `ffprobe -show_format -show_streams -of json` output.
pub fn parse_probe(json: &str) -> VideoInfo {
    let value: serde_json::Value = serde_json::from_str(json).unwrap_or_default();
    let streams = value["streams"].as_array().cloned().unwrap_or_default();
    let video = streams.iter().find(|s| s["codec_type"] == "video");
    let audio = streams.iter().find(|s| s["codec_type"] == "audio");
    let fps = video
        .and_then(|v| v["avg_frame_rate"].as_str())
        .and_then(|rate| {
            let (n, d) = rate.split_once('/')?;
            let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
            (d > 0.0 && n > 0.0).then_some(n / d)
        });
    VideoInfo {
        duration_secs: value["format"]["duration"]
            .as_str()
            .and_then(|d| d.parse().ok()),
        width: video.and_then(|v| v["width"].as_u64()).map(|w| w as u32),
        height: video.and_then(|v| v["height"].as_u64()).map(|h| h as u32),
        fps,
        video_codec: video
            .and_then(|v| v["codec_name"].as_str())
            .map(str::to_string),
        audio_codec: audio
            .and_then(|a| a["codec_name"].as_str())
            .map(str::to_string),
        container: value["format"]["format_long_name"]
            .as_str()
            .or_else(|| value["format"]["format_name"].as_str())
            .map(str::to_string),
    }
}

/// Tile frames into a grid of `columns`, each scaled to `tile_width`.
pub fn compose_sheet(frames: &[DynamicImage], columns: u32, tile_width: u32) -> Option<Vec<u8>> {
    let first = frames.first()?;
    let (w, h) = first.dimensions();
    let tile_height = ((h as f64) * (tile_width as f64) / (w.max(1) as f64))
        .round()
        .max(1.0) as u32;
    let rows = (frames.len() as u32).div_ceil(columns);
    let gap = 4;
    let mut sheet = RgbaImage::from_pixel(
        columns * tile_width + (columns - 1) * gap,
        rows * tile_height + (rows - 1) * gap,
        Rgba([0, 0, 0, 255]),
    );
    for (index, frame) in frames.iter().enumerate() {
        let tile = frame.resize_exact(
            tile_width,
            tile_height,
            image::imageops::FilterType::Triangle,
        );
        let (col, row) = (index as u32 % columns, index as u32 / columns);
        sheet
            .copy_from(
                &tile.to_rgba8(),
                col * (tile_width + gap),
                row * (tile_height + gap),
            )
            .ok()?;
    }
    let mut out = Vec::new();
    DynamicImage::ImageRgba8(sheet)
        .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .ok()?;
    Some(out)
}

async fn run(program: &Path, args: &[String]) -> Result<Vec<u8>, String> {
    let output = tokio::time::timeout(
        PROCESS_TIMEOUT,
        tokio::process::Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| format!("{} timed out", program.display()))?
    .map_err(|e| format!("{} failed to start: {e}", program.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let last = stderr.lines().last().unwrap_or("").trim().to_string();
        return Err(format!("{} failed: {last}", program.display()));
    }
    Ok(output.stdout)
}

/// ffprobe metadata.
pub async fn probe(ffprobe: &Path, video: &Path) -> Result<VideoInfo, String> {
    let args = [
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
    ]
    .iter()
    .map(|s| s.to_string())
    .chain(std::iter::once(video.display().to_string()))
    .collect::<Vec<_>>();
    let stdout = run(ffprobe, &args).await?;
    Ok(parse_probe(&String::from_utf8_lossy(&stdout)))
}

/// One frame at `seconds` as PNG, scaled to `width` when set.
pub async fn frame_at(
    ffmpeg: &Path,
    video: &Path,
    seconds: f64,
    width: Option<u32>,
) -> Result<Vec<u8>, String> {
    let mut args: Vec<String> = vec![
        "-v".into(),
        "error".into(),
        "-ss".into(),
        format!("{seconds:.3}"),
        "-i".into(),
        video.display().to_string(),
        "-frames:v".into(),
        "1".into(),
    ];
    if let Some(width) = width {
        args.push("-vf".into());
        args.push(format!("scale={width}:-2"));
    }
    args.extend(["-f", "image2pipe", "-vcodec", "png", "-"].map(String::from));
    let png = run(ffmpeg, &args).await?;
    if png.is_empty() {
        return Err(format!("no frame at {}", format_timestamp(seconds)));
    }
    Ok(png)
}

/// The contact sheet PNG for a video of `duration` seconds.
pub async fn contact_sheet(ffmpeg: &Path, video: &Path, duration: f64) -> Result<Vec<u8>, String> {
    let mut frames = Vec::new();
    for time in sheet_times(duration, SHEET_FRAMES) {
        let png = frame_at(ffmpeg, video, time, Some(TILE_WIDTH)).await?;
        frames.push(image::load_from_memory(&png).map_err(|e| e.to_string())?);
    }
    compose_sheet(&frames, SHEET_COLUMNS, TILE_WIDTH)
        .ok_or_else(|| "could not compose frames".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_parse_and_format() {
        assert_eq!(parse_timestamp("75"), Some(75.0));
        assert_eq!(parse_timestamp("1:15"), Some(75.0));
        assert_eq!(parse_timestamp("1:02:03"), Some(3723.0));
        assert_eq!(parse_timestamp("1:2:3:4"), None);
        assert_eq!(parse_timestamp("abc"), None);
        assert_eq!(format_timestamp(75.4), "1:15");
        assert_eq!(format_timestamp(3723.0), "1:02:03");
    }

    #[test]
    fn sheet_frames_sit_in_the_middle_of_equal_segments() {
        assert_eq!(
            sheet_times(60.0, 6),
            vec![5.0, 15.0, 25.0, 35.0, 45.0, 55.0]
        );
    }

    #[test]
    fn probe_output_becomes_a_summary() {
        let json = r#"{
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001"},
                {"codec_type": "audio", "codec_name": "aac"}
            ],
            "format": {"duration": "125.4", "format_long_name": "QuickTime / MOV"}
        }"#;
        let info = parse_probe(json);
        assert_eq!(info.width, Some(1920));
        assert_eq!(
            info.summary(),
            "QuickTime / MOV, 2:05, 1920x1080, 29.97 fps, video h264, audio aac"
        );
    }

    #[test]
    fn frames_tile_into_a_grid() {
        let frames: Vec<DynamicImage> = (0..6)
            .map(|i| {
                DynamicImage::ImageRgba8(RgbaImage::from_pixel(640, 360, Rgba([i * 40, 0, 0, 255])))
            })
            .collect();
        let png = compose_sheet(&frames, 3, 320).unwrap();
        let sheet = image::load_from_memory(&png).unwrap();
        assert_eq!(sheet.dimensions(), (3 * 320 + 2 * 4, 2 * 180 + 4));
    }

    /// Runs only where ffmpeg is installed (generates a clip, then reads it).
    #[tokio::test]
    async fn real_ffmpeg_produces_metadata_and_a_sheet() {
        let (Some(ffmpeg), Some(ffprobe)) = (find_tool("ffmpeg"), find_tool("ffprobe")) else {
            eprintln!("skipping: ffmpeg not installed");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let clip = dir.path().join("clip.mp4");
        let args: Vec<String> = [
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
        ]
        .iter()
        .map(|s| s.to_string())
        .chain(std::iter::once(clip.display().to_string()))
        .collect();
        run(&ffmpeg, &args).await.unwrap();
        let info = probe(&ffprobe, &clip).await.unwrap();
        assert_eq!((info.width, info.height), (Some(320), Some(240)));
        let sheet = contact_sheet(&ffmpeg, &clip, info.duration_secs.unwrap())
            .await
            .unwrap();
        assert!(sheet.starts_with(b"\x89PNG"));
    }
}
