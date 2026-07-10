use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::ffmpeg::discover_ffprobe_path;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct VideoInfo {
    pub path: String,
    pub codec: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f64>,
    pub duration_seconds: Option<f64>,
    pub frame_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ProbeOutput {
    streams: Vec<ProbeStream>,
    format: Option<ProbeFormat>,
}

#[derive(Debug, Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProbeStream {
    codec_name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    nb_frames: Option<String>,
}

pub fn inspect_video(
    path: &Path,
    ffprobe_override: Option<std::path::PathBuf>,
) -> Result<VideoInfo> {
    let ffprobe =
        discover_ffprobe_path(ffprobe_override).ok_or_else(|| anyhow!("ffprobe not found"))?;
    let output = Command::new(ffprobe)
        .arg("-v")
        .arg("error")
        .arg("-print_format")
        .arg("json")
        .arg("-show_format")
        .arg("-show_streams")
        .arg(path)
        .output()
        .with_context(|| format!("failed to run ffprobe for {}", path.display()))?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let probed: ProbeOutput = serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to parse ffprobe JSON for {}", path.display()))?;
    let stream = probed.streams.into_iter().find(|s| s.width.is_some());
    let mut info = VideoInfo {
        path: path.display().to_string(),
        ..VideoInfo::default()
    };

    if let Some(stream) = stream {
        info.codec = stream.codec_name;
        info.width = stream.width;
        info.height = stream.height;
        info.fps = stream.avg_frame_rate.as_deref().and_then(parse_ratio);
        info.frame_count = stream.nb_frames.and_then(|v| v.parse::<u64>().ok());
    }
    info.duration_seconds = probed
        .format
        .and_then(|f| f.duration)
        .and_then(|duration| duration.parse::<f64>().ok());
    Ok(info)
}

fn parse_ratio(raw: &str) -> Option<f64> {
    let (left, right) = raw.split_once('/')?;
    let numerator = left.parse::<f64>().ok()?;
    let denominator = right.parse::<f64>().ok()?;
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::parse_ratio;

    #[test]
    fn parses_frame_rate_ratio() {
        assert_eq!(
            parse_ratio("30000/1001").map(|v| (v * 100.0).round() / 100.0),
            Some(29.97)
        );
        assert_eq!(parse_ratio("0/0"), None);
    }
}
