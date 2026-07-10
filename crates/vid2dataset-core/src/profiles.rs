use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::Deserialize;

use crate::config::{
    ColorSpace, Crop, CropSpace, ImageFormat, ManifestFormat, Resize, RuntimeConfig, Sampling,
    TransformConfig, VideoJob,
};

pub fn profile_dir() -> Result<PathBuf> {
    let dirs = ProjectDirs::from("dev", "vid2dataset", "vid2dataset")
        .context("failed to resolve OS app-data directory")?;
    let dir = dirs.config_dir().join("profiles");
    fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    Ok(dir)
}

pub fn import_legacy_config(path: &Path, output_path: Option<&Path>) -> Result<VideoJob> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read legacy config {}", path.display()))?;
    let legacy: LegacyConfig = serde_yaml::from_str(&text)
        .with_context(|| format!("failed to parse legacy config {}", path.display()))?;
    let job = legacy.into_job();

    if let Some(output_path) = output_path {
        let serialized = serde_yaml::to_string(&job)?;
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(output_path, serialized)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }

    Ok(job)
}

#[derive(Debug, Deserialize)]
struct LegacyConfig {
    input_dir: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    interval: Option<u32>,
    prefix: Option<String>,
    start_number: Option<u32>,
    limit: Option<u32>,
    crop_area: Option<Vec<u32>>,
    resize_to: Option<LegacyResize>,
    color_space: Option<String>,
    ext: Option<String>,
    parallel_workers: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct LegacyResize {
    width: Option<u32>,
    height: Option<u32>,
}

impl LegacyConfig {
    fn into_job(self) -> VideoJob {
        VideoJob {
            input: self.input_dir.unwrap_or_else(|| "input".into()),
            output_dir: self.output_dir.unwrap_or_else(|| "output".into()),
            sampling: Sampling {
                every_n_frames: self.interval.unwrap_or(1).max(1),
                limit: self.limit,
            },
            transforms: TransformConfig {
                crop: self.crop_area.and_then(|parts| {
                    if parts.len() == 4 {
                        Some(Crop {
                            x: parts[0],
                            y: parts[1],
                            width: parts[2],
                            height: parts[3],
                        })
                    } else {
                        None
                    }
                }),
                crop_space: CropSpace::Source,
                resize: self
                    .resize_to
                    .and_then(|resize| match (resize.width, resize.height) {
                        (Some(width), Some(height)) => Some(Resize { width, height }),
                        _ => None,
                    }),
                color_space: self.color_space.and_then(parse_color_space),
            },
            format: self
                .ext
                .as_deref()
                .map(parse_format)
                .unwrap_or(ImageFormat::Png),
            prefix: self.prefix.unwrap_or_else(|| "frame".to_string()),
            start_number: self.start_number.unwrap_or(0),
            manifests: vec![ManifestFormat::Jsonl, ManifestFormat::Csv],
            runtime: RuntimeConfig {
                workers: self.parallel_workers.unwrap_or(1).max(1),
                ..RuntimeConfig::default()
            },
        }
    }
}

fn parse_format(raw: &str) -> ImageFormat {
    match raw
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => ImageFormat::Jpeg,
        "webp" => ImageFormat::Webp,
        _ => ImageFormat::Png,
    }
}

fn parse_color_space(raw: String) -> Option<ColorSpace> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bgr" => Some(ColorSpace::Bgr),
        "rgb" => Some(ColorSpace::Rgb),
        "gray" | "grey" => Some(ColorSpace::Gray),
        "hsv" => Some(ColorSpace::Hsv),
        _ => None,
    }
}
