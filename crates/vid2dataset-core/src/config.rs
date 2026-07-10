use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("input path is required")]
    MissingInput,
    #[error("output directory is required")]
    MissingOutput,
    #[error("every_n_frames must be >= 1")]
    InvalidFrameInterval,
    #[error("crop width and height must be > 0")]
    InvalidCrop,
    #[error("resize width and height must be > 0 when present")]
    InvalidResize,
    #[error("filename prefix cannot be empty")]
    InvalidPrefix,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CropSpace {
    Source,
    Output,
}

impl Default for CropSpace {
    fn default() -> Self {
        Self::Source
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    Bgr,
    Rgb,
    Gray,
    Hsv,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
}

impl ImageFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpg",
            Self::Webp => "webp",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestFormat {
    Jsonl,
    Csv,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Crop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Resize {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Sampling {
    pub every_n_frames: u32,
    pub limit: Option<u32>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            every_n_frames: 1,
            limit: None,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct TransformConfig {
    pub crop: Option<Crop>,
    #[serde(default)]
    pub crop_space: CropSpace,
    pub resize: Option<Resize>,
    pub color_space: Option<ColorSpace>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeConfig {
    pub ffmpeg_path: Option<PathBuf>,
    pub ffprobe_path: Option<PathBuf>,
    pub workers: usize,
    pub overwrite: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            ffmpeg_path: None,
            ffprobe_path: None,
            workers: 1,
            overwrite: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VideoJob {
    pub input: PathBuf,
    pub output_dir: PathBuf,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub transforms: TransformConfig,
    pub format: ImageFormat,
    pub prefix: String,
    pub start_number: u32,
    pub manifests: Vec<ManifestFormat>,
    #[serde(default)]
    pub runtime: RuntimeConfig,
}

impl VideoJob {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.input.as_os_str().is_empty() {
            return Err(ConfigError::MissingInput);
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err(ConfigError::MissingOutput);
        }
        if self.sampling.every_n_frames == 0 {
            return Err(ConfigError::InvalidFrameInterval);
        }
        if self.prefix.trim().is_empty() {
            return Err(ConfigError::InvalidPrefix);
        }
        if let Some(crop) = self.transforms.crop {
            if crop.width == 0 || crop.height == 0 {
                return Err(ConfigError::InvalidCrop);
            }
        }
        if let Some(resize) = self.transforms.resize {
            if resize.width == 0 || resize.height == 0 {
                return Err(ConfigError::InvalidResize);
            }
        }
        Ok(())
    }

    pub fn output_pattern(&self) -> PathBuf {
        self.output_dir
            .join(format!("{}_%06d.{}", self.prefix, self.format.extension()))
    }

    pub fn output_filename(&self, sequence: u32) -> String {
        format!(
            "{}_{:06}.{}",
            self.prefix,
            self.start_number + sequence,
            self.format.extension()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_job() -> VideoJob {
        VideoJob {
            input: "in.mp4".into(),
            output_dir: "out".into(),
            sampling: Sampling::default(),
            transforms: TransformConfig::default(),
            format: ImageFormat::Png,
            prefix: "frame".to_string(),
            start_number: 0,
            manifests: vec![ManifestFormat::Jsonl],
            runtime: RuntimeConfig::default(),
        }
    }

    #[test]
    fn validates_positive_sampling() {
        let mut job = base_job();
        job.sampling.every_n_frames = 0;
        assert!(matches!(
            job.validate(),
            Err(ConfigError::InvalidFrameInterval)
        ));
    }

    #[test]
    fn formats_output_filenames() {
        let mut job = base_job();
        job.start_number = 42;
        assert_eq!(job.output_filename(3), "frame_000045.png");
    }
}
