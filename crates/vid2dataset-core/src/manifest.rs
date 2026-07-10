use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{Crop, CropSpace, ImageFormat, ManifestFormat, Resize, Sampling, VideoJob};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ManifestRecord {
    pub source_path: String,
    pub source_video_id: String,
    pub output_path: String,
    pub sequence_number: u32,
    pub frame_index: Option<u64>,
    pub timestamp_seconds: Option<f64>,
    pub sampling: Sampling,
    pub crop: Option<Crop>,
    pub crop_space: CropSpace,
    pub resize: Option<Resize>,
    pub color_space: Option<String>,
    pub image_format: ImageFormat,
    pub file_size: u64,
    pub sha256: String,
}

pub fn build_manifest_records(job: &VideoJob, outputs: &[PathBuf]) -> Result<Vec<ManifestRecord>> {
    outputs
        .iter()
        .enumerate()
        .map(|(idx, output)| {
            let metadata = fs::metadata(output)
                .with_context(|| format!("failed to stat output {}", output.display()))?;
            let sequence = idx as u32;
            Ok(ManifestRecord {
                source_path: job.input.display().to_string(),
                source_video_id: source_video_id(&job.input),
                output_path: output.display().to_string(),
                sequence_number: job.start_number + sequence,
                frame_index: Some((job.sampling.every_n_frames as u64) * (sequence as u64)),
                timestamp_seconds: None,
                sampling: job.sampling.clone(),
                crop: job.transforms.crop,
                crop_space: job.transforms.crop_space.clone(),
                resize: job.transforms.resize,
                color_space: job
                    .transforms
                    .color_space
                    .as_ref()
                    .map(|value| format!("{value:?}").to_lowercase()),
                image_format: job.format.clone(),
                file_size: metadata.len(),
                sha256: sha256_file(output)?,
            })
        })
        .collect()
}

pub fn write_manifests(job: &VideoJob, records: &[ManifestRecord]) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    for format in &job.manifests {
        match format {
            ManifestFormat::Jsonl => {
                let path = job.output_dir.join("manifest.jsonl");
                write_jsonl(&path, records)?;
                written.push(path);
            }
            ManifestFormat::Csv => {
                let path = job.output_dir.join("manifest.csv");
                write_csv(&path, records)?;
                written.push(path);
            }
        }
    }
    Ok(written)
}

fn write_jsonl(path: &Path, records: &[ManifestRecord]) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_csv(path: &Path, records: &[ManifestRecord]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("failed to create CSV {}", path.display()))?;
    writer.write_record([
        "source_path",
        "source_video_id",
        "output_path",
        "sequence_number",
        "frame_index",
        "timestamp_seconds",
        "every_n_frames",
        "crop",
        "crop_space",
        "resize",
        "color_space",
        "image_format",
        "file_size",
        "sha256",
    ])?;
    for record in records {
        writer.write_record([
            record.source_path.clone(),
            record.source_video_id.clone(),
            record.output_path.clone(),
            record.sequence_number.to_string(),
            record
                .frame_index
                .map(|value| value.to_string())
                .unwrap_or_default(),
            record
                .timestamp_seconds
                .map(|value| value.to_string())
                .unwrap_or_default(),
            record.sampling.every_n_frames.to_string(),
            record
                .crop
                .map(|crop| format!("{},{},{},{}", crop.x, crop.y, crop.width, crop.height))
                .unwrap_or_default(),
            format!("{:?}", record.crop_space).to_lowercase(),
            record
                .resize
                .map(|resize| format!("{}x{}", resize.width, resize.height))
                .unwrap_or_default(),
            record.color_space.clone().unwrap_or_default(),
            format!("{:?}", record.image_format).to_lowercase(),
            record.file_size.to_string(),
            record.sha256.clone(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{digest:x}"))
}

fn source_video_id(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("video");
    stem.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}
