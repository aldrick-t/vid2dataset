use std::fs;
use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::VideoJob;
use crate::ffmpeg::{discover_ffmpeg_path, extraction_command};
use crate::manifest::{build_manifest_records, write_manifests};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExtractionReport {
    pub input: PathBuf,
    pub output_dir: PathBuf,
    pub frames_written: usize,
    pub manifests_written: Vec<PathBuf>,
}

pub fn extract_video(job: &VideoJob) -> Result<ExtractionReport> {
    job.validate()?;
    fs::create_dir_all(&job.output_dir)
        .with_context(|| format!("failed to create {}", job.output_dir.display()))?;

    let ffmpeg = discover_ffmpeg_path(job.runtime.ffmpeg_path.clone())
        .ok_or_else(|| anyhow!("ffmpeg not found; set VID2DATASET_FFMPEG or configure a path"))?;
    let mut command = extraction_command(job, ffmpeg);
    let output = command
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .output()
        .with_context(|| format!("failed to run FFmpeg for {}", job.input.display()))?;

    if !output.status.success() {
        return Err(anyhow!(
            "FFmpeg failed for {}: {}",
            job.input.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let outputs = collect_outputs(job)?;
    let records = build_manifest_records(job, &outputs)?;
    let manifests_written = write_manifests(job, &records)?;
    Ok(ExtractionReport {
        input: job.input.clone(),
        output_dir: job.output_dir.clone(),
        frames_written: outputs.len(),
        manifests_written,
    })
}

fn collect_outputs(job: &VideoJob) -> Result<Vec<PathBuf>> {
    let mut outputs = Vec::new();
    let ext = job.format.extension();
    for entry in fs::read_dir(&job.output_dir)
        .with_context(|| format!("failed to read {}", job.output_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with(&format!("{}_", job.prefix)))
                .unwrap_or(false)
            && path.extension().and_then(|value| value.to_str()) == Some(ext)
        {
            outputs.push(path);
        }
    }
    outputs.sort();
    Ok(outputs)
}
