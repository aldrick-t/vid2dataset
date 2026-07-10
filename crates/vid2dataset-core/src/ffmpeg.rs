use std::env;
use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::config::{ColorSpace, CropSpace, VideoJob};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolStatus {
    pub name: String,
    pub path: Option<PathBuf>,
    pub available: bool,
    pub version: Option<String>,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FfmpegDiscovery {
    pub ffmpeg: ToolStatus,
    pub ffprobe: ToolStatus,
    pub remediation: Vec<String>,
}

pub fn doctor(
    runtime_ffmpeg: Option<PathBuf>,
    runtime_ffprobe: Option<PathBuf>,
) -> FfmpegDiscovery {
    let ffmpeg = discover_tool("ffmpeg", "VID2DATASET_FFMPEG", runtime_ffmpeg);
    let ffprobe = discover_tool("ffprobe", "VID2DATASET_FFPROBE", runtime_ffprobe);
    let remediation = remediation_for(&ffmpeg, &ffprobe);
    FfmpegDiscovery {
        ffmpeg,
        ffprobe,
        remediation,
    }
}

pub fn remediation_for(ffmpeg: &ToolStatus, ffprobe: &ToolStatus) -> Vec<String> {
    if ffmpeg.available && ffprobe.available {
        return vec!["FFmpeg toolchain is ready.".to_string()];
    }

    let mut hints = Vec::new();
    if !ffmpeg.available {
        hints.push("ffmpeg was not found or did not run with -version.".to_string());
    }
    if !ffprobe.available {
        hints.push("ffprobe was not found or did not run with -version.".to_string());
    }
    hints.push("Install FFmpeg system-wide, set VID2DATASET_FFMPEG and VID2DATASET_FFPROBE, pass --ffmpeg/--ffprobe, or run scripts/setup-ffmpeg.sh to stage binaries for desktop bundles.".to_string());
    hints.push(
        "Bundled means binaries staged under apps/desktop/src-tauri/ffmpeg/bin; binaries are intentionally not committed."
            .to_string(),
    );
    hints
}

pub fn expected_bundled_tool_path(name: &str) -> Option<PathBuf> {
    bundled_tool_paths(name).into_iter().next()
}

pub fn discover_ffmpeg_path(override_path: Option<PathBuf>) -> Option<PathBuf> {
    discover_tool("ffmpeg", "VID2DATASET_FFMPEG", override_path).path
}

pub fn discover_ffprobe_path(override_path: Option<PathBuf>) -> Option<PathBuf> {
    discover_tool("ffprobe", "VID2DATASET_FFPROBE", override_path).path
}

pub fn build_filtergraph(job: &VideoJob) -> String {
    let mut filters = vec![format!(
        "select='not(mod(n\\,{}))'",
        job.sampling.every_n_frames
    )];

    match job.transforms.crop_space {
        CropSpace::Source => {
            push_crop(&mut filters, job);
            push_scale(&mut filters, job);
        }
        CropSpace::Output => {
            push_scale(&mut filters, job);
            push_crop(&mut filters, job);
        }
    }

    push_color(&mut filters, job);
    filters.push("setpts=N/FRAME_RATE/TB".to_string());
    filters.join(",")
}

pub fn extraction_command(job: &VideoJob, ffmpeg_path: PathBuf) -> Command {
    let mut command = Command::new(ffmpeg_path);
    if job.runtime.overwrite {
        command.arg("-y");
    } else {
        command.arg("-n");
    }
    command
        .arg("-hide_banner")
        .arg("-i")
        .arg(&job.input)
        .arg("-vf")
        .arg(build_filtergraph(job))
        .arg("-vsync")
        .arg("0")
        .arg("-start_number")
        .arg(job.start_number.to_string());

    if let Some(limit) = job.sampling.limit {
        command.arg("-frames:v").arg(limit.to_string());
    }

    command.arg(job.output_pattern());
    command
}

fn discover_tool(name: &str, env_var: &str, override_path: Option<PathBuf>) -> ToolStatus {
    if let Some(path) = override_path {
        return status_for_path(name, path, "config_override");
    }
    if let Ok(raw) = env::var(env_var) {
        if !raw.trim().is_empty() {
            return status_for_path(name, PathBuf::from(raw), "environment");
        }
    }
    for path in bundled_tool_paths(name) {
        if path.exists() {
            return status_for_path(name, path, "bundled");
        }
    }
    if let Ok(path) = which::which(name) {
        return status_for_path(name, path, "path");
    }
    ToolStatus {
        name: name.to_string(),
        path: None,
        available: false,
        version: None,
        source: "not_found".to_string(),
    }
}

fn status_for_path(name: &str, path: PathBuf, source: &str) -> ToolStatus {
    let version = Command::new(&path)
        .arg("-version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .and_then(|stdout| stdout.lines().next().map(|line| line.to_string()));

    ToolStatus {
        name: name.to_string(),
        path: Some(path),
        available: version.is_some(),
        version,
        source: source.to_string(),
    }
}

fn bundled_tool_path(name: &str) -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let suffix = if cfg!(windows) { ".exe" } else { "" };
    Some(
        exe.parent()?
            .join("ffmpeg")
            .join("bin")
            .join(format!("{name}{suffix}")),
    )
}

fn bundled_tool_paths(name: &str) -> Vec<PathBuf> {
    let suffix = if cfg!(windows) { ".exe" } else { "" };
    let mut paths = Vec::new();
    if let Some(path) = bundled_tool_path(name) {
        paths.push(path);
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(macos_dir) = exe.parent() {
            if let Some(contents_dir) = macos_dir.parent() {
                paths.push(
                    contents_dir
                        .join("Resources")
                        .join("ffmpeg")
                        .join("bin")
                        .join(format!("{name}{suffix}")),
                );
            }
        }
    }
    paths.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("apps")
            .join("desktop")
            .join("src-tauri")
            .join("ffmpeg")
            .join("bin")
            .join(format!("{name}{suffix}")),
    );
    paths
}

fn push_crop(filters: &mut Vec<String>, job: &VideoJob) {
    if let Some(crop) = job.transforms.crop {
        filters.push(format!(
            "crop={}:{}:{}:{}",
            crop.width, crop.height, crop.x, crop.y
        ));
    }
}

fn push_scale(filters: &mut Vec<String>, job: &VideoJob) {
    if let Some(resize) = job.transforms.resize {
        filters.push(format!("scale={}:{}", resize.width, resize.height));
    }
}

fn push_color(filters: &mut Vec<String>, job: &VideoJob) {
    match job.transforms.color_space {
        Some(ColorSpace::Gray) => filters.push("format=gray".to_string()),
        Some(ColorSpace::Rgb) => filters.push("format=rgb24".to_string()),
        Some(ColorSpace::Bgr) => filters.push("format=bgr24".to_string()),
        Some(ColorSpace::Hsv) => filters.push("format=hsv24".to_string()),
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{
        Crop, CropSpace, ImageFormat, ManifestFormat, Resize, RuntimeConfig, Sampling,
        TransformConfig, VideoJob,
    };

    use super::*;

    fn job(crop_space: CropSpace) -> VideoJob {
        VideoJob {
            input: "input.mp4".into(),
            output_dir: "out".into(),
            sampling: Sampling {
                every_n_frames: 8,
                limit: None,
            },
            transforms: TransformConfig {
                crop: Some(Crop {
                    x: 10,
                    y: 20,
                    width: 300,
                    height: 200,
                }),
                crop_space,
                resize: Some(Resize {
                    width: 640,
                    height: 480,
                }),
                color_space: None,
            },
            format: ImageFormat::Png,
            prefix: "frame".to_string(),
            start_number: 0,
            manifests: vec![ManifestFormat::Jsonl],
            runtime: RuntimeConfig::default(),
        }
    }

    #[test]
    fn source_crop_happens_before_scale() {
        let filter = build_filtergraph(&job(CropSpace::Source));
        assert!(filter.find("crop=300:200:10:20").unwrap() < filter.find("scale=640:480").unwrap());
    }

    #[test]
    fn output_crop_happens_after_scale() {
        let filter = build_filtergraph(&job(CropSpace::Output));
        assert!(filter.find("scale=640:480").unwrap() < filter.find("crop=300:200:10:20").unwrap());
    }
}
