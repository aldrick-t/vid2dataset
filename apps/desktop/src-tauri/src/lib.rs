use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use tauri::Manager;
use vid2dataset_core::{
    discover_ffmpeg_path, doctor, extract_video, import_legacy_config, inspect_video,
    ExtractionReport, FfmpegDiscovery, VideoInfo, VideoJob,
};

#[derive(serde::Deserialize, serde::Serialize)]
struct PreviewFrame {
    path: PathBuf,
    timestamp_seconds: f64,
}

#[tauri::command]
fn doctor_command(ffmpeg: Option<PathBuf>, ffprobe: Option<PathBuf>) -> FfmpegDiscovery {
    doctor(ffmpeg, ffprobe)
}

#[tauri::command]
fn inspect_video_command(path: PathBuf, ffprobe: Option<PathBuf>) -> Result<VideoInfo, String> {
    inspect_video(&path, ffprobe).map_err(|error| error.to_string())
}

#[tauri::command]
fn allow_asset_path_command(app: tauri::AppHandle, path: PathBuf) -> Result<(), String> {
    app.asset_protocol_scope()
        .allow_file(path)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn extract_video_command(job: VideoJob) -> Result<ExtractionReport, String> {
    extract_video(&job).map_err(|error| error.to_string())
}

#[tauri::command]
fn preview_frame_command(
    path: PathBuf,
    timestamp_seconds: Option<f64>,
    ffmpeg: Option<PathBuf>,
) -> Result<PreviewFrame, String> {
    let ffmpeg_path = discover_ffmpeg_path(ffmpeg)
        .ok_or_else(|| "ffmpeg not found; run doctor or stage bundled binaries".to_string())?;
    let timestamp_seconds = timestamp_seconds.unwrap_or(0.0).max(0.0);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis();
    let output = std::env::temp_dir().join(format!("vid2dataset-preview-{stamp}.jpg"));
    let status = Command::new(ffmpeg_path)
        .arg("-y")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-ss")
        .arg(format!("{timestamp_seconds:.3}"))
        .arg("-i")
        .arg(&path)
        .arg("-frames:v")
        .arg("1")
        .arg("-q:v")
        .arg("3")
        .arg(&output)
        .status()
        .map_err(|error| format!("failed to run ffmpeg preview extraction: {error}"))?;

    if !status.success() {
        return Err(format!(
            "ffmpeg preview extraction failed for {}",
            path.display()
        ));
    }

    Ok(PreviewFrame {
        path: output,
        timestamp_seconds,
    })
}

#[tauri::command]
fn import_legacy_config_command(
    path: PathBuf,
    output: Option<PathBuf>,
) -> Result<VideoJob, String> {
    import_legacy_config(&path, output.as_deref()).map_err(|error| error.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            doctor_command,
            allow_asset_path_command,
            inspect_video_command,
            extract_video_command,
            preview_frame_command,
            import_legacy_config_command
        ])
        .run(tauri::generate_context!())
        .expect("error while running vid2dataset desktop app");
}
