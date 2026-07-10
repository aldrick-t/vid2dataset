use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use vid2dataset_core::{
    doctor, extract_video, import_legacy_config, inspect_video, ColorSpace, Crop, CropSpace,
    ImageFormat, ManifestFormat, Resize, RuntimeConfig, Sampling, TransformConfig, VideoJob,
};
use walkdir::WalkDir;

#[derive(Debug, Parser)]
#[command(name = "vid2dataset")]
#[command(about = "Create image datasets from videos using FFmpeg-backed extraction")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Extract(ExtractArgs),
    Inspect(InspectArgs),
    Doctor(DoctorArgs),
    Profiles {
        #[command(subcommand)]
        command: ProfileCommands,
    },
    Presets {
        #[command(subcommand)]
        command: PresetCommands,
    },
}

#[derive(Clone, Debug, Parser)]
struct ExtractArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1)]
    every_n_frames: u32,
    #[arg(long)]
    limit: Option<u32>,
    #[arg(long)]
    crop: Option<String>,
    #[arg(long, value_enum, default_value = "source")]
    crop_space: CropSpaceArg,
    #[arg(long)]
    resize: Option<String>,
    #[arg(long, value_enum)]
    color_space: Option<ColorSpaceArg>,
    #[arg(long, value_enum, default_value = "png")]
    format: ImageFormatArg,
    #[arg(long, default_value = "frame")]
    prefix: String,
    #[arg(long, default_value_t = 0)]
    start_number: u32,
    #[arg(long, value_delimiter = ',', default_value = "jsonl,csv")]
    manifest: Vec<ManifestFormatArg>,
    #[arg(long)]
    ffmpeg: Option<PathBuf>,
    #[arg(long)]
    ffprobe: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    workers: usize,
    #[arg(long)]
    overwrite: bool,
    #[arg(long)]
    recursive: bool,
}

#[derive(Debug, Parser)]
struct InspectArgs {
    video: PathBuf,
    #[arg(long)]
    ffprobe: Option<PathBuf>,
}

#[derive(Debug, Parser)]
struct DoctorArgs {
    #[arg(long)]
    ffmpeg: Option<PathBuf>,
    #[arg(long)]
    ffprobe: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum ProfileCommands {
    Import {
        legacy_yaml: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum PresetCommands {
    List,
}

#[derive(Clone, Debug, ValueEnum)]
enum CropSpaceArg {
    Source,
    Output,
}

#[derive(Clone, Debug, ValueEnum)]
enum ColorSpaceArg {
    Bgr,
    Rgb,
    Gray,
    Hsv,
}

#[derive(Clone, Debug, ValueEnum)]
enum ImageFormatArg {
    Png,
    Jpeg,
    Webp,
}

#[derive(Clone, Debug, ValueEnum)]
enum ManifestFormatArg {
    Jsonl,
    Csv,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Extract(args) => run_extract(args),
        Commands::Inspect(args) => {
            let info = inspect_video(&args.video, args.ffprobe)?;
            println!("{}", serde_json::to_string_pretty(&info)?);
            Ok(())
        }
        Commands::Doctor(args) => {
            let report = doctor(args.ffmpeg, args.ffprobe);
            println!("{}", serde_json::to_string_pretty(&report)?);
            if report.ffmpeg.available && report.ffprobe.available {
                Ok(())
            } else {
                Err(anyhow!("FFmpeg toolchain is incomplete"))
            }
        }
        Commands::Profiles { command } => match command {
            ProfileCommands::Import {
                legacy_yaml,
                output,
            } => {
                let job = import_legacy_config(&legacy_yaml, output.as_deref())?;
                println!("{}", serde_json::to_string_pretty(&job)?);
                Ok(())
            }
        },
        Commands::Presets { command } => match command {
            PresetCommands::List => {
                println!("{}", serde_json::to_string_pretty(&default_presets())?);
                Ok(())
            }
        },
    }
}

fn run_extract(args: ExtractArgs) -> Result<()> {
    let inputs = collect_inputs(&args.input, args.recursive)?;
    if inputs.is_empty() {
        return Err(anyhow!("no video inputs found at {}", args.input.display()));
    }

    let input_count = inputs.len();
    let worker_count = args.workers.max(1).min(input_count);
    let args = Arc::new(args);
    let queue = Arc::new(Mutex::new(VecDeque::from(inputs)));
    let total_frames = thread::scope(|scope| -> Result<usize> {
        let mut handles = Vec::new();
        for _ in 0..worker_count {
            let args = Arc::clone(&args);
            let queue = Arc::clone(&queue);
            handles.push(scope.spawn(move || -> Result<usize> {
                let mut worker_frames = 0usize;
                loop {
                    let input = {
                        let mut guard = queue.lock().expect("extract queue poisoned");
                        guard.pop_front()
                    };
                    let Some(input) = input else {
                        break;
                    };
                    let output_dir = output_dir_for(&args.output, &input, input_count);
                    let job = build_job(&args, input, output_dir)?;
                    let report = extract_video(&job)?;
                    worker_frames += report.frames_written;
                    eprintln!(
                        "processed {} -> {} frames",
                        report.input.display(),
                        report.frames_written
                    );
                }
                Ok(worker_frames)
            }));
        }

        let mut total = 0usize;
        for handle in handles {
            total += handle
                .join()
                .map_err(|_| anyhow!("extract worker panicked"))??;
        }
        Ok(total)
    })?;
    eprintln!("done: {input_count} video(s), {total_frames} frame(s)");
    Ok(())
}

fn output_dir_for(base_output: &Path, input: &Path, input_count: usize) -> PathBuf {
    if input_count == 1 {
        return base_output.to_path_buf();
    }
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("video");
    base_output.join(sanitize(stem))
}

fn build_job(args: &ExtractArgs, input: PathBuf, output_dir: PathBuf) -> Result<VideoJob> {
    Ok(VideoJob {
        input,
        output_dir,
        sampling: Sampling {
            every_n_frames: args.every_n_frames,
            limit: args.limit,
        },
        transforms: TransformConfig {
            crop: args.crop.as_deref().map(parse_crop).transpose()?,
            crop_space: match args.crop_space {
                CropSpaceArg::Source => CropSpace::Source,
                CropSpaceArg::Output => CropSpace::Output,
            },
            resize: args.resize.as_deref().map(parse_resize).transpose()?,
            color_space: args.color_space.as_ref().map(|value| match value {
                ColorSpaceArg::Bgr => ColorSpace::Bgr,
                ColorSpaceArg::Rgb => ColorSpace::Rgb,
                ColorSpaceArg::Gray => ColorSpace::Gray,
                ColorSpaceArg::Hsv => ColorSpace::Hsv,
            }),
        },
        format: match args.format {
            ImageFormatArg::Png => ImageFormat::Png,
            ImageFormatArg::Jpeg => ImageFormat::Jpeg,
            ImageFormatArg::Webp => ImageFormat::Webp,
        },
        prefix: args.prefix.clone(),
        start_number: args.start_number,
        manifests: args
            .manifest
            .iter()
            .map(|value| match value {
                ManifestFormatArg::Jsonl => ManifestFormat::Jsonl,
                ManifestFormatArg::Csv => ManifestFormat::Csv,
            })
            .collect(),
        runtime: RuntimeConfig {
            ffmpeg_path: args.ffmpeg.clone(),
            ffprobe_path: args.ffprobe.clone(),
            workers: args.workers.max(1),
            overwrite: args.overwrite,
        },
    })
}

fn collect_inputs(input: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if !input.is_dir() {
        return Err(anyhow!("input does not exist: {}", input.display()));
    }

    let mut videos = Vec::new();
    if recursive {
        for entry in WalkDir::new(input)
            .into_iter()
            .filter_map(|entry| entry.ok())
        {
            if entry.file_type().is_file() && is_video(entry.path()) {
                videos.push(entry.path().to_path_buf());
            }
        }
    } else {
        for entry in
            fs::read_dir(input).with_context(|| format!("failed to read {}", input.display()))?
        {
            let path = entry?.path();
            if path.is_file() && is_video(&path) {
                videos.push(path);
            }
        }
    }
    videos.sort();
    Ok(videos)
}

fn is_video(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("mp4" | "mov" | "m4v" | "mkv" | "avi" | "webm")
    )
}

fn parse_crop(raw: &str) -> Result<Crop> {
    let parts = parse_u32_list(raw, 4)?;
    Ok(Crop {
        x: parts[0],
        y: parts[1],
        width: parts[2],
        height: parts[3],
    })
}

fn parse_resize(raw: &str) -> Result<Resize> {
    let normalized = raw.replace('x', ",");
    let parts = parse_u32_list(&normalized, 2)?;
    Ok(Resize {
        width: parts[0],
        height: parts[1],
    })
}

fn parse_u32_list(raw: &str, expected: usize) -> Result<Vec<u32>> {
    let parts: Result<Vec<u32>, _> = raw
        .split(',')
        .map(|value| value.trim().parse::<u32>())
        .collect();
    let parts = parts.with_context(|| format!("invalid numeric list: {raw}"))?;
    if parts.len() != expected {
        return Err(anyhow!("expected {expected} values, got {}", parts.len()));
    }
    Ok(parts)
}

fn sanitize(raw: &str) -> String {
    raw.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn default_presets() -> Vec<VideoJob> {
    vec![VideoJob {
        input: "input/video.mp4".into(),
        output_dir: "output".into(),
        sampling: Sampling {
            every_n_frames: 30,
            limit: None,
        },
        transforms: TransformConfig {
            resize: Some(Resize {
                width: 1280,
                height: 720,
            }),
            ..TransformConfig::default()
        },
        format: ImageFormat::Png,
        prefix: "frame".to_string(),
        start_number: 0,
        manifests: vec![ManifestFormat::Jsonl, ManifestFormat::Csv],
        runtime: RuntimeConfig::default(),
    }]
}
