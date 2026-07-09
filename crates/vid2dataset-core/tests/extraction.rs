use std::fs;
use std::process::Command;

use tempfile::tempdir;
use vid2dataset_core::{
    extract_video, import_legacy_config, ColorSpace, Crop, CropSpace, ImageFormat, ManifestFormat,
    Resize, RuntimeConfig, Sampling, TransformConfig, VideoJob,
};

#[test]
fn extracts_synthetic_video_and_writes_manifests_when_ffmpeg_exists() {
    let Some(ffmpeg) = find_tool("ffmpeg") else {
        eprintln!("skipping FFmpeg integration test: ffmpeg not found");
        return;
    };

    let temp = tempdir().expect("tempdir");
    let input = temp.path().join("input.mp4");
    let output_dir = temp.path().join("frames");

    let status = Command::new(&ffmpeg)
        .args([
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=8",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&input)
        .status()
        .expect("run ffmpeg test generator");
    assert!(status.success(), "synthetic video generation failed");

    let job = VideoJob {
        input,
        output_dir: output_dir.clone(),
        sampling: Sampling {
            every_n_frames: 2,
            limit: Some(3),
        },
        transforms: TransformConfig {
            crop: Some(Crop {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
            }),
            crop_space: CropSpace::Source,
            resize: Some(Resize {
                width: 16,
                height: 16,
            }),
            color_space: Some(ColorSpace::Rgb),
        },
        format: ImageFormat::Png,
        prefix: "frame".to_string(),
        start_number: 0,
        manifests: vec![ManifestFormat::Jsonl, ManifestFormat::Csv],
        runtime: RuntimeConfig {
            ffmpeg_path: Some(ffmpeg),
            overwrite: true,
            ..RuntimeConfig::default()
        },
    };

    let report = extract_video(&job).expect("extract video");
    assert_eq!(report.frames_written, 3);
    assert!(output_dir.join("frame_000000.png").exists());
    assert!(output_dir.join("frame_000001.png").exists());
    assert!(output_dir.join("frame_000002.png").exists());

    let jsonl = fs::read_to_string(output_dir.join("manifest.jsonl")).expect("manifest jsonl");
    assert_eq!(jsonl.lines().count(), 3);
    assert!(jsonl.contains("\"crop_space\":\"source\""));

    let csv = fs::read_to_string(output_dir.join("manifest.csv")).expect("manifest csv");
    assert!(csv.contains("source_path,source_video_id,output_path"));
    assert_eq!(csv.lines().count(), 4);
}

#[test]
fn imports_legacy_yaml_config() {
    let temp = tempdir().expect("tempdir");
    let legacy = temp.path().join("legacy.yaml");
    fs::write(
        &legacy,
        r#"
input_dir: /tmp/video.mov
output_dir: /tmp/out
interval: 8
prefix: berries
start_number: 4
limit: 10
crop_area:
  - 0
  - 10
  - 100
  - 120
resize_to:
  width: 64
  height: 64
color_space: RGB
ext: jpg
parallel_workers: 3
"#,
    )
    .expect("write legacy config");

    let job = import_legacy_config(&legacy, None).expect("import legacy config");
    assert_eq!(job.sampling.every_n_frames, 8);
    assert_eq!(job.prefix, "berries");
    assert_eq!(job.start_number, 4);
    assert_eq!(job.format, ImageFormat::Jpeg);
    assert_eq!(job.runtime.workers, 3);
    assert_eq!(
        job.transforms.crop,
        Some(Crop {
            x: 0,
            y: 10,
            width: 100,
            height: 120
        })
    );
    assert_eq!(
        job.transforms.resize,
        Some(Resize {
            width: 64,
            height: 64
        })
    );
}

fn find_tool(name: &str) -> Option<std::path::PathBuf> {
    std::env::var_os(format!("VID2DATASET_{}", name.to_ascii_uppercase()))
        .map(std::path::PathBuf::from)
        .or_else(|| which::which(name).ok())
}
