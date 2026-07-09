# v2 Architecture

## Components

- `crates/vid2dataset-core`: shared domain model, config validation, FFmpeg discovery, ffprobe inspection, FFmpeg extraction command generation, manifests, and legacy config import.
- `crates/vid2dataset-cli`: full command-line interface for extraction, inspection, doctor checks, profiles, and presets.
- `apps/desktop`: Tauri v2 desktop app with React/Vite frontend and Rust command handlers into the core.
- `legacy/`: preserved Python/OpenCV/PySide implementation.

## FFmpeg Policy

Discovery order:

1. Explicit config/CLI path.
2. `VID2DATASET_FFMPEG` and `VID2DATASET_FFPROBE`.
3. Bundled binaries next to the app under `ffmpeg/bin/`.
4. System `PATH`.

Bundled builds should be LGPL-friendly by default. Advanced users can override with their own system FFmpeg when they need custom codec support.

## Transform Order

The core builds one FFmpeg filtergraph where possible:

- Sampling filter: `select='not(mod(n\,N))'`
- `source` crop space: crop, then scale.
- `output` crop space: scale, then crop.
- Optional color format conversion.

This keeps long or high-resolution video processing streaming and avoids loading whole videos into application memory.

## Deferred Scope

COCO/YOLO exports are intentionally deferred until annotation import or annotation UI exists. v1 emits raw images and traceability manifests only.

