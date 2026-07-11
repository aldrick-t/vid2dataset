# vid2dataset
[![Release](https://github.com/aldrick-t/vid2dataset/actions/workflows/release.yml/badge.svg)](https://github.com/aldrick-t/vid2dataset/actions/workflows/release.yml)

`vid2dataset` creates image datasets from video files. Version 2 is a clean Rust + Tauri rewrite with a shared FFmpeg-backed core, a full CLI, and a non-Qt desktop GUI.

The previous Python/OpenCV/PySide implementation is preserved in [`legacy/`](legacy/) for reference and migration.

## Current vid2dataset2 Status

`vid2dataset` remains the app and binary name. `vid2dataset2` is the Rust +
Tauri revamp release train. The current revamp milestone is
`vid2dataset2 Alpha 2`, with machine version `0.2.0-alpha.2` and release tag
`v0.2.0-alpha.2`.

- Rust workspace scaffolded with `vid2dataset-core` and `vid2dataset` CLI crates.
- Tauri + React desktop app skeleton with guided and advanced extraction controls.
- FFmpeg/ffprobe discovery supports config overrides, `VID2DATASET_FFMPEG`, `VID2DATASET_FFPROBE`, bundled binaries, and `PATH`.
- Extraction uses FFmpeg filtergraphs for frame sampling, crop, resize, color conversion, and image sequence output.
- Manifests are written as JSONL and CSV.
- Legacy YAML config import is available.
- Desktop preview renders the selected video, supports scrubbing, time/frame seek, and interactive crop editing.

## Current Status

This repository is a functional vid2dataset2 alpha, not a signed stable
production release.

Implemented:

- Rust core and CLI.
- Tauri + React desktop app.
- FFmpeg discovery through explicit paths, env vars, staged bundled resources, and `PATH`.
- FFmpeg-backed extraction command generation.
- JSONL/CSV manifest writing.
- Video preview with scrubber, time seek, frame seek, drag-to-create crop, crop move/resize, and explicit crop/no-crop confirmation.
- Legacy Python implementation preserved under [`legacy/`](legacy/).

Still deferred beyond the current alpha:

- Hosted web/server backend.
- COCO/YOLO annotation exports.

The release flow for signed/notarized public installers and automated bundled
FFmpeg staging is documented in [`docs/release.md`](docs/release.md).

## Development Requirements

- Rust toolchain, including `cargo`.
- Node.js and npm for the desktop app.
- FFmpeg and ffprobe available on `PATH`, via env vars, or staged under `apps/desktop/src-tauri/ffmpeg/bin/`.
- Tauri system prerequisites for your OS.

Useful checks:

```bash
cargo --version
node --version
npm --version
ffmpeg -version
ffprobe -version
```

## Install

```bash
cargo fetch
cd apps/desktop
npm ci
```

## Verify

```bash
cargo fmt --all --check
cargo test --workspace --all-targets
cargo run -p vid2dataset -- doctor
cd apps/desktop
npm ci
npm run build
```

`doctor` exits non-zero when FFmpeg or ffprobe are missing, but it should print remediation instructions.

## CLI

```bash
cargo run -p vid2dataset -- doctor
cargo run -p vid2dataset -- inspect path/to/video.mp4
cargo run -p vid2dataset -- extract \
  --input path/to/video.mp4 \
  --output output/dataset \
  --every-n-frames 8 \
  --crop 0,376,1080,1080 \
  --crop-space source \
  --resize 640x640 \
  --format png \
  --manifest jsonl,csv
```

Import old settings:

```bash
cargo run -p vid2dataset -- profiles import legacy/config/config.yaml
```

## Desktop

```bash
cd apps/desktop
npm ci
npm run tauri dev
```

The GUI intentionally avoids Qt. It uses Tauri v2 with a React/Vite frontend and calls the same Rust core as the CLI.

For frontend-only UI development:

```bash
cd apps/desktop
npm run dev
```

This starts Vite on `http://localhost:1420/`.

## Stopping Dev Servers

Stale Vite or Tauri dev servers can keep port `1420` busy and make the next dev run misleading.

Check the port:

```bash
lsof -i :1420
```

Stop a specific process:

```bash
kill <PID>
```

Force stop only when the process ignores a normal signal:

```bash
kill -9 <PID>
```

Scoped cleanup options:

```bash
pkill -f vite
pkill -f "npm run dev"
```

Use `pkill` carefully because it can stop other matching dev servers.

## FFmpeg Setup

“Bundled FFmpeg” means binaries are staged locally under:

```text
apps/desktop/src-tauri/ffmpeg/bin/
```

The repo does not commit FFmpeg binaries. For development, system FFmpeg from Homebrew, a package manager, or explicit env vars is enough. For portable desktop installers, stage redistributable LGPL-compatible `ffmpeg` and `ffprobe` binaries before packaging so Tauri includes them as app resources.

To stage local binaries for Tauri packaging:

```bash
scripts/setup-ffmpeg.sh /path/to/ffmpeg /path/to/ffprobe
```

or use environment variables:

```bash
VID2DATASET_FFMPEG=/path/to/ffmpeg \
VID2DATASET_FFPROBE=/path/to/ffprobe \
scripts/setup-ffmpeg.sh
```

If `ffmpeg` and `ffprobe` are already on `PATH`, run:

```bash
scripts/setup-ffmpeg.sh
```

The script validates both tools with `-version`, copies them into the Tauri
resource path, and keeps them gitignored. Installed desktop apps should not
require end users to install FFmpeg separately when release packaging has staged
the binaries first. Bundled FFmpeg stays inside the app package/resource
directory; it is not installed globally, does not mutate PATH, and is not
downloaded on first run.

Public builds do not use manually maintained GitHub variables. The reviewed
dependency record lives in [`.github/ffmpeg.lock.json`](.github/ffmpeg.lock.json),
and the quarterly/manual dependency workflow builds FFmpeg and ffprobe from
verified official source for every release target. See
[`docs/release.md`](docs/release.md) for first-bootstrap and update procedures.

## Desktop vs Web

The React/Vite UI can run in a browser during development. A browser-only app cannot fully process large local videos with FFmpeg because it lacks native process and broad filesystem access.

Viable future web paths:

- React web UI plus a local Rust backend service.
- Hosted web UI plus server-side video processing.
- Current v1 default: Tauri desktop app using the same React UI and Rust core locally.

## Crop Semantics

- `source`: crop coordinates are interpreted against the original video frame, then resize and other transforms are applied.
- `output`: resize is applied first, then crop coordinates are interpreted against the resized output frame.

`source` is the default for reproducible dataset creation.

## Manifests

Each extraction can write:

- `manifest.jsonl`
- `manifest.csv`

Records include source path, output path, sequence number, inferred frame index, sampling rule, crop, crop coordinate space, resize, color space, image format, file size, and SHA-256.

## Packaging

Use [`scripts/package-local.sh`](scripts/package-local.sh) after installing Rust, Node, npm dependencies, and Tauri system prerequisites.

For local unsigned builds with bundled FFmpeg, stage LGPL-friendly FFmpeg and ffprobe binaries first:

```bash
scripts/setup-ffmpeg.sh
scripts/package-local.sh
```

or use system FFmpeg/env var overrides at runtime instead of staging for source/development builds. Public release packaging stages locked LGPL-compatible binaries with checksum verification through [`scripts/stage-release-ffmpeg.sh`](scripts/stage-release-ffmpeg.sh); the application never performs a first-run download.
