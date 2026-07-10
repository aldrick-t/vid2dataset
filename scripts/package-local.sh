#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building CLI"
cargo build --release -p vid2dataset

if [[ ! -x "$ROOT_DIR/apps/desktop/src-tauri/ffmpeg/bin/ffmpeg" && ! -x "$ROOT_DIR/apps/desktop/src-tauri/ffmpeg/bin/ffmpeg.exe" ]]; then
  echo "Warning: no staged FFmpeg binary found under apps/desktop/src-tauri/ffmpeg/bin"
  echo "Run scripts/setup-ffmpeg.sh before packaging if you want FFmpeg bundled as a Tauri resource."
fi

echo "Building desktop bundle"
cd "$ROOT_DIR/apps/desktop"
npm ci
npm run tauri build

echo "Local packages are under target/release/bundle"
