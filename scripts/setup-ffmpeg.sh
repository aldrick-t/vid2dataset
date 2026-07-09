#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT_DIR/apps/desktop/src-tauri/ffmpeg/bin"
FFMPEG_ARG="${1:-}"
FFPROBE_ARG="${2:-}"

resolve_tool() {
  local name="$1"
  local explicit="$2"
  local env_name="$3"

  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return 0
  fi

  local env_value="${!env_name:-}"
  if [[ -n "$env_value" ]]; then
    printf '%s\n' "$env_value"
    return 0
  fi

  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi

  return 1
}

validate_tool() {
  local path="$1"
  local name="$2"

  if [[ ! -f "$path" ]]; then
    echo "$name not found at $path" >&2
    return 1
  fi

  if [[ ! -x "$path" ]]; then
    chmod +x "$path"
  fi

  "$path" -version >/dev/null
}

FFMPEG_PATH="$(resolve_tool ffmpeg "$FFMPEG_ARG" VID2DATASET_FFMPEG || true)"
FFPROBE_PATH="$(resolve_tool ffprobe "$FFPROBE_ARG" VID2DATASET_FFPROBE || true)"

if [[ -z "$FFMPEG_PATH" || -z "$FFPROBE_PATH" ]]; then
  cat >&2 <<'EOF'
Unable to find ffmpeg and ffprobe.

Provide explicit paths:
  scripts/setup-ffmpeg.sh /path/to/ffmpeg /path/to/ffprobe

or set:
  VID2DATASET_FFMPEG=/path/to/ffmpeg
  VID2DATASET_FFPROBE=/path/to/ffprobe

or install FFmpeg so both tools are on PATH.
EOF
  exit 1
fi

validate_tool "$FFMPEG_PATH" ffmpeg
validate_tool "$FFPROBE_PATH" ffprobe

mkdir -p "$DEST_DIR"

FFMPEG_DEST="$DEST_DIR/ffmpeg"
FFPROBE_DEST="$DEST_DIR/ffprobe"
if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* ]]; then
  FFMPEG_DEST="$DEST_DIR/ffmpeg.exe"
  FFPROBE_DEST="$DEST_DIR/ffprobe.exe"
fi

cp "$FFMPEG_PATH" "$FFMPEG_DEST"
cp "$FFPROBE_PATH" "$FFPROBE_DEST"
chmod u+wx,go+rx "$FFMPEG_DEST" "$FFPROBE_DEST"

"$FFMPEG_DEST" -version >/dev/null
"$FFPROBE_DEST" -version >/dev/null

echo "Staged FFmpeg tools:"
echo "  $FFMPEG_DEST"
echo "  $FFPROBE_DEST"
echo
echo "These binaries are gitignored and will be packaged as Tauri resources when present."
