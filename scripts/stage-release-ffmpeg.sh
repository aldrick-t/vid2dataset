#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT_DIR/apps/desktop/src-tauri/ffmpeg/bin"
NOTICE_DEST="$ROOT_DIR/apps/desktop/src-tauri/ffmpeg/THIRD_PARTY_FFMPEG.txt"

ARCHIVE_URL="${VID2DATASET_RELEASE_FFMPEG_ARCHIVE_URL:-}"
ARCHIVE_SHA256="${VID2DATASET_RELEASE_FFMPEG_SHA256:-}"
ARCHIVE_PATH="${VID2DATASET_RELEASE_FFMPEG_ARCHIVE_PATH:-}"
SOURCE_URL="${VID2DATASET_RELEASE_FFMPEG_SOURCE_URL:-}"
LICENSE_URL="${VID2DATASET_RELEASE_FFMPEG_LICENSE_URL:-https://ffmpeg.org/legal.html}"
FFMPEG_PATH="${1:-}"
FFPROBE_PATH="${2:-}"

fail() {
  echo "stage-release-ffmpeg: $*" >&2
  exit 1
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
    return 0
  fi
  fail "sha256sum or shasum is required"
}

extract_archive() {
  local archive="$1"
  local dest="$2"

  mkdir -p "$dest"
  case "$archive" in
    *.zip)
      unzip -q "$archive" -d "$dest"
      ;;
    *.tar|*.tar.gz|*.tgz|*.tar.xz|*.txz)
      tar -xf "$archive" -C "$dest"
      ;;
    *)
      fail "unsupported FFmpeg archive format: $archive"
      ;;
  esac
}

find_tool() {
  local root="$1"
  local name="$2"
  find "$root" -type f \( -name "$name" -o -name "$name.exe" \) -print -quit
}

stage_from_paths() {
  local ffmpeg="$1"
  local ffprobe="$2"
  "$ROOT_DIR/scripts/setup-ffmpeg.sh" "$ffmpeg" "$ffprobe"
}

if [[ -n "$FFMPEG_PATH" || -n "$FFPROBE_PATH" ]]; then
  [[ -n "$FFMPEG_PATH" && -n "$FFPROBE_PATH" ]] || fail "provide both ffmpeg and ffprobe paths"
  stage_from_paths "$FFMPEG_PATH" "$FFPROBE_PATH"
  exit 0
fi

[[ -n "$ARCHIVE_URL" || -n "$ARCHIVE_PATH" ]] || fail "set VID2DATASET_RELEASE_FFMPEG_ARCHIVE_URL or VID2DATASET_RELEASE_FFMPEG_ARCHIVE_PATH"
[[ -n "$ARCHIVE_SHA256" ]] || fail "set VID2DATASET_RELEASE_FFMPEG_SHA256"
[[ -n "$SOURCE_URL" ]] || fail "set VID2DATASET_RELEASE_FFMPEG_SOURCE_URL"

WORK_DIR="$(mktemp -d)"
ARCHIVE="$WORK_DIR/ffmpeg-archive"
EXTRACT_DIR="$WORK_DIR/extract"
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ -n "$ARCHIVE_PATH" ]]; then
  cp "$ARCHIVE_PATH" "$ARCHIVE"
else
  curl -fsSL "$ARCHIVE_URL" -o "$ARCHIVE"
fi

ACTUAL_SHA256="$(sha256_file "$ARCHIVE")"
if [[ "${ACTUAL_SHA256,,}" != "${ARCHIVE_SHA256,,}" ]]; then
  fail "FFmpeg archive checksum mismatch: expected $ARCHIVE_SHA256, got $ACTUAL_SHA256"
fi

extract_archive "$ARCHIVE" "$EXTRACT_DIR"
FFMPEG_FOUND="$(find_tool "$EXTRACT_DIR" ffmpeg)"
FFPROBE_FOUND="$(find_tool "$EXTRACT_DIR" ffprobe)"
[[ -n "$FFMPEG_FOUND" ]] || fail "ffmpeg executable not found in archive"
[[ -n "$FFPROBE_FOUND" ]] || fail "ffprobe executable not found in archive"

stage_from_paths "$FFMPEG_FOUND" "$FFPROBE_FOUND"

mkdir -p "$(dirname "$NOTICE_DEST")"
cat > "$NOTICE_DEST" <<EOF
Bundled FFmpeg notice

Archive URL: ${ARCHIVE_URL:-local archive}
Source URL: $SOURCE_URL
License URL: $LICENSE_URL
SHA-256: $ARCHIVE_SHA256

vid2dataset2 bundles ffmpeg and ffprobe as app resources. The release build
does not install FFmpeg globally, does not add it to PATH, and does not
download it on first run.
EOF

echo "Wrote FFmpeg release notice:"
echo "  $NOTICE_DEST"
