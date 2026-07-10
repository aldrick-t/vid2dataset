#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${VID2DATASET_FFMPEG_LOCK:-$ROOT_DIR/.github/ffmpeg.lock.json}"
DEST_ROOT="$ROOT_DIR/apps/desktop/src-tauri/ffmpeg"
NOTICE_DEST="$DEST_ROOT/THIRD_PARTY_FFMPEG.txt"
FFMPEG_LICENSE_DEST="$DEST_ROOT/LICENSE.LGPLv2.1.txt"
ZLIB_LICENSE_DEST="$DEST_ROOT/ZLIB_LICENSE.txt"
BUILD_INFO_DEST="$DEST_ROOT/FFMPEG_BUILD_INFO.txt"

TARGET=""
if [[ "${1:-}" == "--target" ]]; then
  TARGET="${2:-}"
  [[ -n "$TARGET" ]] || { echo "--target requires a value" >&2; exit 1; }
  shift 2
fi

ARCHIVE_URL="${VID2DATASET_RELEASE_FFMPEG_ARCHIVE_URL:-}"
ARCHIVE_SHA256="${VID2DATASET_RELEASE_FFMPEG_SHA256:-}"
ARCHIVE_FORMAT="${VID2DATASET_RELEASE_FFMPEG_ARCHIVE_FORMAT:-}"
ARCHIVE_PATH="${VID2DATASET_RELEASE_FFMPEG_ARCHIVE_PATH:-}"
FFMPEG_MEMBER=""
FFPROBE_MEMBER=""
FFMPEG_VERSION="unknown"
FFMPEG_SOURCE_URL="${VID2DATASET_RELEASE_FFMPEG_SOURCE_URL:-}"
FFMPEG_REDISTRIBUTION_SOURCE_URL=""
FFMPEG_SOURCE_SHA256=""
FFMPEG_SIGNATURE_URL=""
FFMPEG_SIGNING_FINGERPRINT=""
FFMPEG_LICENSE="LGPL-2.1-or-later"
FFMPEG_LICENSE_URL="${VID2DATASET_RELEASE_FFMPEG_LICENSE_URL:-https://ffmpeg.org/legal.html}"
ZLIB_VERSION="unknown"
ZLIB_SOURCE_URL=""
ZLIB_REDISTRIBUTION_SOURCE_URL=""
ZLIB_SOURCE_SHA256=""
ZLIB_LICENSE="Zlib"
ZLIB_LICENSE_URL=""
DEPENDENCY_RELEASE_TAG="manual"
FFMPEG_PATH="${1:-}"
FFPROBE_PATH="${2:-}"

fail() {
  echo "stage-release-ffmpeg: $*" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail "sha256sum or shasum is required"
  fi
}

extract_archive() {
  local archive="$1"
  local format="$2"
  local dest="$3"
  mkdir -p "$dest"
  case "$format" in
    zip) unzip -q "$archive" -d "$dest" ;;
    tar|tar.gz|tgz|tar.xz|txz) tar -xf "$archive" -C "$dest" ;;
    *) fail "unsupported FFmpeg archive format: $format" ;;
  esac
}

find_tool() {
  find "$1" -type f \( -name "$2" -o -name "$2.exe" \) -print -quit
}

if [[ -n "$TARGET" ]]; then
  while IFS=$'\t' read -r key value; do
    case "$key" in
      ARCHIVE_URL) ARCHIVE_URL="$value" ;;
      ARCHIVE_SHA256) ARCHIVE_SHA256="$value" ;;
      ARCHIVE_FORMAT) ARCHIVE_FORMAT="$value" ;;
      FFMPEG_MEMBER) FFMPEG_MEMBER="$value" ;;
      FFPROBE_MEMBER) FFPROBE_MEMBER="$value" ;;
      FFMPEG_VERSION) FFMPEG_VERSION="$value" ;;
      FFMPEG_SOURCE_URL) FFMPEG_SOURCE_URL="$value" ;;
      FFMPEG_REDISTRIBUTION_SOURCE_URL) FFMPEG_REDISTRIBUTION_SOURCE_URL="$value" ;;
      FFMPEG_SOURCE_SHA256) FFMPEG_SOURCE_SHA256="$value" ;;
      FFMPEG_SIGNATURE_URL) FFMPEG_SIGNATURE_URL="$value" ;;
      FFMPEG_SIGNING_FINGERPRINT) FFMPEG_SIGNING_FINGERPRINT="$value" ;;
      FFMPEG_LICENSE) FFMPEG_LICENSE="$value" ;;
      FFMPEG_LICENSE_URL) FFMPEG_LICENSE_URL="$value" ;;
      ZLIB_VERSION) ZLIB_VERSION="$value" ;;
      ZLIB_SOURCE_URL) ZLIB_SOURCE_URL="$value" ;;
      ZLIB_REDISTRIBUTION_SOURCE_URL) ZLIB_REDISTRIBUTION_SOURCE_URL="$value" ;;
      ZLIB_SOURCE_SHA256) ZLIB_SOURCE_SHA256="$value" ;;
      ZLIB_LICENSE) ZLIB_LICENSE="$value" ;;
      ZLIB_LICENSE_URL) ZLIB_LICENSE_URL="$value" ;;
      DEPENDENCY_RELEASE_TAG) DEPENDENCY_RELEASE_TAG="$value" ;;
      *) fail "unexpected lock field: $key" ;;
    esac
  done < <(node "$ROOT_DIR/scripts/ffmpeg-dependency.mjs" target-tsv --lock "$LOCK_FILE" --target "$TARGET")
fi

if [[ -n "$FFMPEG_PATH" || -n "$FFPROBE_PATH" ]]; then
  [[ -n "$FFMPEG_PATH" && -n "$FFPROBE_PATH" ]] || fail "provide both ffmpeg and ffprobe paths"
  "$ROOT_DIR/scripts/setup-ffmpeg.sh" "$FFMPEG_PATH" "$FFPROBE_PATH"
  exit 0
fi

[[ -n "$ARCHIVE_URL" || -n "$ARCHIVE_PATH" ]] || fail "use --target with a ready lock, or set an archive URL/path"
[[ -n "$ARCHIVE_SHA256" ]] || fail "FFmpeg archive SHA-256 is required"
[[ -n "$FFMPEG_SOURCE_URL" ]] || fail "FFmpeg source URL is required"

if [[ -z "$ARCHIVE_FORMAT" ]]; then
  case "${ARCHIVE_PATH:-$ARCHIVE_URL}" in
    *.zip) ARCHIVE_FORMAT="zip" ;;
    *.tar.gz|*.tgz) ARCHIVE_FORMAT="tar.gz" ;;
    *.tar.xz|*.txz) ARCHIVE_FORMAT="tar.xz" ;;
    *.tar) ARCHIVE_FORMAT="tar" ;;
    *) fail "set VID2DATASET_RELEASE_FFMPEG_ARCHIVE_FORMAT" ;;
  esac
fi

WORK_DIR="$(mktemp -d)"
ARCHIVE="$WORK_DIR/ffmpeg-archive.$ARCHIVE_FORMAT"
EXTRACT_DIR="$WORK_DIR/extract"
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ -n "$ARCHIVE_PATH" ]]; then
  cp "$ARCHIVE_PATH" "$ARCHIVE"
else
  curl --proto '=https' --tlsv1.2 -fsSL "$ARCHIVE_URL" -o "$ARCHIVE"
fi

ACTUAL_SHA256="$(sha256_file "$ARCHIVE")"
[[ "$ACTUAL_SHA256" == "$ARCHIVE_SHA256" ]] || fail "FFmpeg archive checksum mismatch: expected $ARCHIVE_SHA256, got $ACTUAL_SHA256"
extract_archive "$ARCHIVE" "$ARCHIVE_FORMAT" "$EXTRACT_DIR"

if [[ -n "$FFMPEG_MEMBER" ]]; then
  FFMPEG_FOUND="$EXTRACT_DIR/$FFMPEG_MEMBER"
  FFPROBE_FOUND="$EXTRACT_DIR/$FFPROBE_MEMBER"
else
  FFMPEG_FOUND="$(find_tool "$EXTRACT_DIR" ffmpeg)"
  FFPROBE_FOUND="$(find_tool "$EXTRACT_DIR" ffprobe)"
fi
[[ -f "$FFMPEG_FOUND" ]] || fail "ffmpeg executable not found at the locked archive path"
[[ -f "$FFPROBE_FOUND" ]] || fail "ffprobe executable not found at the locked archive path"

"$ROOT_DIR/scripts/setup-ffmpeg.sh" "$FFMPEG_FOUND" "$FFPROBE_FOUND"
mkdir -p "$DEST_ROOT"

if [[ -f "$EXTRACT_DIR/COPYING.LGPLv2.1" ]]; then cp "$EXTRACT_DIR/COPYING.LGPLv2.1" "$FFMPEG_LICENSE_DEST"; fi
if [[ -f "$EXTRACT_DIR/ZLIB-LICENSE" ]]; then cp "$EXTRACT_DIR/ZLIB-LICENSE" "$ZLIB_LICENSE_DEST"; fi
if [[ -f "$EXTRACT_DIR/BUILD-INFO.json" ]]; then cp "$EXTRACT_DIR/BUILD-INFO.json" "$BUILD_INFO_DEST"; fi

cat > "$NOTICE_DEST" <<EOF
Bundled FFmpeg dependency notice

Dependency release: $DEPENDENCY_RELEASE_TAG
Binary archive: ${ARCHIVE_URL:-local archive}
Binary SHA-256: $ARCHIVE_SHA256

FFmpeg $FFMPEG_VERSION
License: $FFMPEG_LICENSE
License URL: $FFMPEG_LICENSE_URL
Source: $FFMPEG_SOURCE_URL
Redistributed source: $FFMPEG_REDISTRIBUTION_SOURCE_URL
Source SHA-256: $FFMPEG_SOURCE_SHA256
Signature: $FFMPEG_SIGNATURE_URL
Signing fingerprint: $FFMPEG_SIGNING_FINGERPRINT

zlib $ZLIB_VERSION
License: $ZLIB_LICENSE
License URL: $ZLIB_LICENSE_URL
Source: $ZLIB_SOURCE_URL
Redistributed source: $ZLIB_REDISTRIBUTION_SOURCE_URL
Source SHA-256: $ZLIB_SOURCE_SHA256

vid2dataset2 bundles ffmpeg and ffprobe as app resources. The application does
not install them globally, add them to PATH, or download them on first run.
EOF

echo "Staged locked FFmpeg dependency $DEPENDENCY_RELEASE_TAG for ${TARGET:-manual packaging}"
