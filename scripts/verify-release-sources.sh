#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="${1:?candidate JSON is required}"
OUTPUT_DIR="${2:?output directory is required}"

json_get() {
  node -e 'const fs=require("fs"); const value=process.argv[2].split(".").reduce((item,key)=>item[key],JSON.parse(fs.readFileSync(process.argv[1],"utf8"))); if(value!==null&&value!==undefined) process.stdout.write(String(value));' "$CANDIDATE" "$1"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_signature() {
  local source="$1"
  local signature="$2"
  local key_file="$3"
  local expected_fingerprint="$4"
  local keyring="$5"
  local actual_fingerprint

  mkdir -p "$keyring"
  chmod 700 "$keyring"
  actual_fingerprint="$(gpg --homedir "$keyring" --batch --with-colons --import-options show-only --import "$key_file" | awk -F: '$1 == "fpr" {print $10; exit}')"
  if [[ "$actual_fingerprint" != "$expected_fingerprint" ]]; then
    echo "signing-key fingerprint mismatch: expected $expected_fingerprint, got $actual_fingerprint" >&2
    exit 1
  fi
  gpg --homedir "$keyring" --batch --import "$key_file"
  gpgv --keyring "$keyring/pubring.kbx" "$signature" "$source"
  printf '%s' "$actual_fingerprint"
}

mkdir -p "$OUTPUT_DIR"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

FFMPEG_URL="$(json_get ffmpeg.source_url)"
FFMPEG_SIGNATURE_URL="$(json_get ffmpeg.signature_url)"
FFMPEG_KEY_URL="$(json_get ffmpeg.signing_key_url)"
FFMPEG_FINGERPRINT="$(json_get ffmpeg.signing_key_fingerprint)"
FFMPEG_EXPECTED_SHA="$(json_get ffmpeg.source_sha256)"
FFMPEG_VERSION="$(json_get ffmpeg.version)"
ZLIB_URL="$(json_get zlib.source_url)"
ZLIB_SIGNATURE_URL="$(json_get zlib.signature_url)"
ZLIB_KEY_URL="$(json_get zlib.signing_key_url)"
ZLIB_FINGERPRINT="$(json_get zlib.signing_key_fingerprint)"
ZLIB_EXPECTED_SHA="$(json_get zlib.source_sha256)"
ZLIB_VERSION="$(json_get zlib.version)"

curl --proto '=https' --tlsv1.2 -fsSL "$FFMPEG_URL" -o "$OUTPUT_DIR/ffmpeg-source.tar.xz"
curl --proto '=https' --tlsv1.2 -fsSL "$FFMPEG_SIGNATURE_URL" -o "$OUTPUT_DIR/ffmpeg-source.tar.xz.asc"
curl --proto '=https' --tlsv1.2 -fsSL "$FFMPEG_KEY_URL" -o "$OUTPUT_DIR/ffmpeg-signing-key.asc"
curl --proto '=https' --tlsv1.2 -fsSL "$ZLIB_URL" -o "$OUTPUT_DIR/zlib-source.tar.gz"
curl --proto '=https' --tlsv1.2 -fsSL "$ZLIB_SIGNATURE_URL" -o "$OUTPUT_DIR/zlib-source.tar.gz.asc"
curl --proto '=https' --tlsv1.2 -fsSL "$ZLIB_KEY_URL" -o "$WORK_DIR/zlib-key.html"
sed -n '/BEGIN PGP PUBLIC KEY BLOCK/,/END PGP PUBLIC KEY BLOCK/p' "$WORK_DIR/zlib-key.html" > "$OUTPUT_DIR/zlib-signing-key.asc"

FFMPEG_ACTUAL_FINGERPRINT="$(verify_signature "$OUTPUT_DIR/ffmpeg-source.tar.xz" "$OUTPUT_DIR/ffmpeg-source.tar.xz.asc" "$OUTPUT_DIR/ffmpeg-signing-key.asc" "$FFMPEG_FINGERPRINT" "$WORK_DIR/ffmpeg-gpg")"
ZLIB_ACTUAL_FINGERPRINT="$(verify_signature "$OUTPUT_DIR/zlib-source.tar.gz" "$OUTPUT_DIR/zlib-source.tar.gz.asc" "$OUTPUT_DIR/zlib-signing-key.asc" "$ZLIB_FINGERPRINT" "$WORK_DIR/zlib-gpg")"

FFMPEG_ACTUAL_SHA="$(sha256_file "$OUTPUT_DIR/ffmpeg-source.tar.xz")"
ZLIB_ACTUAL_SHA="$(sha256_file "$OUTPUT_DIR/zlib-source.tar.gz")"
if [[ -n "$FFMPEG_EXPECTED_SHA" && "$FFMPEG_EXPECTED_SHA" != "$FFMPEG_ACTUAL_SHA" ]]; then
  echo "FFmpeg source checksum mismatch: expected $FFMPEG_EXPECTED_SHA, got $FFMPEG_ACTUAL_SHA" >&2
  exit 1
fi
if [[ "$ZLIB_EXPECTED_SHA" != "$ZLIB_ACTUAL_SHA" ]]; then
  echo "zlib source checksum mismatch: expected $ZLIB_EXPECTED_SHA, got $ZLIB_ACTUAL_SHA" >&2
  exit 1
fi

tar -xJOf "$OUTPUT_DIR/ffmpeg-source.tar.xz" "ffmpeg-$FFMPEG_VERSION/COPYING.LGPLv2.1" > "$OUTPUT_DIR/FFMPEG-COPYING.LGPLv2.1"
tar -xzOf "$OUTPUT_DIR/zlib-source.tar.gz" "zlib-$ZLIB_VERSION/LICENSE" > "$OUTPUT_DIR/ZLIB-LICENSE"

node "$ROOT_DIR/scripts/ffmpeg-dependency.mjs" finalize-sources \
  --candidate "$CANDIDATE" \
  --ffmpeg-sha256 "$FFMPEG_ACTUAL_SHA" \
  --zlib-sha256 "$ZLIB_ACTUAL_SHA" \
  --ffmpeg-fingerprint "$FFMPEG_ACTUAL_FINGERPRINT" \
  --zlib-fingerprint "$ZLIB_ACTUAL_FINGERPRINT" \
  --output "$OUTPUT_DIR/candidate.json"

echo "Verified FFmpeg and zlib release sources in $OUTPUT_DIR"
