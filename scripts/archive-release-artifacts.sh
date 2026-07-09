#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLUG="${1:-local}"
BUNDLE_ROOT="$ROOT_DIR/target/release/bundle"
FALLBACK_BUNDLE_ROOT="$ROOT_DIR/apps/desktop/src-tauri/target/release/bundle"
RELEASE_DIR="$ROOT_DIR/dist/release/$SLUG"

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
  echo "sha256sum or shasum is required" >&2
  exit 1
}

rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

if [[ ! -d "$BUNDLE_ROOT" && -d "$FALLBACK_BUNDLE_ROOT" ]]; then
  BUNDLE_ROOT="$FALLBACK_BUNDLE_ROOT"
fi

if [[ ! -d "$BUNDLE_ROOT" ]]; then
  echo "No Tauri bundle directory found at $BUNDLE_ROOT or $FALLBACK_BUNDLE_ROOT" >&2
  exit 1
fi

while IFS= read -r artifact; do
  cp "$artifact" "$RELEASE_DIR/"
done < <(
  find "$BUNDLE_ROOT" -type f \
    \( -name "*.dmg" -o -name "*.msi" -o -name "*.exe" -o -name "*.deb" -o -name "*.AppImage" \) \
    -print
)

while IFS= read -r app_bundle; do
  app_name="$(basename "$app_bundle")"
  zip_path="$RELEASE_DIR/${app_name%.app}-${SLUG}.app.zip"
  (
    cd "$(dirname "$app_bundle")"
    zip -qry "$zip_path" "$app_name"
  )
done < <(find "$BUNDLE_ROOT" -type d -name "*.app" -prune -print)

if ! find "$RELEASE_DIR" -type f ! -name SHA256SUMS -print -quit | grep -q .; then
  echo "No release artifacts were copied from $BUNDLE_ROOT" >&2
  exit 1
fi

(
  cd "$RELEASE_DIR"
  CHECKSUM_FILE="SHA256SUMS-${SLUG}"
  : > "$CHECKSUM_FILE"
  while IFS= read -r file; do
    checksum="$(sha256_file "$file")"
    printf '%s  %s\n' "$checksum" "$file" >> "$CHECKSUM_FILE"
  done < <(find . -maxdepth 1 -type f ! -name 'SHA256SUMS*' -print | sed 's#^\./##' | sort)
)

echo "Release artifacts are ready:"
find "$RELEASE_DIR" -maxdepth 1 -type f -print | sort
