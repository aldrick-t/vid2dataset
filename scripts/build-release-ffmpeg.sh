#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:?target is required}"
CANDIDATE="${2:?candidate JSON is required}"
SOURCE_DIR="${3:?verified source directory is required}"
OUTPUT_DIR="${4:?output directory is required}"

case "$TARGET" in
  macos-arm64) ARCHITECTURE="aarch64-apple-darwin"; export MACOSX_DEPLOYMENT_TARGET=11.0 ;;
  macos-x64) ARCHITECTURE="x86_64-apple-darwin"; export MACOSX_DEPLOYMENT_TARGET=10.15 ;;
  windows-x64) ARCHITECTURE="x86_64-pc-windows-gnu" ;;
  linux-x64) ARCHITECTURE="x86_64-unknown-linux-gnu" ;;
  *) echo "unsupported FFmpeg build target: $TARGET" >&2; exit 1 ;;
esac

json_get() {
  node -e 'const fs=require("fs"); const value=process.argv[2].split(".").reduce((item,key)=>item[key],JSON.parse(fs.readFileSync(process.argv[1],"utf8"))); process.stdout.write(String(value));' "$CANDIDATE" "$1"
}

FFMPEG_VERSION="$(json_get ffmpeg.version)"
ZLIB_VERSION="$(json_get zlib.version)"
REVISION="$(json_get build.revision)"
RECIPE_SHA256="$(json_get build.recipe_sha256)"
ARCHIVE_NAME="ffmpeg-${FFMPEG_VERSION}-r${REVISION}-${TARGET}.tar.gz"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
PREFIX="$WORK_DIR/prefix"
PACKAGE_ROOT="$WORK_DIR/package"
mkdir -p "$PREFIX" "$PACKAGE_ROOT/bin" "$OUTPUT_DIR"

tar -xf "$SOURCE_DIR/zlib-source.tar.gz" -C "$WORK_DIR"
tar -xf "$SOURCE_DIR/ffmpeg-source.tar.xz" -C "$WORK_DIR"
ZLIB_SOURCE="$WORK_DIR/zlib-$ZLIB_VERSION"
FFMPEG_SOURCE="$WORK_DIR/ffmpeg-$FFMPEG_VERSION"
[[ -n "$ZLIB_SOURCE" && -n "$FFMPEG_SOURCE" ]] || { echo "source archives did not contain expected directories" >&2; exit 1; }

JOBS="${NUMBER_OF_PROCESSORS:-}"
if [[ -z "$JOBS" ]]; then JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2)"; fi

(
  cd "$ZLIB_SOURCE"
  ./configure --static --prefix="$PREFIX"
  make -j"$JOBS"
  make test
  make install
)

CONFIGURE_FLAGS=()
while IFS= read -r flag; do
  CONFIGURE_FLAGS+=("$flag")
done < <(node -e 'const fs=require("fs"); for(const value of JSON.parse(fs.readFileSync(process.argv[1],"utf8")).build.configure_flags) console.log(value);' "$CANDIDATE")
EXTRA_LDFLAGS="-L../prefix/lib"
EXTRA_LIBS="-lz"
if [[ "$TARGET" == "windows-x64" ]]; then
  EXTRA_LDFLAGS="$EXTRA_LDFLAGS -static -static-libgcc"
  CONFIGURE_FLAGS+=("--enable-w32threads")
fi

(
  cd "$FFMPEG_SOURCE"
  PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" ./configure \
    --prefix=/opt/vid2dataset-ffmpeg \
    --pkg-config-flags=--static \
    --extra-cflags="-I../prefix/include" \
    --extra-ldflags="$EXTRA_LDFLAGS" \
    --extra-libs="$EXTRA_LIBS" \
    "${CONFIGURE_FLAGS[@]}"
  make -j"$JOBS" ffmpeg ffprobe
)

EXE_SUFFIX=""
if [[ "$TARGET" == "windows-x64" ]]; then EXE_SUFFIX=".exe"; fi
FFMPEG_BIN="$FFMPEG_SOURCE/ffmpeg$EXE_SUFFIX"
FFPROBE_BIN="$FFMPEG_SOURCE/ffprobe$EXE_SUFFIX"
[[ -x "$FFMPEG_BIN" && -x "$FFPROBE_BIN" ]] || { echo "FFmpeg build did not produce both executables" >&2; exit 1; }

if ! "$FFMPEG_BIN" -hide_banner -L | grep -F "GNU Lesser General Public" >/dev/null; then
  echo "FFmpeg did not report the expected LGPL license" >&2
  exit 1
fi
BUILD_CONF="$("$FFMPEG_BIN" -hide_banner -buildconf 2>&1)"
for flag in "${CONFIGURE_FLAGS[@]}"; do grep -F -- "$flag" <<<"$BUILD_CONF" >/dev/null; done
if grep -E -- '--enable-(gpl|nonfree)' <<<"$BUILD_CONF"; then
  echo "GPL or nonfree FFmpeg configuration detected" >&2
  exit 1
fi

for decoder in h264 hevc av1 vp9 mpeg4; do
  "$FFMPEG_BIN" -hide_banner -decoders 2>/dev/null | awk '{print $2}' | grep -Fx "$decoder" >/dev/null
done
for encoder in png mjpeg mpeg4; do
  "$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx "$encoder" >/dev/null
done
for filter in select crop scale format setpts; do
  "$FFMPEG_BIN" -hide_banner -filters 2>/dev/null | awk '{print $2}' | grep -Fx "$filter" >/dev/null
done

SMOKE_DIR="$WORK_DIR/smoke"
mkdir -p "$SMOKE_DIR/png" "$SMOKE_DIR/jpeg"
"$FFMPEG_BIN" -hide_banner -loglevel error -y -f lavfi -i testsrc=size=64x64:rate=8 -t 1 -c:v mpeg4 "$SMOKE_DIR/fixture.mp4"
"$FFPROBE_BIN" -v error -print_format json -show_format -show_streams "$SMOKE_DIR/fixture.mp4" | grep -F '"codec_name": "mpeg4"' >/dev/null
"$FFMPEG_BIN" -hide_banner -loglevel error -i "$SMOKE_DIR/fixture.mp4" -vf "select='not(mod(n\,2))',scale=16:16,format=rgb24,setpts=N/FRAME_RATE/TB" -frames:v 3 -vsync 0 "$SMOKE_DIR/png/frame_%03d.png"
"$FFMPEG_BIN" -hide_banner -loglevel error -i "$SMOKE_DIR/fixture.mp4" -vf "crop=32:32:0:0,scale=16:16" -frames:v 3 -vsync 0 "$SMOKE_DIR/jpeg/frame_%03d.jpg"
[[ "$(find "$SMOKE_DIR/png" -name '*.png' | wc -l | tr -d ' ')" == "3" ]]
[[ "$(find "$SMOKE_DIR/jpeg" -name '*.jpg' | wc -l | tr -d ' ')" == "3" ]]

case "$TARGET" in
  macos-*)
    if otool -L "$FFMPEG_BIN" | grep -E '/(opt|usr/local)|libz\.'; then
      echo "unexpected non-system macOS runtime dependency" >&2
      exit 1
    fi
    ;;
  linux-x64)
    if ldd "$FFMPEG_BIN" | grep -E 'libz\.so|not found'; then
      echo "unexpected Linux runtime dependency" >&2
      exit 1
    fi
    ;;
  windows-x64)
    if objdump -p "$FFMPEG_BIN" | grep 'DLL Name' | grep -Ei 'zlib|libgcc|libstdc\+\+|libwinpthread'; then
      echo "unexpected Windows runtime dependency" >&2
      exit 1
    fi
    ;;
esac

cp "$FFMPEG_BIN" "$PACKAGE_ROOT/bin/ffmpeg$EXE_SUFFIX"
cp "$FFPROBE_BIN" "$PACKAGE_ROOT/bin/ffprobe$EXE_SUFFIX"
cp "$FFMPEG_SOURCE/COPYING.LGPLv2.1" "$PACKAGE_ROOT/COPYING.LGPLv2.1"
cp "$ZLIB_SOURCE/LICENSE" "$PACKAGE_ROOT/ZLIB-LICENSE"
chmod u+wx,go+rx "$PACKAGE_ROOT/bin/ffmpeg$EXE_SUFFIX" "$PACKAGE_ROOT/bin/ffprobe$EXE_SUFFIX"

cat > "$PACKAGE_ROOT/THIRD-PARTY-NOTICES.txt" <<EOF
vid2dataset2 bundled media tools

FFmpeg $FFMPEG_VERSION
License: LGPL-2.1-or-later
Source: $(json_get ffmpeg.source_url)
Source SHA-256: $(json_get ffmpeg.source_sha256)

zlib $ZLIB_VERSION
License: Zlib
Source: $(json_get zlib.source_url)
Source SHA-256: $(json_get zlib.source_sha256)

These command-line tools are bundled as app resources. They are not installed
globally, added to PATH, or downloaded when vid2dataset2 starts.
EOF

TARGET="$TARGET" ARCHITECTURE="$ARCHITECTURE" FFMPEG_VERSION="$FFMPEG_VERSION" ZLIB_VERSION="$ZLIB_VERSION" REVISION="$REVISION" RECIPE_SHA256="$RECIPE_SHA256" \
  node -e 'const fs=require("fs"); const cp=require("child_process"); const info={target:process.env.TARGET,architecture:process.env.ARCHITECTURE,ffmpeg_version:process.env.FFMPEG_VERSION,zlib_version:process.env.ZLIB_VERSION,build_revision:Number(process.env.REVISION),recipe_sha256:process.env.RECIPE_SHA256,macos_deployment_target:process.env.MACOSX_DEPLOYMENT_TARGET||null,compiler:cp.execFileSync(process.env.CC||"gcc",["--version"],{encoding:"utf8"}).split("\n")[0]}; fs.writeFileSync(process.argv[1],JSON.stringify(info,null,2)+"\n");' "$PACKAGE_ROOT/BUILD-INFO.json"

find "$PACKAGE_ROOT" -exec touch -t 202001010000 {} +
(
  cd "$PACKAGE_ROOT"
  find . -type f -print | LC_ALL=C sort | COPYFILE_DISABLE=1 tar --format=ustar -cf - -T -
) | gzip -n > "$OUTPUT_DIR/$ARCHIVE_NAME"

node "$ROOT_DIR/scripts/ffmpeg-dependency.mjs" artifact-manifest \
  --target "$TARGET" \
  --archive "$OUTPUT_DIR/$ARCHIVE_NAME" \
  --architecture "$ARCHITECTURE" \
  --output "$OUTPUT_DIR/$TARGET.json"

echo "Built $OUTPUT_DIR/$ARCHIVE_NAME"
