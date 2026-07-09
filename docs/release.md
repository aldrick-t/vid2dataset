# vid2dataset2 Release Flow

## Nomenclature and Versioning

`vid2dataset` remains the product name, CLI binary name, and desktop app name.
`vid2dataset2` names the Rust + Tauri revamp release train. The legacy Python
implementation remains under `legacy/` for reference and migration checks.

The current development release is:

- Product release title: `vid2dataset2 Alpha 2`
- Machine version: `0.2.0-alpha.2`
- Git tag: `v0.2.0-alpha.2`

Use SemVer prerelease identifiers for every public prerelease:

- Alpha: `0.x.y-alpha.N`
- Beta: `0.x.y-beta.N`
- Release candidate: `0.x.y-rc.N`
- Stable: `0.x.y`

Keep these files in sync for every version bump:

- `crates/vid2dataset-core/Cargo.toml`
- `crates/vid2dataset-cli/Cargo.toml`
- `apps/desktop/src-tauri/Cargo.toml`
- `apps/desktop/package.json`
- `apps/desktop/package-lock.json`
- `apps/desktop/src-tauri/tauri.conf.json`
- `Cargo.lock`

If the project later chooses `2.0.0-alpha.N` instead of `0.x.y-alpha.N`, update
all package metadata, tags, release titles, and docs together.

## Distribution Policy

GitHub Releases are the first public distribution channel for alpha, beta,
release candidate, and stable artifacts. Hosted web/server processing and
COCO/YOLO export support are deferred release trains and must not block
vid2dataset2 desktop packaging.

Public desktop builds target:

- macOS ARM64 and x64: DMG installer plus zipped `.app` where Tauri produces it.
- Windows x64: MSI installer. Portable Windows packaging can be added after the
  app resource layout is validated outside the installer.
- Linux x64: AppImage and deb. AppImage is the primary portable-style Linux
  artifact; deb is a normal system package install.

Every uploaded release artifact must have a SHA-256 checksum. The release
workflow uploads per-platform `SHA256SUMS-<platform>` files and a combined
`SHA256SUMS` file.

## FFmpeg Portability Policy

Public desktop builds must bundle redistributable LGPL-compatible `ffmpeg` and
`ffprobe` binaries as Tauri app resources under `ffmpeg/bin/`.

Bundled FFmpeg must not:

- install into a global user or system directory;
- mutate the user's PATH;
- require a first-run download;
- require users to install FFmpeg separately for normal desktop use.

The core discovery order intentionally remains:

1. Explicit config/CLI path.
2. `VID2DATASET_FFMPEG` and `VID2DATASET_FFPROBE`.
3. Bundled app resources under `ffmpeg/bin/`.
4. System `PATH`.

The overrides are for development, diagnostics, and advanced users. Public
installer portability is validated by confirming `doctor` reports the source as
`bundled` from a packaged app with no system FFmpeg dependency.

## Release Automation

The release workflow is `.github/workflows/release.yml`.

It runs on pushed `v*` tags and can also be started manually with:

- `tag`: defaults to `v0.2.0-alpha.2`
- `release_name`: defaults to `vid2dataset2 Alpha 2`
- `prerelease`: defaults to `true`
- `draft`: defaults to `true`

Before packaging, configure GitHub repository variables for each platform:

- `VID2DATASET_FFMPEG_MACOS_ARM64_URL`
- `VID2DATASET_FFMPEG_MACOS_ARM64_SHA256`
- `VID2DATASET_FFMPEG_MACOS_ARM64_SOURCE_URL`
- `VID2DATASET_FFMPEG_MACOS_ARM64_LICENSE_URL`
- `VID2DATASET_FFMPEG_MACOS_X64_URL`
- `VID2DATASET_FFMPEG_MACOS_X64_SHA256`
- `VID2DATASET_FFMPEG_MACOS_X64_SOURCE_URL`
- `VID2DATASET_FFMPEG_MACOS_X64_LICENSE_URL`
- `VID2DATASET_FFMPEG_WINDOWS_X64_URL`
- `VID2DATASET_FFMPEG_WINDOWS_X64_SHA256`
- `VID2DATASET_FFMPEG_WINDOWS_X64_SOURCE_URL`
- `VID2DATASET_FFMPEG_WINDOWS_X64_LICENSE_URL`
- `VID2DATASET_FFMPEG_LINUX_X64_URL`
- `VID2DATASET_FFMPEG_LINUX_X64_SHA256`
- `VID2DATASET_FFMPEG_LINUX_X64_SOURCE_URL`
- `VID2DATASET_FFMPEG_LINUX_X64_LICENSE_URL`

Each FFmpeg archive must contain executable `ffmpeg` and `ffprobe` files. The
workflow verifies the configured SHA-256 before staging binaries into the Tauri
resource directory.

Optional macOS signing and notarization secrets:

- `APPLE_ID`
- `APPLE_PASSWORD`
- `APPLE_TEAM_ID`
- `APPLE_CERTIFICATE`
- `APPLE_CERTIFICATE_PASSWORD`
- `KEYCHAIN_PASSWORD`

If Apple certificate secrets are absent, macOS alpha builds are produced
unsigned. Public beta, release candidate, and stable builds should be signed and
notarized before being marked ready.

Optional Windows signing secrets:

- `WINDOWS_CERTIFICATE`
- `WINDOWS_CERTIFICATE_PASSWORD`

`WINDOWS_CERTIFICATE` is a base64-encoded PFX certificate. When it is absent,
Windows alpha builds are produced unsigned. When it is present, the release
workflow signs Windows MSI/EXE bundle artifacts with `signtool` and a SHA-256
timestamp. Prefer Azure Trusted Signing or another CI-safe signing service for
stable releases if long-lived PFX material is not acceptable.

## Stage Gates

Alpha proves feature parity with vid2dataset1, bundled FFmpeg packaging, and
public installability. Alpha releases are GitHub prereleases and may contain
known issues.

Required alpha checks:

- `cargo fmt --all --check`
- `cargo test --workspace --all-targets`
- `cargo run -p vid2dataset -- doctor`
- frontend `npm run build`
- packaged app opens on each target OS
- packaged app reports FFmpeg and ffprobe source as `bundled`
- small video smoke extraction writes image outputs plus JSONL/CSV manifests

Beta is feature freeze. Only fixes, polish, docs, migration cleanup, and
packaging hardening should land. Add install/uninstall smoke tests and complete
the manual OS test matrix before beta.

Release candidates are stable packaging rehearsals. Only blocker fixes should
land after the first RC. Final release notes and checksums must be ready.

Stable releases are non-prerelease GitHub Releases produced by the same
automation. Stable macOS and Windows public artifacts should be signed before
the release is published.

## Local Release Commands

For a local unsigned package with bundled FFmpeg:

```bash
scripts/setup-ffmpeg.sh /path/to/ffmpeg /path/to/ffprobe
scripts/package-local.sh
scripts/archive-release-artifacts.sh local
```

For a CI-like FFmpeg staging check from a pinned archive:

```bash
VID2DATASET_RELEASE_FFMPEG_ARCHIVE_URL=https://example.invalid/ffmpeg.zip \
VID2DATASET_RELEASE_FFMPEG_SHA256=<sha256> \
VID2DATASET_RELEASE_FFMPEG_SOURCE_URL=https://example.invalid/source \
VID2DATASET_RELEASE_FFMPEG_LICENSE_URL=https://ffmpeg.org/legal.html \
scripts/stage-release-ffmpeg.sh
```
