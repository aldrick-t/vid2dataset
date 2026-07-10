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
`ffprobe` binaries as Tauri app resources under `ffmpeg/bin/`. The project builds
these tools from cryptographically verified official FFmpeg source with pinned,
statically linked zlib for PNG output. GPL and nonfree FFmpeg components are
disabled; the built-in H.264, HEVC, AV1, VP9, MPEG-4, PNG, and JPEG support used
by vid2dataset remains enabled.

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

FFmpeg dependency metadata is committed in `.github/ffmpeg.lock.json`; no
repository variables are required. A `ready` lock records immutable archive and
redistributed-source URLs plus checksums for `macos-arm64`, `macos-x64`,
`windows-x64`, and `linux-x64`. Product releases fail closed while the lock is
`bootstrap_required` or any target is missing.

The dependency workflow is `.github/workflows/ffmpeg-dependency.yml`. It checks
for numbered stable FFmpeg and zlib releases quarterly at 09:00 UTC on January,
April, July, and October 1, and it supports manual dispatch for urgent updates.
It performs the following sequence:

1. Resolve release metadata only from `ffmpeg.org` and `zlib.net`.
2. Verify source checksums, signatures, and pinned signing-key fingerprints.
3. Build and smoke-test LGPL binaries on the four release targets.
4. Upload binaries, exact sources, signatures, notices, and checksums to an
   immutable draft dependency release.
5. Open a review PR containing the completed lock manifest.
6. Promote the draft to a GitHub prerelease only after the lock PR merges.

Accepted dependency releases are retained permanently. A rebuild of the same
FFmpeg version increments `rN`; existing assets are never replaced.

### First Bootstrap

The committed bootstrap sources are FFmpeg `8.1.2` and zlib `1.3.2`. Their
source records are verified, but the four binary checksums are intentionally
absent until GitHub's native runners build them.

One repository setting is required: under **Settings > Actions > General >
Workflow permissions**, enable **Allow GitHub Actions to create and approve pull
requests**. Then run **FFmpeg dependency** manually from `stage/vid2dataset2`
with `base_ref` set to `stage/vid2dataset2`. Review and merge the generated lock
PR. That merge publishes the dependency prerelease and changes the lock to
`ready`, after which normal product releases need no FFmpeg setup.

Scheduled checks run only from GitHub's default branch. They begin operating
quarterly after vid2dataset2 is merged into `main`.

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
VID2DATASET_RELEASE_FFMPEG_ARCHIVE_FORMAT=zip \
VID2DATASET_RELEASE_FFMPEG_SOURCE_URL=https://example.invalid/source \
VID2DATASET_RELEASE_FFMPEG_LICENSE_URL=https://ffmpeg.org/legal.html \
scripts/stage-release-ffmpeg.sh
```

To stage the accepted dependency exactly as a public release does:

```bash
scripts/stage-release-ffmpeg.sh --target macos-arm64
```
