# vid2dataset2 Release Flow

## Nomenclature and Versioning

`vid2dataset` remains the product name, CLI binary name, and desktop app name.
`vid2dataset2` names the Rust + Tauri revamp release train. The legacy Python
implementation remains under `legacy/` for reference and migration checks.

The current prepared release is:

- Product release title: `vid2dataset2 Alpha 4`
- Machine version: `0.2.0-alpha.4`
- Git tag: `v0.2.0-alpha.4`

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

- macOS ARM64: DMG installer plus zipped `.app` where Tauri produces it.
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

Release requests are recorded by `.github/workflows/release-request.yml`; the
privileged build and publish workflow is `.github/workflows/release.yml`. The
request workflow carries no signing secrets. The release workflow runs from the
default branch and builds the exact commit named by the release tag. Stable tags
must point to the current default-branch commit; prerelease tags may point to
any commit still reachable from that branch so failed prerelease builds can be
retried after workflow fixes.

Pushing a `v*` tag starts the release build automatically and creates a draft
GitHub Release for review. Do not manually dispatch the same tag after pushing
it, since the tag push already started a run. Use **Release Request** manually
from the repository default branch only to retry an existing tag whose earlier
run did not complete. Manual dispatch from another branch is ignored.
For a stable release, the tag must identify the reviewed current default-branch
commit. Prerelease tags must remain reachable from the default branch.

Manual requests accept:

- `tag`: defaults to `v0.2.0-alpha.4`
- `release_name`: defaults to `vid2dataset2 Alpha 4`
- `draft`: defaults to `true`

### Alpha 4 release checklist

1. Merge the Alpha 4 version bump and release notes to `main`, then update the
   local branch with `git pull --ff-only origin main`. Confirm the worktree is
   clean and that `main` contains the exact commit you intend to release.
2. Run the required alpha checks: `cargo fmt --all --check`,
   `cargo test --workspace --all-targets`, `cargo run -p vid2dataset -- doctor`,
   and `npm ci && npm run build` from `apps/desktop`.
3. Create and push the tag on the reviewed `main` commit:

   ```bash
   git tag -a v0.2.0-alpha.4 -m "vid2dataset2 Alpha 4"
   git push origin v0.2.0-alpha.4
   ```

4. The tag push starts **Release Request**, followed by **Release**. Monitor
   both runs. The release validates
   the tag, runs workspace/frontend checks, builds macOS ARM64, Windows
   x64, and Linux x64 packages, and uploads SHA-256 checksums.
5. Review the draft notes and platform artifacts, verify the checksums, and
   smoke-test installers on the target operating systems. Publish the draft
   from GitHub Releases when it is ready.

If the automatic run fails, retry the existing tag by running **Release
Request** manually on `main` with tag `v0.2.0-alpha.4`, release name
`vid2dataset2 Alpha 4`, and draft `true`.

Alpha 4's FFmpeg dependency lock is already `ready`; the dependency bootstrap
workflow is not needed for this release.

FFmpeg dependency metadata is committed in `.github/ffmpeg.lock.json`; no
repository variables are required. A `ready` lock records immutable archive and
redistributed-source URLs plus checksums for `macos-arm64`, `windows-x64`,
and `linux-x64`. Product releases fail closed while the lock is
`bootstrap_required` or any target is missing.

The dependency workflow is `.github/workflows/ffmpeg-dependency.yml`. It checks
for numbered stable FFmpeg and zlib releases quarterly at 09:00 UTC on January,
April, July, and October 1, and it supports manual dispatch for urgent updates.
It performs the following sequence:

1. Resolve release metadata only from `ffmpeg.org` and `zlib.net`.
2. Verify source checksums, signatures, and pinned signing-key fingerprints.
3. Build and smoke-test LGPL binaries on macOS ARM64, Windows x64, and Linux x64.
4. Upload binaries, exact sources, signatures, notices, and checksums to an
   immutable draft dependency release.
5. Open a review PR containing the completed lock manifest.
6. Promote the draft to a GitHub prerelease only after the lock PR merges.

Accepted dependency releases are retained permanently. A rebuild of the same
FFmpeg version increments `rN`; existing assets are never replaced.

### FFmpeg Dependency Workflow Configuration

The current ready lock records FFmpeg `8.1.2` and zlib `1.3.2`, with verified
source records and checksums for the three supported targets. Alpha 4 needs no
dependency bootstrap. For future updates, scheduled checks run from the default
branch; manual dispatch can select the branch that receives the generated lock
pull request.

GitHub Actions must have permission to create pull requests under **Settings >
Actions > General > Workflow permissions** for automatic dependency update
pull requests.

Stable macOS releases require these signing secrets:

- `APPLE_CERTIFICATE`
- `APPLE_CERTIFICATE_PASSWORD`
- `KEYCHAIN_PASSWORD`

macOS alpha, beta, and release-candidate builds use ad-hoc signatures and do
not use Apple signing credentials. Stable builds fail before packaging if the
certificate or identity is unavailable. Notarization is optional: set
`APPLE_ID`, `APPLE_PASSWORD`, and `APPLE_TEAM_ID` together for stable builds.
Leaving all three unset skips notarization; a partial configuration fails with
a clear error.

Stable Windows releases require these signing secrets:

- `WINDOWS_CERTIFICATE`
- `WINDOWS_CERTIFICATE_PASSWORD`

`WINDOWS_CERTIFICATE` is a base64-encoded PFX certificate. Alpha, beta, and
release-candidate Windows artifacts are unsigned. Stable Windows builds fail
before packaging if the certificate is unavailable, then sign and verify MSI
and EXE artifacts with `signtool` and a SHA-256 timestamp.

## Stage Gates

Alpha proves feature parity with vid2dataset1, bundled FFmpeg packaging, and
public installability. Alpha releases are GitHub prereleases and may contain
known issues. macOS alpha builds use ad-hoc signatures; Windows alpha builds
are unsigned.

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
the manual OS test matrix before beta. Beta macOS builds use ad-hoc signatures;
Windows beta builds are unsigned.

Release candidates are stable packaging rehearsals. Only blocker fixes should
land after the first RC. Final release notes and checksums must be ready.
Release-candidate macOS builds use ad-hoc signatures; Windows builds are
unsigned.

Stable releases are non-prerelease GitHub Releases produced by the same
automation. Stable macOS and Windows builds must pass signature verification
before the release can be published.

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
