# Repository Guidelines

## Project Structure & Module Organization

This is a Rust workspace for `vid2dataset`, a video-to-image dataset tool. Core logic lives in `crates/vid2dataset-core/src`, including FFmpeg discovery, inspection, extraction, manifests, profiles, and config import. The CLI crate is `crates/vid2dataset-cli`, published as the `vid2dataset` binary. The desktop app is in `apps/desktop`: React/Vite code is under `apps/desktop/src`, and Tauri Rust handlers live in `apps/desktop/src-tauri/src`. Integration tests live in `crates/vid2dataset-core/tests`. Historical Python code is preserved in `legacy/`; avoid expanding it unless maintaining legacy behavior.

## Build, Test, and Development Commands

- `cargo fetch`: download Rust workspace dependencies.
- `cargo fmt --all --check`: verify Rust formatting.
- `cargo test --workspace --all-targets`: run Rust tests.
- `cargo run -p vid2dataset -- doctor`: validate FFmpeg/ffprobe discovery.
- `cargo run -p vid2dataset -- inspect path/to/video.mp4`: inspect a video through the CLI.
- `cd apps/desktop && npm ci`: install desktop dependencies from `package-lock.json`.
- `cd apps/desktop && npm run build`: type-check with `tsc` and build the Vite frontend.
- `cd apps/desktop && npm run dev`: start frontend-only Vite development on port `1420`.
- `cd apps/desktop && npm run tauri dev`: run the full desktop app.

## Coding Style & Naming Conventions

Use Rust 2021 style and keep code `rustfmt` clean. Keep CLI/Tauri layers thin by delegating domain behavior to `vid2dataset-core`. Use `snake_case` for Rust modules, functions, and variables; `PascalCase` for types and enum variants. Frontend code uses TypeScript + React; use component names in `PascalCase`, local variables in `camelCase`, and keep shared CSS in `apps/desktop/src/styles.css`.

## Testing Guidelines

Add Rust tests near changed behavior. Use `crates/*/tests` for integration coverage and inline unit tests for small pure functions. FFmpeg-dependent tests should skip gracefully when `ffmpeg` is unavailable. Run `cargo test --workspace --all-targets` before submitting. For desktop changes, also run `npm run build`.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, for example `Rename programs.` and `Add video frame extraction module with various processing options`. Keep subjects concise and behavior-focused. Pull requests should include a description, test commands run, linked issues when applicable, and screenshots or recordings for visible desktop UI changes.

## Security & Configuration Tips

Do not commit FFmpeg binaries or generated datasets. Stage local binaries under `apps/desktop/src-tauri/ffmpeg/bin/` with `scripts/setup-ffmpeg.sh`; the path is intended for local packaging resources. Prefer environment overrides such as `VID2DATASET_FFMPEG` and `VID2DATASET_FFPROBE` for development-specific tool paths.
