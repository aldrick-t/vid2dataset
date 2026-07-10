#!/usr/bin/env node

import { appendFile, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  buildFinalLock,
  finalizeSources,
  parseFfmpegReleasePage,
  parseZlibReleasePage,
  recipeSha256,
  resolveCandidate,
  sha256File,
  targetTsv,
  validateLock,
} from "./lib/ffmpeg-dependency.mjs";

const [command, ...rawArgs] = process.argv.slice(2);
const args = parseArgs(rawArgs);

try {
  switch (command) {
    case "resolve":
      await resolve();
      break;
    case "finalize-sources":
      await finalize();
      break;
    case "make-lock":
      await makeLock();
      break;
    case "validate-lock":
      await validate();
      break;
    case "target-tsv":
      await printTarget();
      break;
    case "artifact-manifest":
      await artifactManifest();
      break;
    case "verify-assets":
      await verifyAssets();
      break;
    default:
      throw new Error("usage: ffmpeg-dependency.mjs <resolve|finalize-sources|make-lock|validate-lock|target-tsv|artifact-manifest|verify-assets> [options]");
  }
} catch (error) {
  console.error(`ffmpeg-dependency: ${error.message}`);
  process.exitCode = 1;
}

async function resolve() {
  const lockPath = required("lock");
  const output = required("output");
  const root = path.resolve(args.root ?? path.join(path.dirname(new URL(import.meta.url).pathname), ".."));
  const lock = JSON.parse(await readFile(lockPath, "utf8"));
  const [ffmpegResponse, zlibResponse] = await Promise.all([
    fetch("https://ffmpeg.org/download.html"),
    fetch("https://zlib.net/"),
  ]);
  if (!ffmpegResponse.ok || !zlibResponse.ok) throw new Error("failed to fetch official dependency release pages");
  const latest = {
    ffmpeg: parseFfmpegReleasePage(await ffmpegResponse.text()),
    zlib: parseZlibReleasePage(await zlibResponse.text()),
  };
  const candidate = resolveCandidate(lock, latest, {
    force: args.force === "true",
    buildRevision: args["build-revision"],
    recipeSha256: await recipeSha256(root),
  });
  await writeJson(output, candidate);
  await githubOutput({
    update_needed: String(candidate.update_needed),
    ffmpeg_version: candidate.ffmpeg.version,
    zlib_version: candidate.zlib.version,
    release_tag: candidate.release.tag,
    bootstrap: String(candidate.bootstrap),
  });
}

async function finalize() {
  const candidate = JSON.parse(await readFile(required("candidate"), "utf8"));
  const result = finalizeSources(candidate, required("ffmpeg-sha256"), required("zlib-sha256"), {
    ffmpeg: required("ffmpeg-fingerprint"),
    zlib: required("zlib-fingerprint"),
  });
  await writeJson(required("output"), result);
}

async function makeLock() {
  const candidate = JSON.parse(await readFile(required("candidate"), "utf8"));
  const directory = required("manifests-dir");
  const manifests = [];
  for (const entry of await readdir(directory)) {
    if (entry.endsWith(".json")) manifests.push(JSON.parse(await readFile(path.join(directory, entry), "utf8")));
  }
  const lock = buildFinalLock(candidate, manifests, required("repository"));
  await writeJson(required("output"), lock);
}

async function validate() {
  const lock = JSON.parse(await readFile(required("lock"), "utf8"));
  validateLock(lock, { requireReady: args["require-ready"] === "true" });
  console.log(`${lock.ffmpeg.version} / zlib ${lock.zlib.version} (${lock.status})`);
}

async function printTarget() {
  const lock = JSON.parse(await readFile(required("lock"), "utf8"));
  console.log(targetTsv(lock, required("target")));
}

async function artifactManifest() {
  const target = required("target");
  const archive = required("archive");
  const windows = target.startsWith("windows-");
  await writeJson(required("output"), {
    target,
    archive_name: path.basename(archive),
    archive_format: "tar.gz",
    sha256: await sha256File(archive),
    architecture: required("architecture"),
    ffmpeg_path: `bin/ffmpeg${windows ? ".exe" : ""}`,
    ffprobe_path: `bin/ffprobe${windows ? ".exe" : ""}`,
  });
}

async function verifyAssets() {
  const lock = JSON.parse(await readFile(required("lock"), "utf8"));
  validateLock(lock, { requireReady: true });
  const directory = required("assets-dir");
  for (const artifact of Object.values(lock.artifacts)) {
    const file = path.join(directory, path.basename(new URL(artifact.url).pathname));
    const actual = await sha256File(file);
    if (actual !== artifact.sha256) throw new Error(`asset checksum mismatch for ${path.basename(file)}`);
  }
}

function parseArgs(values) {
  const result = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith("--")) throw new Error(`unexpected argument: ${value}`);
    const key = value.slice(2);
    const next = values[index + 1];
    if (!next || next.startsWith("--")) result[key] = "true";
    else {
      result[key] = next;
      index += 1;
    }
  }
  return result;
}

function required(name) {
  if (!args[name]) throw new Error(`--${name} is required`);
  return args[name];
}

async function writeJson(file, value) {
  await mkdir(path.dirname(path.resolve(file)), { recursive: true });
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`);
}

async function githubOutput(values) {
  if (!process.env.GITHUB_OUTPUT) return;
  await appendFile(process.env.GITHUB_OUTPUT, Object.entries(values).map(([key, value]) => `${key}=${value}\n`).join(""));
}
