import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

export const LOCK_SCHEMA_VERSION = 1;
export const TARGETS = ["macos-arm64", "macos-x64", "windows-x64", "linux-x64"];

const VERSION_PATTERN = /^[0-9]+\.[0-9]+(?:\.[0-9]+)?$/;
const SHA256_PATTERN = /^[a-f0-9]{64}$/;

export function parseVersion(value) {
  if (!VERSION_PATTERN.test(value)) {
    throw new Error(`invalid stable version: ${value}`);
  }
  return value.split(".").map(Number);
}

export function compareVersions(left, right) {
  const a = parseVersion(left);
  const b = parseVersion(right);
  const length = Math.max(a.length, b.length);
  for (let index = 0; index < length; index += 1) {
    const difference = (a[index] ?? 0) - (b[index] ?? 0);
    if (difference !== 0) return Math.sign(difference);
  }
  return 0;
}

export function parseFfmpegReleasePage(html) {
  const plainText = html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ");
  const match = plainText.match(
    /Download Source Code\s+ffmpeg-([0-9]+\.[0-9]+(?:\.[0-9]+)?)\.tar\.xz/i,
  );
  if (!match) throw new Error("official FFmpeg page did not expose one current stable source release");
  const version = match[1];
  parseVersion(version);
  return {
    version,
    url: `https://ffmpeg.org/releases/ffmpeg-${version}.tar.xz`,
    signature_url: `https://ffmpeg.org/releases/ffmpeg-${version}.tar.xz.asc`,
  };
}

export function parseZlibReleasePage(html) {
  const plainText = html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ");
  const match = plainText.match(
    /zlib source code, version ([0-9]+\.[0-9]+(?:\.[0-9]+)?), tar\.gz format.*?SHA-256 hash\s*([a-f0-9]{64})/i,
  );
  if (!match) throw new Error("official zlib page did not expose one current stable tar.gz release and checksum");
  const version = match[1];
  parseVersion(version);
  return {
    version,
    sha256: assertSha256(match[2]),
    url: `https://zlib.net/zlib-${version}.tar.gz`,
    signature_url: `https://zlib.net/zlib-${version}.tar.gz.asc`,
  };
}

export function resolveCandidate(lock, latest, options = {}) {
  validateLock(lock, { requireReady: false });
  const ffmpegComparison = compareVersions(latest.ffmpeg.version, lock.ffmpeg.version);
  const zlibComparison = compareVersions(latest.zlib.version, lock.zlib.version);
  if (ffmpegComparison < 0 || zlibComparison < 0) {
    throw new Error("upstream resolution attempted to downgrade a dependency");
  }

  const ffmpegChanged = ffmpegComparison > 0;
  const zlibChanged = zlibComparison > 0;
  const recipeChanged = Boolean(options.recipeSha256) && options.recipeSha256 !== lock.build.recipe_sha256;
  const updateNeeded = ffmpegChanged || zlibChanged || recipeChanged || Boolean(options.force) || lock.status !== "ready";
  const requestedRevision = options.buildRevision ? Number(options.buildRevision) : null;
  if (requestedRevision !== null && (!Number.isInteger(requestedRevision) || requestedRevision < 1)) {
    throw new Error("build revision must be a positive integer");
  }
  const revision = requestedRevision ?? (
    ffmpegChanged
      ? 1
      : lock.status !== "ready"
        ? lock.build.revision
        : lock.build.revision + (updateNeeded ? 1 : 0)
  );
  const ffmpegVersion = latest.ffmpeg.version;
  const zlibVersion = latest.zlib.version;

  return {
    schema_version: LOCK_SCHEMA_VERSION,
    status: updateNeeded ? "candidate" : "unchanged",
    update_needed: updateNeeded,
    bootstrap: lock.status !== "ready",
    ffmpeg: {
      ...lock.ffmpeg,
      version: ffmpegVersion,
      source_url: latest.ffmpeg.url,
      signature_url: latest.ffmpeg.signature_url,
      source_sha256: ffmpegChanged ? null : lock.ffmpeg.source_sha256,
      license_url: `https://github.com/FFmpeg/FFmpeg/blob/n${ffmpegVersion}/COPYING.LGPLv2.1`,
    },
    zlib: {
      ...lock.zlib,
      version: zlibVersion,
      source_url: latest.zlib.url,
      signature_url: latest.zlib.signature_url,
      source_sha256: latest.zlib.sha256,
      license_url: `https://github.com/madler/zlib/blob/v${zlibVersion}/LICENSE`,
    },
    build: {
      ...lock.build,
      revision,
      recipe_sha256: options.recipeSha256 ?? lock.build.recipe_sha256,
    },
    release: {
      tag: `ffmpeg-${ffmpegVersion}-r${revision}`,
      prerelease: true,
    },
    artifacts: {},
  };
}

export function finalizeSources(candidate, ffmpegSha256, zlibSha256, fingerprints) {
  const result = structuredClone(candidate);
  result.ffmpeg.source_sha256 = assertSha256(ffmpegSha256);
  const actualZlibSha = assertSha256(zlibSha256);
  if (actualZlibSha !== result.zlib.source_sha256) {
    throw new Error(`zlib checksum mismatch: expected ${result.zlib.source_sha256}, got ${actualZlibSha}`);
  }
  assertFingerprint(fingerprints?.ffmpeg, result.ffmpeg.signing_key_fingerprint, "FFmpeg");
  assertFingerprint(fingerprints?.zlib, result.zlib.signing_key_fingerprint, "zlib");
  return result;
}

export function buildFinalLock(candidate, manifests, repository) {
  if (candidate.status !== "candidate") throw new Error("candidate metadata is not buildable");
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)) {
    throw new Error(`invalid GitHub repository: ${repository}`);
  }
  const byTarget = Object.fromEntries(manifests.map((manifest) => [manifest.target, manifest]));
  for (const target of TARGETS) {
    if (!byTarget[target]) throw new Error(`missing artifact manifest for ${target}`);
  }
  const tag = candidate.release.tag;
  const baseUrl = `https://github.com/${repository}/releases/download/${tag}`;
  const artifacts = {};
  for (const target of TARGETS) {
    const manifest = byTarget[target];
    artifacts[target] = {
      url: `${baseUrl}/${encodeURIComponent(manifest.archive_name)}`,
      sha256: assertSha256(manifest.sha256),
      archive_format: manifest.archive_format,
      architecture: manifest.architecture,
      ffmpeg_path: manifest.ffmpeg_path,
      ffprobe_path: manifest.ffprobe_path,
    };
  }
  return {
    ...candidate,
    status: "ready",
    update_needed: undefined,
    bootstrap: undefined,
    ffmpeg: {
      ...candidate.ffmpeg,
      redistribution_source_url: `${baseUrl}/ffmpeg-source.tar.xz`,
    },
    zlib: {
      ...candidate.zlib,
      redistribution_source_url: `${baseUrl}/zlib-source.tar.gz`,
    },
    release: {
      ...candidate.release,
      url: `https://github.com/${repository}/releases/tag/${tag}`,
    },
    artifacts,
  };
}

export function validateLock(lock, options = {}) {
  if (lock.schema_version !== LOCK_SCHEMA_VERSION) throw new Error("unsupported FFmpeg lock schema");
  if (!lock.ffmpeg || !lock.zlib || !lock.build || !lock.release || !lock.artifacts) {
    throw new Error("FFmpeg lock is missing required sections");
  }
  if (!["bootstrap_required", "ready"].includes(lock.status)) throw new Error(`invalid lock status: ${lock.status}`);
  parseVersion(lock.ffmpeg.version);
  parseVersion(lock.zlib.version);
  assertHttpsUrl(lock.ffmpeg.source_url, "ffmpeg.org");
  assertHttpsUrl(lock.ffmpeg.signature_url, "ffmpeg.org");
  assertHttpsUrl(lock.zlib.source_url, "zlib.net");
  assertHttpsUrl(lock.zlib.signature_url, "zlib.net");
  if (lock.ffmpeg.source_sha256) assertSha256(lock.ffmpeg.source_sha256);
  if (lock.zlib.source_sha256) assertSha256(lock.zlib.source_sha256);
  if (!Number.isInteger(lock.build.revision) || lock.build.revision < 1) {
    throw new Error("invalid build revision");
  }
  if (!Array.isArray(lock.build.configure_flags) || lock.build.configure_flags.length === 0) {
    throw new Error("configure flags are required");
  }
  const expectedTag = `ffmpeg-${lock.ffmpeg.version}-r${lock.build.revision}`;
  if (lock.release.tag !== expectedTag) throw new Error(`dependency release tag must be ${expectedTag}`);
  if (options.requireReady) {
    if (lock.status !== "ready") throw new Error("FFmpeg dependency bootstrap has not completed");
    assertHttpsUrl(lock.ffmpeg.redistribution_source_url, "github.com");
    assertHttpsUrl(lock.zlib.redistribution_source_url, "github.com");
    for (const target of TARGETS) validateArtifact(lock.artifacts[target], target, lock.release.tag);
  }
  return lock;
}

export function targetTsv(lock, target) {
  validateLock(lock, { requireReady: true });
  if (!TARGETS.includes(target)) throw new Error(`unsupported FFmpeg target: ${target}`);
  const artifact = lock.artifacts[target];
  return [
    ["ARCHIVE_URL", artifact.url],
    ["ARCHIVE_SHA256", artifact.sha256],
    ["ARCHIVE_FORMAT", artifact.archive_format],
    ["FFMPEG_MEMBER", artifact.ffmpeg_path],
    ["FFPROBE_MEMBER", artifact.ffprobe_path],
    ["FFMPEG_VERSION", lock.ffmpeg.version],
    ["FFMPEG_SOURCE_URL", lock.ffmpeg.source_url],
    ["FFMPEG_REDISTRIBUTION_SOURCE_URL", lock.ffmpeg.redistribution_source_url],
    ["FFMPEG_SOURCE_SHA256", lock.ffmpeg.source_sha256],
    ["FFMPEG_SIGNATURE_URL", lock.ffmpeg.signature_url],
    ["FFMPEG_SIGNING_FINGERPRINT", lock.ffmpeg.signing_key_fingerprint],
    ["FFMPEG_LICENSE", lock.ffmpeg.license],
    ["FFMPEG_LICENSE_URL", lock.ffmpeg.license_url],
    ["ZLIB_VERSION", lock.zlib.version],
    ["ZLIB_SOURCE_URL", lock.zlib.source_url],
    ["ZLIB_REDISTRIBUTION_SOURCE_URL", lock.zlib.redistribution_source_url],
    ["ZLIB_SOURCE_SHA256", lock.zlib.source_sha256],
    ["ZLIB_LICENSE", lock.zlib.license],
    ["ZLIB_LICENSE_URL", lock.zlib.license_url],
    ["DEPENDENCY_RELEASE_TAG", lock.release.tag],
  ]
    .map(([key, value]) => `${key}\t${value}`)
    .join("\n");
}

export async function sha256File(filePath) {
  const hash = createHash("sha256");
  hash.update(await readFile(filePath));
  return hash.digest("hex");
}

export async function recipeSha256(rootDir) {
  const hash = createHash("sha256");
  for (const relative of [
    "scripts/build-release-ffmpeg.sh",
    "scripts/ffmpeg-dependency.mjs",
    "scripts/lib/ffmpeg-dependency.mjs",
    "scripts/verify-release-sources.sh",
    ".github/workflows/ffmpeg-dependency.yml",
  ]) {
    hash.update(relative);
    hash.update(await readFile(path.join(rootDir, relative)));
  }
  return hash.digest("hex");
}

function validateArtifact(artifact, target, releaseTag) {
  if (!artifact) throw new Error(`missing FFmpeg artifact for ${target}`);
  assertHttpsUrl(artifact.url, "github.com");
  if (!new URL(artifact.url).pathname.includes(`/releases/download/${releaseTag}/`)) {
    throw new Error(`artifact URL for ${target} does not use the locked release tag`);
  }
  assertSha256(artifact.sha256);
  if (artifact.archive_format !== "tar.gz") throw new Error(`unsupported archive format for ${target}`);
  const windows = target.startsWith("windows-");
  const suffix = windows ? ".exe" : "";
  if (artifact.ffmpeg_path !== `bin/ffmpeg${suffix}` || artifact.ffprobe_path !== `bin/ffprobe${suffix}`) {
    throw new Error(`unexpected executable paths for ${target}`);
  }
}

function assertSha256(value) {
  const normalized = String(value).toLowerCase();
  if (!SHA256_PATTERN.test(normalized)) throw new Error(`invalid SHA-256: ${value}`);
  return normalized;
}

function assertFingerprint(actual, expected, name) {
  const normalizedActual = String(actual ?? "").replaceAll(" ", "").toUpperCase();
  const normalizedExpected = String(expected ?? "").replaceAll(" ", "").toUpperCase();
  if (!/^[A-F0-9]{40}$/.test(normalizedExpected) || normalizedActual !== normalizedExpected) {
    throw new Error(`${name} signing-key fingerprint mismatch`);
  }
}

function assertHttpsUrl(value, expectedHost) {
  const url = new URL(value);
  if (url.protocol !== "https:" || url.hostname !== expectedHost) {
    throw new Error(`unexpected dependency URL: ${value}`);
  }
}
