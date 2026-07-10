import assert from "node:assert/strict";
import test from "node:test";

import {
  buildFinalLock,
  compareVersions,
  finalizeSources,
  parseFfmpegReleasePage,
  parseZlibReleasePage,
  resolveCandidate,
} from "../lib/ffmpeg-dependency.mjs";

const lock = {
  schema_version: 1,
  status: "ready",
  ffmpeg: {
    version: "8.1.1",
    source_url: "https://ffmpeg.org/releases/ffmpeg-8.1.1.tar.xz",
    signature_url: "https://ffmpeg.org/releases/ffmpeg-8.1.1.tar.xz.asc",
    source_sha256: "a".repeat(64),
    signing_key_fingerprint: "A".repeat(40),
    license: "LGPL-2.1-or-later",
    license_url: "https://example.test/ffmpeg-license",
  },
  zlib: {
    version: "1.3.1",
    source_url: "https://zlib.net/zlib-1.3.1.tar.gz",
    signature_url: "https://zlib.net/zlib-1.3.1.tar.gz.asc",
    source_sha256: "b".repeat(64),
    signing_key_fingerprint: "B".repeat(40),
    license: "Zlib",
    license_url: "https://example.test/zlib-license",
  },
  build: { revision: 2, recipe_version: 1, recipe_sha256: "c".repeat(64), configure_flags: ["--disable-gpl"] },
  release: { tag: "ffmpeg-8.1.1-r2", prerelease: true },
  artifacts: {},
};

test("parses current stable releases from official page shapes", () => {
  assert.equal(
    parseFfmpegReleasePage('<a href="releases/ffmpeg-8.1.2.tar.xz">Download Source Code ffmpeg-8.1.2.tar.xz</a>').version,
    "8.1.2",
  );
  const zlib = parseZlibReleasePage(
    `zlib source code, version 1.3.2, tar.gz format (1468K, SHA-256 hash <code>${"d".repeat(64)}</code>)`,
  );
  assert.equal(zlib.version, "1.3.2");
  assert.equal(zlib.sha256, "d".repeat(64));
});

test("rejects malformed release pages", () => {
  assert.throws(() => parseFfmpegReleasePage("nightly snapshot"));
  assert.throws(() => parseZlibReleasePage("zlib current"));
});

test("compares numeric versions instead of lexicographic versions", () => {
  assert.equal(compareVersions("8.10.0", "8.9.9"), 1);
  assert.equal(compareVersions("8.1", "8.1.0"), 0);
});

test("new FFmpeg starts build revision one", () => {
  const candidate = resolveCandidate(lock, {
    ffmpeg: { version: "8.1.2", url: "https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz", signature_url: "https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz.asc" },
    zlib: { version: "1.3.1", sha256: "b".repeat(64), url: "https://zlib.net/zlib-1.3.1.tar.gz", signature_url: "https://zlib.net/zlib-1.3.1.tar.gz.asc" },
  });
  assert.equal(candidate.build.revision, 1);
  assert.equal(candidate.release.tag, "ffmpeg-8.1.2-r1");
  assert.equal(candidate.ffmpeg.source_sha256, null);
});

test("zlib-only updates increment the build revision", () => {
  const candidate = resolveCandidate(lock, {
    ffmpeg: { version: "8.1.1", url: lock.ffmpeg.source_url, signature_url: lock.ffmpeg.signature_url },
    zlib: { version: "1.3.2", sha256: "d".repeat(64), url: "https://zlib.net/zlib-1.3.2.tar.gz", signature_url: "https://zlib.net/zlib-1.3.2.tar.gz.asc" },
  });
  assert.equal(candidate.build.revision, 3);
  assert.equal(candidate.release.tag, "ffmpeg-8.1.1-r3");
});

test("unchanged versions do not trigger an update", () => {
  const candidate = resolveCandidate(lock, {
    ffmpeg: { version: "8.1.1", url: lock.ffmpeg.source_url, signature_url: lock.ffmpeg.signature_url },
    zlib: { version: "1.3.1", sha256: "b".repeat(64), url: lock.zlib.source_url, signature_url: lock.zlib.signature_url },
  });
  assert.equal(candidate.update_needed, false);
});

test("a changed build recipe creates a new revision", () => {
  const candidate = resolveCandidate(lock, {
    ffmpeg: { version: "8.1.1", url: lock.ffmpeg.source_url, signature_url: lock.ffmpeg.signature_url },
    zlib: { version: "1.3.1", sha256: "b".repeat(64), url: lock.zlib.source_url, signature_url: lock.zlib.signature_url },
  }, { recipeSha256: "e".repeat(64) });
  assert.equal(candidate.update_needed, true);
  assert.equal(candidate.build.revision, 3);
});

test("bootstrap keeps its declared first build revision", () => {
  const bootstrapLock = { ...lock, status: "bootstrap_required" };
  const candidate = resolveCandidate(bootstrapLock, {
    ffmpeg: { version: "8.1.1", url: lock.ffmpeg.source_url, signature_url: lock.ffmpeg.signature_url },
    zlib: { version: "1.3.1", sha256: "b".repeat(64), url: lock.zlib.source_url, signature_url: lock.zlib.signature_url },
  });
  assert.equal(candidate.build.revision, 2);
  assert.equal(candidate.bootstrap, true);
  assert.equal(candidate.update_needed, true);
});

test("downgrades are rejected", () => {
  assert.throws(() => resolveCandidate(lock, {
    ffmpeg: { version: "8.0.3", url: "https://ffmpeg.org/releases/ffmpeg-8.0.3.tar.xz", signature_url: "https://ffmpeg.org/releases/ffmpeg-8.0.3.tar.xz.asc" },
    zlib: { version: "1.3.1", sha256: "b".repeat(64), url: lock.zlib.source_url, signature_url: lock.zlib.signature_url },
  }), /downgrade/);
});

test("source finalization rejects checksum and signing fingerprint failures", () => {
  const candidate = resolveCandidate(lock, {
    ffmpeg: { version: "8.1.1", url: lock.ffmpeg.source_url, signature_url: lock.ffmpeg.signature_url },
    zlib: { version: "1.3.1", sha256: "b".repeat(64), url: lock.zlib.source_url, signature_url: lock.zlib.signature_url },
  }, { force: true });
  assert.throws(
    () => finalizeSources(candidate, "a".repeat(64), "c".repeat(64), { ffmpeg: "A".repeat(40), zlib: "B".repeat(40) }),
    /zlib checksum mismatch/,
  );
  assert.throws(
    () => finalizeSources(candidate, "a".repeat(64), "b".repeat(64), { ffmpeg: "C".repeat(40), zlib: "B".repeat(40) }),
    /FFmpeg signing-key fingerprint mismatch/,
  );
});

test("completed locks require all four immutable target artifacts", () => {
  const candidate = { ...lock, status: "candidate", release: { tag: "ffmpeg-8.1.1-r2", prerelease: true } };
  const manifests = [
    ["macos-arm64", "aarch64-apple-darwin", ""],
    ["macos-x64", "x86_64-apple-darwin", ""],
    ["windows-x64", "x86_64-pc-windows-gnu", ".exe"],
    ["linux-x64", "x86_64-unknown-linux-gnu", ""],
  ].map(([target, architecture, suffix]) => ({
    target,
    architecture,
    archive_name: `${target}.tar.gz`,
    archive_format: "tar.gz",
    sha256: target[0].charCodeAt(0).toString(16).padStart(2, "0").repeat(32),
    ffmpeg_path: `bin/ffmpeg${suffix}`,
    ffprobe_path: `bin/ffprobe${suffix}`,
  }));
  const finalLock = buildFinalLock(candidate, manifests, "aldrick-t/vid2dataset");
  assert.equal(finalLock.status, "ready");
  assert.match(finalLock.artifacts["windows-x64"].url, /windows-x64\.tar\.gz$/);
  assert.match(finalLock.ffmpeg.redistribution_source_url, /ffmpeg-source\.tar\.xz$/);
});
