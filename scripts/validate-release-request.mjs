import { appendFileSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

function fail(message) {
  console.error(message);
  process.exit(1);
}

function git(args) {
  return execFileSync("git", args, { encoding: "utf8" }).trim();
}

const sourceEvent = process.env.SOURCE_EVENT;
const sourceRef = process.env.SOURCE_REF;
const sourceSha = process.env.SOURCE_SHA;
const defaultBranch = process.env.DEFAULT_BRANCH;

if (!/^[0-9a-f]{40,64}$/i.test(sourceSha ?? "")) {
  fail("The triggering workflow did not provide a valid source commit SHA.");
}
if (!defaultBranch || !sourceRef) {
  fail("The triggering workflow did not provide a source ref and default branch.");
}

let tag;
let releaseName;
let prerelease;
let draft;

if (sourceEvent === "push") {
  tag = sourceRef;
  releaseName = `vid2dataset2 ${tag}`;
  prerelease = /alpha|beta|rc/i.test(tag);
  draft = false;
} else if (sourceEvent === "workflow_dispatch") {
  if (sourceRef !== defaultBranch) {
    fail("Manual release requests must run from the repository default branch.");
  }

  let request;
  try {
    request = JSON.parse(readFileSync("release-request/request.json", "utf8"));
  } catch {
    fail("The manual release request metadata is missing or invalid JSON.");
  }
  tag = request.tag;
  releaseName = request.releaseName;
  prerelease = request.prerelease;
  draft = request.draft;
} else {
  fail(`Unsupported release request event: ${sourceEvent}`);
}

if (typeof tag !== "string" || !/^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(tag)) {
  fail("Release tags must use the vMAJOR.MINOR.PATCH format, with an optional prerelease or build suffix.");
}
try {
  git(["check-ref-format", `refs/tags/${tag}`]);
} catch {
  fail(`Invalid release tag: ${tag}`);
}
if (typeof releaseName !== "string" || !releaseName.trim() || releaseName.length > 255 || /[\r\n]/.test(releaseName)) {
  fail("The release title must be a non-empty, single-line value of at most 255 characters.");
}
if (typeof prerelease !== "boolean" || typeof draft !== "boolean") {
  fail("Release prerelease and draft settings must be boolean values.");
}

let tagSha;
try {
  git(["fetch", "--no-tags", "origin", `+refs/tags/${tag}:refs/remotes/release-request/target-tag`]);
  tagSha = git(["rev-parse", "--verify", "refs/remotes/release-request/target-tag^{commit}"]);
} catch {
  fail(`Release tag ${tag} does not exist in the repository.`);
}

if (sourceEvent === "push" && tagSha !== sourceSha) {
  fail("The pushed release tag no longer points to the commit that triggered the request.");
}

let defaultBranchRef;
try {
  git(["check-ref-format", `refs/heads/${defaultBranch}`]);
  defaultBranchRef = `refs/remotes/origin/${defaultBranch}`;
  git(["fetch", "--no-tags", "origin", `+refs/heads/${defaultBranch}:${defaultBranchRef}`]);
  git(["merge-base", "--is-ancestor", tagSha, defaultBranchRef]);
} catch {
  fail(`Release tag ${tag} is not reachable from the repository default branch.`);
}

if (sourceEvent === "workflow_dispatch") {
  try {
    git(["merge-base", "--is-ancestor", sourceSha, defaultBranchRef]);
  } catch {
    fail("The manual request commit is not reachable from the repository default branch.");
  }
}

const outputs = [
  `tag=${tag}`,
  `source_sha=${tagSha}`,
  `release_name=${releaseName}`,
  `prerelease=${prerelease}`,
  `draft=${draft}`,
].join("\n");
appendFileSync(process.env.GITHUB_OUTPUT, `${outputs}\n`);
