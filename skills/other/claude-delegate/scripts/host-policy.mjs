/**
 * delegate-skills · claude-delegate · host-policy.mjs
 *
 * Resolve the relay's effective shell boundary from an optional, user-owned
 * host policy. No policy keeps Claude's normal shell sandbox. A valid policy
 * can select outer-container isolation, meaning Claude's inner shell sandbox is
 * disabled, and only inside a Linux container proven by filesystem evidence.
 * Every other policy state fails closed: it throws, and the relay refuses to
 * run.
 *
 * Kept free of process-global reads where practical so tests can inject host
 * facts instead of calling the real CLI.
 */

import { existsSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

export const HOST_POLICY_SCHEMA = "claude-delegate.host-policy.v1";
const POLICY_FIELDS = new Set(["schema", "shellSandbox", "platform", "requireContainer", "approval"]);
const MAX_POLICY_BYTES = 65_536;
// Host paths a container runtime bind-mounts *into* a container. Inside a
// container, /proc/1/mountinfo shows them as a mount's root (field 4) on an
// ordinary mount point such as /etc/hosts. On the host itself they appear as
// mount points under the same tree, which the check below excludes.
const RUNTIME_ROOTS = [
  "/var/lib/kubelet/pods/",
  "/var/lib/docker/containers/",
  "/var/lib/containerd/",
  "/var/lib/containers/",
];

export class HostPolicyError extends Error {}

export function codexHome(env = process.env, home = homedir()) {
  return env.CODEX_HOME ? resolve(env.CODEX_HOME) : join(home, ".codex");
}

export function hostPolicyPath(env = process.env, home = homedir()) {
  return join(codexHome(env, home), "claude-delegate", "host-policy.json");
}

function runtimeBindMount(mountInfo) {
  for (const line of mountInfo.split("\n")) {
    const fields = line.split(" ");
    if (fields.length < 5) continue;
    const [, , , root, mountPoint] = fields;
    const source = RUNTIME_ROOTS.find((prefix) => root.startsWith(prefix));
    if (source && !RUNTIME_ROOTS.some((prefix) => mountPoint.startsWith(prefix))) {
      return `/proc/1/mountinfo: ${mountPoint} is bound from ${source}`;
    }
  }
  return null;
}

/**
 * `detected` is the long-standing heuristic that turns on Claude's weaker
 * nested sandbox settings. `verified` is the stricter test that a policy needs
 * before it may disable the inner sandbox: a runtime marker file or a
 * runtime bind mount on PID 1. Environment variables alone never verify.
 */
export function detectContainer({
  platform = process.platform,
  env = process.env,
  exists = existsSync,
  readText = (path) => readFileSync(path, "utf8"),
} = {}) {
  if (platform === "win32") return { detected: false, verified: false, evidence: null };
  const envSignal = ["KUBERNETES_SERVICE_HOST", "CONTAINER"].find((name) => env[name]);
  const marker = ["/.dockerenv", "/run/.containerenv"].find((path) => exists(path));
  let mountInfo = "";
  try { mountInfo = readText("/proc/1/mountinfo"); } catch { /* not Linux, or hidden */ }
  const bindMount = runtimeBindMount(mountInfo);
  const legacyMount =
    mountInfo.includes("/var/lib/containerd/") ||
    mountInfo.includes("/var/lib/kubelet/pods/") ||
    mountInfo.includes("/containers/");
  const verified = platform === "linux" && Boolean(marker || bindMount);
  return {
    detected: Boolean(envSignal || marker || legacyMount),
    verified,
    evidence: marker ? `file: ${marker}`
      : bindMount ?? (envSignal ? `env: ${envSignal}` : legacyMount ? "/proc/1/mountinfo: runtime path" : null),
  };
}

function policyFailure(path, reason) {
  return new HostPolicyError(
    `host policy ${path} is invalid: ${reason}. Fix or remove it; with no policy the relay keeps Claude's shell sandbox.`,
  );
}

export function validateHostPolicy(policy, path) {
  if (!policy || typeof policy !== "object" || Array.isArray(policy)) {
    throw policyFailure(path, "it must be a JSON object");
  }
  const unknown = Object.keys(policy).filter((key) => !POLICY_FIELDS.has(key));
  if (unknown.length) throw policyFailure(path, `unknown field(s) ${unknown.join(", ")}`);
  if (policy.schema !== HOST_POLICY_SCHEMA) throw policyFailure(path, `schema must be "${HOST_POLICY_SCHEMA}"`);
  if (policy.shellSandbox !== "outer-container") {
    throw policyFailure(path, 'shellSandbox must be "outer-container", the only value this schema defines');
  }
  if (policy.platform !== "linux") throw policyFailure(path, 'platform must be "linux"');
  if (policy.requireContainer !== true) throw policyFailure(path, "requireContainer must be true");
  if (policy.approval !== undefined) {
    const approval = policy.approval;
    if (!approval || typeof approval !== "object" || Array.isArray(approval) ||
        Object.values(approval).some((value) => typeof value !== "string")) {
      throw policyFailure(path, "approval must be an object of string values");
    }
  }
  return {
    schema: policy.schema,
    shellSandbox: policy.shellSandbox,
    platform: policy.platform,
    requireContainer: true,
    approval: policy.approval ?? null,
  };
}

/** Returns null only when the file does not exist; every other problem throws. */
export function loadHostPolicy(path, {
  platform = process.platform,
  uid = typeof process.getuid === "function" ? process.getuid() : null,
  stat = statSync,
  readText = (file) => readFileSync(file, "utf8"),
} = {}) {
  let stats;
  try {
    stats = stat(path);
  } catch (error) {
    if (error && error.code === "ENOENT") return null;
    throw policyFailure(path, `it cannot be inspected (${error && error.code ? error.code : error})`);
  }
  if (!stats.isFile()) throw policyFailure(path, "it is not a regular file");
  if (stats.size > MAX_POLICY_BYTES) throw policyFailure(path, "it is larger than 64 KiB");
  if (platform !== "win32") {
    if (uid !== null && stats.uid !== uid) {
      throw policyFailure(path, `it must be owned by the invoking user (uid ${uid}), not uid ${stats.uid}`);
    }
    if (stats.mode & 0o022) throw policyFailure(path, "it must not be group- or world-writable");
  }
  let text;
  try {
    text = readText(path);
  } catch (error) {
    throw policyFailure(path, `it cannot be read (${error && error.code ? error.code : error})`);
  }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw policyFailure(path, "it is not valid JSON");
  }
  return validateHostPolicy(parsed, path);
}

/**
 * Combine the invocation's flags, the host facts, and an already-validated
 * policy into one effective boundary. Called once per relay invocation; both
 * --preflight-only and the task dispatch consume the same result.
 */
export function resolveShellBoundary({
  readOnly, noShellSandbox, platform, policyPath, policy, container, dangerouslySkipPermissions = false, uid = null,
}) {
  if (policy) {
    if (platform !== policy.platform) {
      throw new HostPolicyError(
        `host policy ${policyPath} selects outer-container isolation for ${policy.platform}, but this host is ${platform}. ` +
        "The relay will not weaken Claude's shell sandbox here; remove the policy on this host.",
      );
    }
    if (policy.requireContainer && !container.verified) {
      throw new HostPolicyError(
        `host policy ${policyPath} requires a Linux container, but none was verified ` +
        `(evidence: ${container.evidence ?? "none"}; environment variables alone do not count). ` +
        "The relay will not weaken Claude's shell sandbox on an ordinary host; remove the policy here.",
      );
    }
  }
  const hostPolicy = {
    path: policyPath,
    status: !policy ? "absent" : readOnly ? "unused-read-only" : noShellSandbox ? "unused-task-opt-in" : "applied",
    ...(policy ? { approval: policy.approval, containerEvidence: container.evidence } : {}),
  };
  // Claude refuses --dangerously-skip-permissions as root unless the process
  // is marked as a deliberate sandbox (IS_SANDBOX=1). Only a verified, explicit
  // container policy may set that marker, and only on the child that needs it.
  // It acknowledges the outer container; it does not create an inner sandbox.
  const childEnv = policy && !readOnly && dangerouslySkipPermissions && uid === 0 ? { IS_SANDBOX: "1" } : {};
  const boundary = (shellSandbox, effective, source, innerSandbox, extra = {}) =>
    ({ shellSandbox, effective, source, innerSandbox, weakerNestedSandbox: false, hostPolicy, childEnv, ...extra });

  if (readOnly) return boundary("not-applicable", "no-shell", "read-only", false);
  if (noShellSandbox) return boundary("disabled-explicit-task-opt-in", "outer-environment", "task-opt-in", false);
  if (platform === "win32") return boundary("unsupported-on-native-windows", "none", "native-windows", false);
  if (policy) return boundary("outer-container-policy", "outer-container", "host-policy", false);
  return container.detected
    ? boundary("container-compatible", "claude-shell-sandbox", "default", true, { weakerNestedSandbox: true })
    : boundary("strict", "claude-shell-sandbox", "default", true);
}
