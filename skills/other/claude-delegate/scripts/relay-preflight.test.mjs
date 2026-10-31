import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  HOST_POLICY_SCHEMA,
  HostPolicyError,
  codexHome,
  detectContainer,
  hostPolicyPath,
  resolveShellBoundary,
  validateHostPolicy,
} from "./host-policy.mjs";

const relay = fileURLToPath(new URL("./relay.mjs", import.meta.url));
// The relay inspects the real host, so integration expectations follow it.
const hostContainer = detectContainer();

const VALID_POLICY = {
  schema: HOST_POLICY_SCHEMA,
  shellSandbox: "outer-container",
  platform: "linux",
  requireContainer: true,
  approval: { approvedBy: "test user", approvedAt: "2026-10-31", reason: "outer container isolates this host" },
};

const FAKE_CLAUDE = `#!/usr/bin/env node
const fs = require("node:fs");
if (process.argv.includes("--version")) { console.log("test-claude"); process.exit(0); }
const argv = process.argv.slice(2);
fs.writeFileSync(process.env.PROBE_ARGV_FILE, JSON.stringify(argv));
fs.writeFileSync(process.env.PROBE_ARGV_FILE + ".env", JSON.stringify({ IS_SANDBOX: process.env.IS_SANDBOX ?? null }));
// Same gate as Claude 2.1.316's isRootOutsideDeliberateSandbox().
if (argv.includes("--dangerously-skip-permissions") && process.getuid() === 0 &&
    process.env.IS_SANDBOX !== "1" && !process.env.CLAUDE_CODE_BUBBLEWRAP) {
  console.error("--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons");
  process.exit(1);
}
const mode = process.env.PROBE_MODE;
process.stdin.resume();
process.stdin.on("end", () => {
  if (mode === "task") {
    console.log(JSON.stringify({type:"system",subtype:"init",session_id:"task-session"}));
    console.log(JSON.stringify({type:"result",subtype:"success",is_error:false,result:"done",session_id:"task-session"}));
    return;
  }
  const id = "bash-probe";
  console.log(JSON.stringify({type:"assistant",message:{content:[{type:"tool_use",id,name:"Bash",input:{command:"pwd"}}]}}));
  if (mode === "missing") return;
  const pwd = process.env.PROBE_PWD || process.cwd();
  const failed = mode === "failed";
  const content = failed
    ? "Exit code 1\\nbwrap: Can't mount proc on /newroot/proc: Operation not permitted"
    : mode === "nonzero" ? "Exit code 1\\n" + pwd : pwd + "\\n";
  console.log(JSON.stringify({type:"user",message:{content:[{type:"tool_result",tool_use_id:id,is_error:failed,content}]}}));
  setInterval(() => {}, 1000);
});
`;

function readJson(path) {
  return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null;
}

/**
 * Run the relay against a fake `claude` with an isolated CODEX_HOME.
 * `policy`: object (JSON-encoded), string (raw text), or undefined (absent).
 * `args`, `pwd`, and `cd` may be functions of the temporary layout.
 */
function runRelay({ args = ["--preflight-only"], mode = "success", policy, policyMode = 0o644, pwd, cd } = {}) {
  const root = mkdtempSync(join(tmpdir(), "claude-relay-preflight-"));
  const layout = {
    root,
    work: join(root, "work"),
    extra: join(root, "extra"),
    link: join(root, "work-link"),
    brief: join(root, "brief.txt"),
    codexHome: join(root, "codex"),
  };
  const bin = join(root, "bin");
  const output = join(root, "out");
  const argvFile = join(root, "claude-argv.json");
  mkdirSync(bin);
  mkdirSync(layout.work);
  mkdirSync(layout.extra);
  symlinkSync(layout.work, layout.link);
  writeFileSync(layout.brief, "Implement nothing; this is a fixture.\n");
  writeFileSync(join(bin, "claude"), FAKE_CLAUDE);
  chmodSync(join(bin, "claude"), 0o755);
  const policyPath = join(layout.codexHome, "claude-delegate", "host-policy.json");
  if (policy !== undefined) {
    mkdirSync(dirname(policyPath), { recursive: true });
    writeFileSync(policyPath, typeof policy === "string" ? policy : JSON.stringify(policy));
    chmodSync(policyPath, policyMode);
  }
  const value = (option) => (typeof option === "function" ? option(layout) : option);
  // Never inherit a deliberate-sandbox marker; only the relay may add one.
  const { IS_SANDBOX: _marker, CLAUDE_CODE_BUBBLEWRAP: _bwrap, ...inherited } = process.env;
  try {
    const run = spawnSync(process.execPath, [
      relay, ...value(args), "--cd", value(cd) ?? layout.work, "--out-dir", output, "--timeout", "5s",
    ], {
      env: {
        ...inherited,
        PATH: `${bin}:${process.env.PATH}`,
        CODEX_HOME: layout.codexHome,
        PROBE_MODE: mode,
        PROBE_PWD: value(pwd) ?? "",
        PROBE_ARGV_FILE: argvFile,
      },
      encoding: "utf8",
      timeout: 12_000,
    });
    assert.equal(run.error, undefined, run.stderr);
    return {
      run,
      result: readJson(join(output, "result.json")),
      profile: readJson(join(output, "profile.json")),
      argv: readJson(argvFile),
      childEnv: readJson(`${argvFile}.env`),
      policyPath,
      ...layout,
    };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

function flagValue(argv, flag) {
  const index = argv.indexOf(flag);
  return index === -1 ? undefined : argv[index + 1];
}

function assertRejectedBeforeLaunch(outcome, pattern) {
  assert.equal(outcome.run.status, 2, outcome.run.stderr);
  assert.match(outcome.run.stderr, pattern);
  assert.equal(outcome.result, null, "a rejected policy must not produce a run");
  assert.equal(outcome.argv, null, "claude must not be launched");
}

// ---------------------------------------------------------------- preflight

test("preflight accepts the actual successful Bash result", () => {
  const { run, result, profile, argv, work, policyPath } = runRelay();
  assert.equal(run.status, 0, run.stderr);
  assert.equal(result.status, "completed");
  assert.equal(result.preflight.passed, true);
  assert.equal(result.preflight.command, "pwd");
  assert.equal(result.preflight.cwd, work);
  assert.equal(result.preflight.observedCwd, work);
  assert.deepEqual(result.toolSurface, ["Bash"]);
  // Missing policy keeps the existing strong default.
  assert.equal(result.shellBoundary.source, "default");
  assert.equal(result.shellBoundary.innerSandbox, true);
  assert.deepEqual(result.shellBoundary.hostPolicy, { path: policyPath, status: "absent" });
  assert.equal(profile.sandbox.enabled, true);
  assert.equal(profile.sandbox.failIfUnavailable, true);
  assert.equal(profile.sandbox.allowUnsandboxedCommands, false);
  assert.equal(argv.includes("--allowedTools"), false);
  assert.equal(result.shellSandbox, hostContainer.detected ? "container-compatible" : "strict");
  if (hostContainer.detected) {
    assert.equal(profile.sandbox.enableWeakerNestedSandbox, true);
    assert.equal(profile.sandbox.network.allowAllUnixSockets, true);
  }
});

test("preflight fails on bubblewrap mount denial even if Claude remains alive", () => {
  const { run, result } = runRelay({ mode: "failed" });
  assert.equal(run.status, 1);
  assert.equal(result.status, "failed");
  assert.equal(result.preflight.passed, false);
  assert.match(result.error, /Can't mount proc/);
});

test("preflight rejects a nonzero Bash exit even if its output resembles pwd", () => {
  const { run, result } = runRelay({ mode: "nonzero" });
  assert.equal(run.status, 1);
  assert.equal(result.preflight.passed, false);
});

test("preflight fails when Claude exits without a Bash result", () => {
  const { run, result } = runRelay({ mode: "missing" });
  assert.equal(run.status, 1);
  assert.equal(result.status, "failed");
  assert.equal(result.preflight.passed, false);
  assert.match(result.error, /did not return a Bash tool result/);
});

test("preflight fails when Bash succeeds in the wrong working directory", () => {
  // Models a login shell profile that changes directory before `pwd` runs.
  const { run, result, root, work } = runRelay({ pwd: (layout) => layout.root });
  assert.equal(run.status, 1);
  assert.equal(result.status, "failed");
  assert.equal(result.preflight.passed, false);
  assert.equal(result.preflight.observedCwd, root);
  assert.equal(result.preflight.cwd, work);
  assert.match(result.error, /did not start in the requested working directory/);
});

test("preflight treats a symlinked --cd and its logical pwd as the same directory", () => {
  const { run, result, work, link } = runRelay({ cd: (layout) => layout.link, pwd: (layout) => layout.link });
  assert.equal(run.status, 0, run.stderr);
  assert.equal(result.preflight.passed, true);
  assert.equal(result.preflight.cwd, work);
  assert.equal(result.preflight.observedCwd, link);
});

// ------------------------------------------------------------ host policy

test("a valid container policy gives preflight and dispatch the same outer-container boundary", () => {
  const preflight = runRelay({ policy: VALID_POLICY });
  const task = runRelay({
    policy: VALID_POLICY,
    mode: "task",
    args: (layout) => ["--brief", layout.brief, "--add-dir", layout.extra],
  });
  if (!hostContainer.verified) {
    assertRejectedBeforeLaunch(preflight, /requires a Linux container/);
    assertRejectedBeforeLaunch(task, /requires a Linux container/);
    return;
  }
  for (const { run, result, profile, argv, policyPath } of [preflight, task]) {
    assert.equal(run.status, 0, run.stderr);
    assert.equal(result.status, "completed");
    assert.equal(result.shellSandbox, "outer-container-policy");
    assert.equal(result.shellBoundary.effective, "outer-container");
    assert.equal(result.shellBoundary.source, "host-policy");
    assert.equal(result.shellBoundary.innerSandbox, false);
    assert.equal(result.shellBoundary.hostPolicy.path, policyPath);
    assert.equal(result.shellBoundary.hostPolicy.status, "applied");
    assert.deepEqual(result.shellBoundary.hostPolicy.approval, VALID_POLICY.approval);
    assert.equal(result.shellBoundary.hostPolicy.containerEvidence, hostContainer.evidence);
    assert.deepEqual(profile.sandbox, { enabled: false });
    assert.equal(profile.permissions.deny.length, 8, "commit/push/nested-claude deny rules stay");
    assert.equal(flagValue(argv, "--allowedTools"), "Bash");
    assert.equal(flagValue(argv, "--permission-mode"), "acceptEdits");
    assert.equal(argv.includes("--dangerously-skip-permissions"), false);
  }
  assert.equal(preflight.result.preflight.passed, true);
  assert.equal(flagValue(preflight.argv, "--tools"), "Bash");
  assert.equal(flagValue(task.argv, "--tools"), "Read,Glob,Grep,Edit,Write,Bash");
  assert.equal(task.result.sessionId, "task-session");
  assert.equal(flagValue(task.argv, "--add-dir"), task.extra);
  assert.deepEqual(task.profile.permissions.additionalDirectories, [task.extra]);
});

test("a malformed or unknown host policy fails closed before Claude starts", () => {
  const cases = [
    ["{ not json", /not valid JSON/],
    ["[]", /must be a JSON object/],
    [{ ...VALID_POLICY, allowEverything: true }, /unknown field\(s\) allowEverything/],
    [{ ...VALID_POLICY, schema: "claude-delegate.host-policy.v2" }, /schema must be/],
    [{ ...VALID_POLICY, shellSandbox: "disabled" }, /shellSandbox must be "outer-container"/],
    [{ ...VALID_POLICY, platform: "darwin" }, /platform must be "linux"/],
    [{ ...VALID_POLICY, requireContainer: false }, /requireContainer must be true/],
    [{ ...VALID_POLICY, approval: { approvedBy: 7 } }, /approval must be an object of string values/],
  ];
  for (const [policy, pattern] of cases) {
    assertRejectedBeforeLaunch(runRelay({ policy }), pattern);
  }
  // Dispatch takes the same path, so a bad policy is never a silent bypass.
  assertRejectedBeforeLaunch(
    runRelay({ policy: "{", mode: "task", args: (layout) => ["--brief", layout.brief] }),
    /not valid JSON/,
  );
});

test("a group- or world-writable host policy is rejected", () => {
  assertRejectedBeforeLaunch(runRelay({ policy: VALID_POLICY, policyMode: 0o666 }), /group- or world-writable/);
});

test("read-only stays shell-free whether or not a host policy exists", () => {
  for (const policy of [undefined, VALID_POLICY]) {
    const outcome = runRelay({ policy, mode: "task", args: (layout) => ["--read-only", "--brief", layout.brief] });
    if (policy && !hostContainer.verified) {
      assertRejectedBeforeLaunch(outcome, /requires a Linux container/);
      continue;
    }
    const { run, result, profile, argv } = outcome;
    assert.equal(run.status, 0, run.stderr);
    assert.deepEqual(result.toolSurface, ["Read", "Glob", "Grep"]);
    assert.equal(flagValue(argv, "--tools"), "Read,Glob,Grep");
    assert.equal(flagValue(argv, "--permission-mode"), "plan");
    assert.equal(argv.includes("--allowedTools"), false);
    assert.equal("sandbox" in profile, false);
    assert.equal(result.shellSandbox, "not-applicable");
    assert.equal(result.shellBoundary.effective, "no-shell");
    assert.equal(result.shellBoundary.hostPolicy.status, policy ? "unused-read-only" : "absent");
  }
});

test("the --no-shell-sandbox task opt-in and --add-dir still work without a policy", () => {
  const task = runRelay({
    mode: "task",
    args: (layout) => ["--no-shell-sandbox", "--brief", layout.brief, "--add-dir", layout.extra],
  });
  const preflight = runRelay({ args: ["--preflight-only", "--no-shell-sandbox"] });
  for (const { run, result, profile, argv } of [task, preflight]) {
    assert.equal(run.status, 0, run.stderr);
    assert.equal(result.shellSandbox, "disabled-explicit-task-opt-in");
    assert.equal(result.shellBoundary.effective, "outer-environment");
    assert.equal(result.shellBoundary.source, "task-opt-in");
    assert.equal(result.shellBoundary.hostPolicy.status, "absent");
    assert.deepEqual(profile.sandbox, { enabled: false });
    assert.equal(flagValue(argv, "--allowedTools"), "Bash");
  }
  assert.equal(preflight.result.preflight.passed, true);
  assert.deepEqual(task.argv.slice(task.argv.indexOf("--add-dir"), task.argv.indexOf("--add-dir") + 2), ["--add-dir", task.extra]);
  assert.deepEqual(task.profile.permissions.additionalDirectories, [task.extra]);
  const readOnly = runRelay({ mode: "task", args: (layout) => ["--read-only", "--no-shell-sandbox", "--brief", layout.brief] });
  assertRejectedBeforeLaunch(readOnly, /--no-shell-sandbox requires a write-capable task/);
});

// ------------------------------------------------------ permission bypass

const isRoot = typeof process.getuid === "function" && process.getuid() === 0;
const ROOT_REFUSAL = /cannot be used with root\/sudo privileges/;

test("an authorized container lets root bypass preflight and dispatch run with the same profile", () => {
  const bypassPreflight = runRelay({ policy: VALID_POLICY, args: ["--preflight-only", "--dangerously-skip-permissions"] });
  const bypassTask = runRelay({
    policy: VALID_POLICY,
    mode: "task",
    args: (layout) => ["--dangerously-skip-permissions", "--brief", layout.brief],
  });
  const normalPreflight = runRelay({ policy: VALID_POLICY });
  if (!hostContainer.verified) {
    for (const outcome of [bypassPreflight, bypassTask, normalPreflight]) {
      assertRejectedBeforeLaunch(outcome, /requires a Linux container/);
    }
    return;
  }
  const marker = isRoot ? { IS_SANDBOX: "1" } : {};
  for (const { run, result, profile, argv, childEnv } of [bypassPreflight, bypassTask]) {
    assert.equal(run.status, 0, run.stderr);
    assert.equal(result.permissionMode, "bypassPermissions");
    assert.equal(result.dangerouslySkipPermissions, true);
    assert.equal(argv.includes("--dangerously-skip-permissions"), true);
    assert.equal(argv.includes("--permission-mode"), false);
    assert.deepEqual(profile.sandbox, { enabled: false });
    assert.equal(result.shellBoundary.source, "host-policy");
    assert.equal(result.shellBoundary.innerSandbox, false);
    assert.deepEqual(result.shellBoundary.childEnv, marker);
    assert.equal(childEnv.IS_SANDBOX, isRoot ? "1" : null);
  }
  assert.deepEqual(bypassPreflight.profile, bypassTask.profile, "preflight proves the task's exact profile");
  const { preflight, maxTurns, maxBudgetUsd, timeout, resumed } = bypassPreflight.result;
  assert.equal(preflight.passed, true);
  assert.equal(preflight.observedCwd, bypassPreflight.work);
  assert.deepEqual([maxTurns, maxBudgetUsd, timeout, resumed], [2, 0.5, "5s", false]);
  // The marker is scoped to bypass; a normal-mode child never receives it.
  assert.equal(normalPreflight.run.status, 0, normalPreflight.run.stderr);
  assert.deepEqual(normalPreflight.result.shellBoundary.childEnv, {});
  assert.equal(normalPreflight.childEnv.IS_SANDBOX, null);
});

test("only explicit bypass inside a validated container policy drops the generated deny list", () => {
  const DEFAULT_DENY = [
    "Bash(git commit *)", "Bash(git * commit *)", "Bash(git * commit)",
    "Bash(git push *)", "Bash(git * push *)", "Bash(git * push)",
    "Bash(claude *)", "Bash(*claude-delegate*)",
  ];
  const bypassArgs = (layout) => ["--dangerously-skip-permissions", "--brief", layout.brief];
  // Profiles that must keep the original deny list, whatever Claude does next.
  const keeps = {
    defaultNoPolicy: runRelay({ mode: "task", args: (layout) => ["--brief", layout.brief] }),
    bypassNoPolicy: runRelay({ mode: "task", args: bypassArgs }),
    bypassTaskOptIn: runRelay({ mode: "task", args: (layout) => ["--no-shell-sandbox", ...bypassArgs(layout)] }),
  };
  for (const [name, { profile }] of Object.entries(keeps)) {
    assert.deepEqual(profile.permissions.deny, DEFAULT_DENY, `${name} keeps the original deny list`);
  }
  const policyNormal = runRelay({ policy: VALID_POLICY, mode: "task", args: (layout) => ["--brief", layout.brief] });
  const policyBypass = runRelay({ policy: VALID_POLICY, mode: "task", args: bypassArgs });
  const policyBypassPreflight = runRelay({ policy: VALID_POLICY, args: ["--preflight-only", "--dangerously-skip-permissions"] });
  if (!hostContainer.verified) {
    for (const outcome of [policyNormal, policyBypass, policyBypassPreflight]) {
      assertRejectedBeforeLaunch(outcome, /requires a Linux container/);
    }
    return;
  }
  assert.deepEqual(policyNormal.profile.permissions.deny, DEFAULT_DENY, "policy without bypass keeps deny rules");
  for (const { run, result, profile, argv } of [policyBypass, policyBypassPreflight]) {
    assert.equal(run.status, 0, run.stderr);
    assert.equal(result.status, "completed");
    assert.equal(result.permissionMode, "bypassPermissions");
    assert.equal(result.shellBoundary.source, "host-policy");
    assert.equal(result.shellBoundary.hostPolicy.status, "applied");
    assert.deepEqual(profile.permissions.deny, [], "authorized container bypass has no shell deny rules");
    // Everything else about the profile is unchanged: no connectors, no MCP, no inner sandbox.
    assert.equal(profile.disableClaudeAiConnectors, true);
    assert.deepEqual(profile.sandbox, { enabled: false });
    assert.equal(flagValue(argv, "--disallowedTools"), "mcp__*");
    assert.equal(argv.includes("--strict-mcp-config"), true);
  }
  assert.equal(flagValue(policyBypass.argv, "--tools"), "Read,Glob,Grep,Edit,Write,Bash");
  assert.deepEqual(policyBypassPreflight.profile, policyBypass.profile, "preflight proves the task's exact profile");
});

test("without an authorized policy root bypass still meets Claude's own refusal", () => {
  const outcomes = [
    runRelay({ args: ["--preflight-only", "--dangerously-skip-permissions"] }),
    runRelay({ mode: "task", args: (layout) => ["--no-shell-sandbox", "--dangerously-skip-permissions", "--brief", layout.brief] }),
  ];
  for (const { run, result, childEnv } of outcomes) {
    assert.deepEqual(result.shellBoundary.childEnv, {});
    assert.equal(childEnv.IS_SANDBOX, null);
    if (isRoot) {
      assert.equal(run.status, 1);
      assert.equal(result.status, "failed");
      assert.match(result.stderrTail.join("\n"), ROOT_REFUSAL);
    } else {
      assert.equal(run.status, 0, run.stderr);
    }
  }
  assert.equal(outcomes[0].result.preflight.passed, !isRoot);
  // Bypass preflight still requires a fresh session.
  assertRejectedBeforeLaunch(
    runRelay({ args: ["--preflight-only", "--dangerously-skip-permissions", "--session", "prior"] }),
    /fresh write-capable session/,
  );
});

// ------------------------------------------------- pure policy resolution

const IN_CONTAINER = { detected: true, verified: true, evidence: "/proc/1/mountinfo: /etc/hosts is bound from /var/lib/kubelet/pods/" };
const ORDINARY_HOST = { detected: false, verified: false, evidence: null };
const ENV_ONLY = { detected: true, verified: false, evidence: "env: KUBERNETES_SERVICE_HOST" };
const policy = validateHostPolicy(VALID_POLICY, "/policy.json");
const boundary = (overrides) => resolveShellBoundary({
  readOnly: false, noShellSandbox: false, platform: "linux", policyPath: "/policy.json", policy, container: IN_CONTAINER, ...overrides,
});

test("an outer-container policy is rejected outside a verified Linux container", () => {
  assert.throws(() => boundary({ container: ORDINARY_HOST }), (error) =>
    error instanceof HostPolicyError && /requires a Linux container/.test(error.message));
  assert.throws(() => boundary({ container: ENV_ONLY }), /environment variables alone do not count/);
  assert.throws(() => boundary({ platform: "darwin" }), /this host is darwin/);
  assert.throws(() => boundary({ platform: "win32", container: ORDINARY_HOST }), /this host is win32/);
  // Read-only and task opt-in do not make a mismatched policy acceptable.
  assert.throws(() => boundary({ container: ORDINARY_HOST, readOnly: true }), HostPolicyError);
  assert.throws(() => boundary({ container: ORDINARY_HOST, noShellSandbox: true }), HostPolicyError);
});

test("without a policy every host keeps its normal boundary", () => {
  const normal = (overrides) => boundary({ policy: null, ...overrides });
  assert.deepEqual(
    [normal({ container: ORDINARY_HOST }), normal({ platform: "darwin", container: ORDINARY_HOST })]
      .map(({ shellSandbox, innerSandbox, source }) => [shellSandbox, innerSandbox, source]),
    [["strict", true, "default"], ["strict", true, "default"]],
  );
  const container = normal({ container: ENV_ONLY });
  assert.equal(container.shellSandbox, "container-compatible");
  assert.equal(container.innerSandbox, true);
  assert.equal(container.weakerNestedSandbox, true);
  assert.equal(normal({ platform: "win32", container: ORDINARY_HOST }).shellSandbox, "unsupported-on-native-windows");
  assert.equal(boundary({}).shellSandbox, "outer-container-policy");
});

test("only a valid container policy with root bypass yields the deliberate-sandbox marker", () => {
  const rootBypass = { dangerouslySkipPermissions: true, uid: 0 };
  assert.deepEqual(boundary(rootBypass).childEnv, { IS_SANDBOX: "1" });
  assert.deepEqual(boundary({ ...rootBypass, noShellSandbox: true }).childEnv, { IS_SANDBOX: "1" });
  for (const overrides of [
    { ...rootBypass, policy: null },
    { ...rootBypass, policy: null, container: ORDINARY_HOST },
    { ...rootBypass, policy: null, platform: "darwin", container: ORDINARY_HOST },
    { ...rootBypass, uid: 1000 },
    { dangerouslySkipPermissions: false, uid: 0 },
  ]) {
    assert.deepEqual(boundary(overrides).childEnv, {}, JSON.stringify(overrides));
  }
  assert.throws(() => boundary({ ...rootBypass, container: ORDINARY_HOST }), HostPolicyError);
});

test("container verification needs filesystem evidence, not a heuristic hit", () => {
  const facts = (mountInfo, { env = {}, files = [] } = {}) => detectContainer({
    platform: "linux", env, exists: (path) => files.includes(path), readText: () => mountInfo,
  });
  const podLine = "3464 3294 253:2 /var/lib/kubelet/pods/2de9/etc-hosts /etc/hosts rw,relatime - ext4 /dev/vda2 rw\n";
  const nodeLine = "812 29 253:2 /var/lib/kubelet/pods/2de9/volumes/x /var/lib/kubelet/pods/2de9/volume-subpaths/x rw - ext4 /dev/vda2 rw\n";
  const dockerHostShm = "901 29 0:52 / /var/lib/docker/containers/abc/mounts/shm rw - tmpfs shm rw\n";
  assert.equal(facts(podLine).verified, true);
  assert.match(facts(podLine).evidence, /\/etc\/hosts is bound from \/var\/lib\/kubelet\/pods\//);
  for (const hostLine of [nodeLine, dockerHostShm]) {
    assert.equal(facts(hostLine).verified, false, hostLine);
    assert.equal(facts(hostLine).detected, true, "the legacy weaker-nested heuristic is unchanged");
  }
  assert.deepEqual(facts("", { env: { KUBERNETES_SERVICE_HOST: "10.0.0.1" } }),
    { detected: true, verified: false, evidence: "env: KUBERNETES_SERVICE_HOST" });
  assert.equal(facts("", { files: ["/.dockerenv"] }).verified, true);
  assert.equal(detectContainer({ platform: "darwin", env: {}, exists: () => true, readText: () => podLine }).verified, false);
  assert.deepEqual(detectContainer({ platform: "win32" }), ORDINARY_HOST);
});

test("the policy path follows Codex home resolution", () => {
  assert.equal(codexHome({ CODEX_HOME: "/srv/codex" }, "/home/u"), "/srv/codex");
  assert.equal(codexHome({ CODEX_HOME: "" }, "/home/u"), "/home/u/.codex");
  assert.equal(hostPolicyPath({}, "/home/u"), "/home/u/.codex/claude-delegate/host-policy.json");
});
