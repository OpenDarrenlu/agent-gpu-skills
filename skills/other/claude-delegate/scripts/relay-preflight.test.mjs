import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const relay = fileURLToPath(new URL("./relay.mjs", import.meta.url));

function probe(mode) {
  const root = mkdtempSync(join(tmpdir(), "claude-relay-preflight-"));
  const bin = join(root, "bin");
  const work = join(root, "work");
  const output = join(root, "out");
  mkdirSync(bin);
  mkdirSync(work);
  const stub = join(bin, "claude");
  writeFileSync(stub, `#!/usr/bin/env node
if (process.argv.includes("--version")) { console.log("test-claude"); process.exit(0); }
process.stdin.resume();
process.stdin.on("end", () => {
  const id = "bash-probe";
  console.log(JSON.stringify({type:"assistant",message:{content:[{type:"tool_use",id,name:"Bash",input:{command:"pwd"}}]}}));
  if (process.env.PROBE_MODE !== "missing") {
    const failed = process.env.PROBE_MODE === "failed";
    const nonzero = process.env.PROBE_MODE === "nonzero";
    const content = failed ? "Exit code 1\\nbwrap: Can't mount proc on /newroot/proc: Operation not permitted" : (nonzero ? "Exit code 1\\n" : "Exit code 0\\n") + process.cwd();
    console.log(JSON.stringify({type:"user",message:{content:[{type:"tool_result",tool_use_id:id,is_error:failed,content}]}}));
    setInterval(() => {}, 1000);
  }
});
`);
  chmodSync(stub, 0o755);
  try {
    const run = spawnSync(process.execPath, [relay, "--preflight-only", "--cd", work, "--out-dir", output, "--timeout", "5s"], {
      env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, PROBE_MODE: mode },
      encoding: "utf8",
      timeout: 12_000,
    });
    assert.equal(run.error, undefined, run.stderr);
    return {
      run,
      result: JSON.parse(readFileSync(join(output, "result.json"), "utf8")),
      profile: JSON.parse(readFileSync(join(output, "profile.json"), "utf8")),
      work,
    };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test("preflight accepts the actual successful Bash result", () => {
  const { run, result, profile, work } = probe("success");
  assert.equal(run.status, 0);
  assert.equal(result.status, "completed");
  assert.equal(result.preflight.passed, true);
  assert.equal(result.preflight.command, "pwd");
  assert.equal(result.preflight.cwd, work);
  assert.deepEqual(result.toolSurface, ["Bash"]);
  if (process.env.KUBERNETES_SERVICE_HOST || process.env.CONTAINER) {
    assert.equal(result.shellSandbox, "container-compatible");
    assert.equal(profile.sandbox.enableWeakerNestedSandbox, true);
    assert.equal(profile.sandbox.network.allowAllUnixSockets, true);
  }
});

test("preflight fails on bubblewrap mount denial even if Claude remains alive", () => {
  const { run, result } = probe("failed");
  assert.equal(run.status, 1);
  assert.equal(result.status, "failed");
  assert.equal(result.preflight.passed, false);
  assert.match(result.error, /Can't mount proc/);
});

test("preflight rejects a nonzero Bash exit even if its output resembles pwd", () => {
  const { run, result } = probe("nonzero");
  assert.equal(run.status, 1);
  assert.equal(result.preflight.passed, false);
});

test("preflight fails when Claude exits without a Bash result", () => {
  const { run, result } = probe("missing");
  assert.equal(run.status, 1);
  assert.equal(result.status, "failed");
  assert.equal(result.preflight.passed, false);
  assert.match(result.error, /did not return a Bash tool result/);
});
