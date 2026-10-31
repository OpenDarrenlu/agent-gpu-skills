# Dispatch and poll

`scripts/relay.mjs` wraps Claude Code's non-interactive `claude -p` mode, sends a brief on stdin,
captures the structured stream, and writes `delegate-relay.result.v1`.

## Before the first run

```bash
command -v claude
claude --version
claude auth status
```

The relay performs its own `claude --version` check and records the answer as `claudeVersion`.
That check does not exercise Claude's Bash sandbox. Before the task, run the same relay with
`--preflight-only` in the target workspace. It resolves the same effective shell boundary the task
will get (default sandbox, task opt-in, or host policy), restricts the tool surface to Bash, invokes
`pwd`, and terminates at the Bash result. Inspect `result.json`'s `preflight.passed`; stop the
delegation round unless it is `true`.
When an orchestrator is itself Claude Code, the relay removes only inherited `CLAUDECODE` from the
child environment so a separate CLI process can start. It preserves credentials,
`CLAUDE_CODE_CHILD_SESSION`, and every other environment entry. Version preflight uses that same
environment.

## Dispatch

```bash
node "<skill-dir>/scripts/relay.mjs" --brief brief.txt --cd /path/to/repo
```

`<skill-dir>` is the installed folder containing this skill's `SKILL.md`.

| Flag | Effect |
| --- | --- |
| `--brief <file>` | Brief path. Omit it to read stdin. The exact text is then sent to Claude on stdin, never argv. |
| `--cd <dir>` | Child process cwd and target working root (default: current directory). |
| `--add-dir <dir>` | Additional authorized directory, repeatable. Passed to Claude's `--add-dir` and recorded in `profile.json`'s `permissions.additionalDirectories`. |
| `--lane <name>` | Fleet lane from `delegate-setup` config. Applies that lane's dials; fails if the lane's `implementer` is not this relay. Explicit dial flags win. |
| `--out-dir <dir>` | Artifact directory (default: a fresh directory under the system temp directory). |
| `--timeout <dur>` | Relay watchdog, such as `30m`, `90s`, or `2h` (default: off). |
| `--model <name>` | Claude model alias or full name (default: Claude's configured choice). The relay does not pin a model version. |
| `--effort <level>` | Claude effort: `low`, `medium`, `high`, `xhigh`, `max`, or `ultracode`; availability depends on the model. |
| `--max-turns <n>` | Positive agentic-turn cap. |
| `--max-budget-usd <amount>` | Positive decimal spend cap for print mode. |
| `--autocompact <auto\|tokens>` | Set Claude Code's auto-compact window on Claude Code `2.1.221` or newer. The relay validates a shell-safe shape; the installed Claude CLI owns the supported range. |
| `--resume-last` | Resume the latest session for this cwd with Claude's `--continue`; send a delta brief. |
| `--session <id>` | Resume a specific session with Claude's `--resume <id>`; mutually exclusive with `--resume-last`. |
| `--read-only` | Plan mode with only Read, Glob, and Grep, plus a Git-visible change tripwire. |
| `--preflight-only` | Fresh probe of the effective Bash boundary on macOS, Linux, or WSL2; ignores the task brief, caps at two turns, $0.50 and 90 seconds, and writes `preflight.passed` to `result.json`. Passes only when `pwd` resolves (through symlinks) to the `--cd` directory. Add `--dangerously-skip-permissions` when the task will use it, so the probe runs the task's exact profile. Cannot combine with read-only or resume flags; rejects native Windows. |
| `--no-shell-sandbox` | Explicit per-task opt-in: disables Claude's shell sandbox and pre-approves Bash, relying on the outer environment for isolation. Requires the human's authorization; rejected with `--read-only`. Pass it to the preflight too, so the probe tests the same boundary. |
| `--dangerously-skip-permissions` | Opt into Claude's `bypassPermissions`; mutually exclusive with `--read-only`. |
| `-h`, `--help` | Print the relay header and option reference. |

Values that can reach an npm `claude.cmd` launch on Windows are token- or number-validated. The brief
never reaches a shell command line.

## What the relay launches

The common shape is:

```bash
claude -p --output-format stream-json --verbose \
  --tools Read,Glob,Grep,Edit,Write,Bash \
  --strict-mcp-config --disallowedTools 'mcp__*' \
  --disable-slash-commands \
  --settings <profile.json> \
  --permission-mode acceptEdits \
  < brief.txt
```

On native Windows, `PowerShell` replaces `Bash` and is passed through `--allowedTools` because Claude's
shell sandbox is unavailable there. Read-only uses `--tools Read,Glob,Grep --permission-mode plan`. A
specific session adds `--resume <id>`; the latest session adds `--continue`. The permission and tool
profile is re-passed on every resumed invocation. When requested, `--autocompact <value>` is also
re-passed on every new, `--resume`, or `--continue` invocation rather than assumed to persist from an
earlier launch.

The relay never adds `--bg`: Claude documents background mode as incompatible with `-p`. It never adds
`--bare`, because bare mode skips `CLAUDE.md` and OAuth/keychain authentication.

`--strict-mcp-config` without an MCP config prevents configured-server discovery. The inline settings
also disable Claude.ai connectors, while `--disallowedTools 'mcp__*'` denies any MCP tool that managed
policy still supplies. `--disable-slash-commands` prevents skill and command recursion, and `--tools`
omits the Agent tool. These controls do not suppress project `CLAUDE.md` or normal authentication.

## Permission reach

### Normal write-capable profile

The normal profile pairs `acceptEdits` with sandbox auto-approval on supported platforms.
`acceptEdits` accepts file edits but does not, by itself, approve ordinary shell gates in a headless
run. `autoAllowBashIfSandboxed: true` approves commands that stay inside Claude's sandbox. A command
that cannot stay sandboxed fails instead of being retried outside it. Native Windows is the explicit
exception: the relay pre-approves PowerShell because no Claude shell sandbox is available there.

The generated `profile.json`:

- uses string rules to deny common direct shell forms of `git commit`, `git push`, and nested
  `claude`, plus any command containing `claude-delegate`; aliases, scripts, and wrappers can bypass
  these speed bumps, so the brief's no-commit instruction and orchestrator review remain the boundary;
- on macOS, Linux, and WSL2, enables Claude's Bash sandbox with `failIfUnavailable: true`,
  `autoAllowBashIfSandboxed: true`, `allowUnsandboxedCommands: false`, and filesystem isolation
  explicitly enabled;
- in a detected Docker, Podman, or Kubernetes container, adds `enableWeakerNestedSandbox: true` and
  `network.allowAllUnixSockets: true` to that run's inline profile. This binds the outer container's
  existing `/proc` and skips the optional Unix-socket seccomp helper, which cannot create its nested
  user namespace after bwrap. Filesystem isolation and proxy-based network restriction remain, while
  container process information and container-visible Unix sockets become reachable;
- on native Windows, leaves the unsupported sandbox unconfigured and enables Claude's PowerShell
  tool through its documented settings environment switch;
- disables Claude.ai connectors for every profile.

On supported platforms, the strict settings prevent a shell command from silently falling back to an
unsandboxed retry. Claude's sandbox covers **Bash and its child processes only**. Edit/Write remain
Claude Code tools governed by its permission system; the Claude process, local hooks, inherited
configuration, and unrelated host processes are not enclosed in a universal workspace sandbox.

Claude merges some sandbox arrays across settings scopes. Existing managed, user, project, or local
allowlists and `excludedCommands` can therefore affect the effective shell boundary, while managed
policy can further restrict or reject the run. Existing `ask` and `deny` permission rules take
precedence over sandbox auto-approval, so they can still stop a headless gate. Inspect both
`profile.json` and the effective local Claude configuration when the precise boundary matters. Use a
container or VM when only a host-level boundary is acceptable.

### Outer-container host policy

Some Linux containers cannot run Claude's shell sandbox at all. For example, bubblewrap fails with
`loopback: Failed RTM_NEWADDR: Operation not permitted`, or it cannot map user namespaces, when the
outer runtime and the host's AppArmor `restrict_unprivileged_userns` setting deny them. The relay
never fixes that by changing the host. Instead, a user who has decided the **container itself** is
the boundary can say so once, in a user-owned file resolved like Codex's own home:

```text
$CODEX_HOME/claude-delegate/host-policy.json   (when CODEX_HOME is set)
~/.codex/claude-delegate/host-policy.json       (otherwise)
```

```json
{
  "schema": "claude-delegate.host-policy.v1",
  "shellSandbox": "outer-container",
  "platform": "linux",
  "requireContainer": true,
  "approval": { "approvedBy": "<who>", "approvedAt": "<date>", "reason": "<why>" }
}
```

`approval` is optional (string values only) and is copied into `result.json`. Every other field is
required, and unknown fields are rejected. Resolution happens once per invocation, and preflight and
dispatch use the result identically:

- **No file:** nothing changes. This holds on ordinary Linux, macOS, WSL2, and Windows.
- **Valid file in a verified Linux container:** `profile.json` gets `sandbox.enabled: false`, Bash is
  pre-approved with `--allowedTools`, and `shellSandbox` is `"outer-container-policy"`. Bash then
  runs with **all of the container's permissions**: its network, its files outside the workspace,
  and its processes. Edit/Write stay under `acceptEdits`. The deny rules, MCP/skill restrictions, and
  tool surface are unchanged. `bypassPermissions` is not enabled.
- **Verification** requires filesystem evidence: `/.dockerenv`, `/run/.containerenv`, or a mount in
  `/proc/1/mountinfo` whose root is a runtime path (`/var/lib/kubelet/pods/`,
  `/var/lib/docker/containers/`, `/var/lib/containerd/`, `/var/lib/containers/`) bound onto an
  ordinary mount point. Environment variables such as `KUBERNETES_SERVICE_HOST` alone do not verify.
  Kubernetes nodes and Docker hosts, whose PID 1 merely lists runtime mount points, do not verify.
- **Refused (exit 2, no `result.json`, Claude not launched):** invalid JSON, an unknown schema,
  field, or `shellSandbox` value, a file that is not regular, not owned by the invoking user, or
  group/world-writable, a non-Linux host, or Linux without verified container evidence. The policy
  is validated even for `--read-only` and `--no-shell-sandbox` runs, so a misplaced file is reported
  rather than ignored.
- **Root with `--dangerously-skip-permissions`:** Claude Code itself exits with
  `--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons`
  unless the process is marked as a deliberate sandbox. In Claude 2.1.316 that check is
  `isRootOutsideDeliberateSandbox()`: non-Windows, uid 0, `IS_SANDBOX` not `"1"`, and no
  `CLAUDE_CODE_BUBBLEWRAP`. With a valid, verified policy, the relay adds `IS_SANDBOX=1` to that
  child's environment only (preflight and task alike) and records it as `shellBoundary.childEnv`.
  The marker acknowledges the outer container; it does not create an inner sandbox. It is not added
  without a policy, for non-root users, or for non-bypass runs, so Claude's own refusal still
  applies everywhere else. The relay never sets `CLAUDE_CODE_BUBBLEWRAP`, because bubblewrap is not
  running.
- **`--read-only`** stays shell-free (`hostPolicy.status: "unused-read-only"`).
  **`--no-shell-sandbox`** still records `"disabled-explicit-task-opt-in"`
  (`hostPolicy.status: "unused-task-opt-in"`).

The relay never creates this file and never retries a failed sandbox without the sandbox. To return
to the inner sandbox, delete the file.

### Read-only profile

`--read-only` removes Edit, Write, Bash/PowerShell, Agent, MCP, skills, and commands from the child and
uses `plan` mode. Local hooks still load outside that tool surface and can write. The relay compares
parsed `git status --porcelain -z -uall` and fingerprints working-tree identity and index entries for
paths that were already dirty on every outcome, including aborts:

- `readOnlyViolation: true` — either signal proves a Git-visible change.
- `readOnlyViolation: false` — coverage was complete and neither signal detected a change.
- `readOnlyViolation: null` — coverage was incomplete, for example because git could not report or a
  dirty submodule or unreadable path could not be fingerprinted.

This detects new dirt and changes to readable, already-dirty Git-visible paths, but cannot attribute a
concurrent change. Ignored paths, submodule internals, and writes perfectly restored before the final
snapshot remain outside coverage. Inspect the diff whenever read-only integrity matters.

### Permission bypass

`--dangerously-skip-permissions` passes Claude's flag and records
`permissionMode: "bypassPermissions"` unless the init event reports another value. The explicit tool
surface, commit/push deny rules, MCP/skill restrictions, and the effective shell boundary remain.
Exception: when the boundary is an applied Linux outer-container host policy, the generated
`permissions.deny` is empty, because that combination is the human's explicit no-local-restriction
choice; MCP/connector restrictions, the tool surface, and the brief's no-commit scope still apply.
However, direct file tools can cross ordinary Claude permission boundaries. This mode requires the
human's explicit acceptance.

## Artifacts

The default artifact directory is outside the repository so relay output does not pollute
`touchedFiles`. A caller-selected `--out-dir` inside the worktree will appear in git status. For a
meaningful `--read-only` review, keep artifacts outside the worktree. The fingerprint signal excludes
only the relay-owned artifact paths, so their later writes do not prove a violation.

- `brief.txt` — exact stdin brief.
- `events.jsonl` — raw stdout bytes from `--output-format stream-json`.
- `final.txt` — final `result` event's `result` text; present even when empty.
- `stderr.txt` — complete stderr.
- `profile.json` — exact inline settings passed to Claude.
- `result.json` — stable result contract, written atomically.

## `result.json`

Core fields:

- `schema` — `"delegate-relay.result.v1"`.
- `tool` — `"claude"`.
- `status` — `completed` | `failed` | `timeout` | `aborted` | `claude_unavailable`.
- `exitCode` — Claude's code, `127` when missing, a signal-derived code when available, or a forced
  non-zero value when the watchdog or terminal error result makes a zero code non-successful.
- `signal` — terminating child/relay signal when reported, otherwise `null`.
- `claudeVersion` — version preflight text, `"unknown"` when the binary answered abnormally, or `null`
  when unavailable.
- `permissionMode` — selected profile, updated from `system/init` when present.
- `sessionId` — parsed defensively from init/result events; use with `--session`.
- `resultSubtype` — final result subtype, such as `success` or an error subtype.
- `finalMessage` — final result text.
- `numTurns`, `usage`, `totalCostUsd` — terminal result metadata when present.
- `touchedFiles` — final `git status --porcelain` lines for `--cd`; `null` means git could not report,
  while `[]` means git reported a clean tree. This is the whole final tree, not attribution.
- `readOnlyViolation` — present only on `--read-only`, with the three-state meaning above.
- `preflight` — present only on `--preflight-only`. `passed` is true only when the Bash tool
  succeeded and its `pwd` (`observedCwd`) resolves to the canonical `--cd` directory (`cwd`);
  `error` records the failure, including a shell that started in another directory. A successful
  Claude exit or final text alone is not evidence that Bash worked.
- `shellSandbox` — `strict`, `container-compatible`, `outer-container-policy`,
  `disabled-explicit-task-opt-in`, `not-applicable` (read-only), or `unsupported-on-native-windows`.
- `shellBoundary` — `effective` (`claude-shell-sandbox`, `outer-container`, `outer-environment`,
  `no-shell`, or `none`), `source` (`default`, `host-policy`, `task-opt-in`, `read-only`, or
  `native-windows`), `innerSandbox` (whether Claude's sandbox was requested), and `hostPolicy`
  (`path`, `status` of `absent`/`applied`/`unused-read-only`/`unused-task-opt-in`, plus `approval`
  and `containerEvidence` when a policy exists), and `childEnv` (environment the relay added for
  Claude, `{}` or `{ "IS_SANDBOX": "1" }` for an authorized root bypass). Use `source` to tell an
  explicit outer-container authorization apart from a working inner sandbox.

Run metadata includes `workdir`, `model`, `effort`, `maxTurns`, `maxBudgetUsd`, optional
`autocompact`, `timeout`, `readOnly`,
`resumed`, `resumeLast`, `toolSurface`, `dangerouslySkipPermissions`, timestamps, and
all artifact paths. Failed, timed-out, and aborted runs include `stderrTail` when available;
launch/watchdog/signal failures include `error`.

`autocompact` records only the value the relay requested and passed in argv. It is absent when the
flag was omitted and is not proof that Claude Code applied or enforced the setting. On Claude Code older than 2.1.221 the launch fails with `unknown option '--autocompact'`; if a newer CLI rejects the value's range it fails at argument parsing. Either way the non-zero exit and stderr are preserved and the relay does not retry without the flag.

The relay prints a concise summary and the complete final report to stdout, then exits with
`result.json`'s `exitCode`.

## Wait for completion

The relay blocks. Use the orchestrator's background-command facility or foreground it for short work.
If using a shell's own background feature, completion requires both:

1. the relay process has exited; and
2. `result.json` contains a terminal `status`.

A usage error exits 2 before creating `result.json`, so also observe process exit. A missing CLI is
different: it exits 127 **with** `status: "claude_unavailable"`.

## Failure recovery

- **`claude_unavailable`:** install Claude Code, authenticate with `claude auth login`, and verify the
  same PATH the orchestrator uses.
- **`failed`:** inspect `resultSubtype`, `error`, `stderrTail`, `stderr.txt`, and the tail of
  `events.jsonl`. Common causes are authentication, a model/effort mismatch, managed policy, a missing
  sandbox dependency, or a gate that needs access outside the strict shell boundary. A preflight
  `error` naming a different directory means the shell starts outside `--cd` (for example a login
  profile that changes directory); fix that environment rather than dispatching.
- **Exit 2 naming `host policy`:** the policy file is malformed, insecurely writable, or does not
  match this host. Fix it or delete it; the relay does not fall back silently.
- **`timeout`:** the relay sent termination to the whole process group/tree and escalated. The working
  tree may contain partial edits; inspect it before resuming or re-dispatching.
- **`aborted`:** the relay caught SIGTERM, SIGINT, or SIGHUP, terminated the implementer tree, wrote an
  outcome, then refreshed `touchedFiles` after a grace window. Native Windows cannot deliver every
  termination as a catchable Node signal; a vanished relay with no result still requires direct
  artifact/tree inspection.
- **Host `SIGKILL`:** the relay cannot catch its own SIGKILL. If the child reports SIGKILL, investigate
  host memory or supervisor deadlines.
- **Empty `finalMessage`:** inspect the raw events and diff. Require a closing report in the next brief.

Never clean, reset, or switch branches before inspecting partial work, staged changes, and untracked
files.

## Windows launch

The relay resolves PATH itself. A native `claude.exe` is spawned directly. An npm `claude.cmd` is
invoked through `cmd.exe /d /v:off /s /c` with every argument quoted and user-selectable values
restricted; stdin still carries the brief. `taskkill /t /f` terminates the process tree.

The npm `claude.cmd` path has a recorded native Windows run covering write, resume, read-only,
watchdog tree termination, and the missing-binary result; a directly spawned `claude.exe` is not yet
verified. Claude's Bash sandbox is unsupported there, so that run verifies launch and termination
mechanics, not the supported-platform shell boundary.

## Commit boundary

The relay never commits. Claude edits; the orchestrator reviews, re-runs gates, and lands. See
[review-and-land.md](review-and-land.md).
