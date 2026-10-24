---
name: claude-delegate
description: >-
  Delegate a coding task to a separate Claude Code CLI process or another Claude session as an
  implementer, then review its diff and land it yourself. Use when the user asks to delegate
  implementation to Claude Code, another Claude session, or the `claude` CLI, including a standing
  delegation preference recorded in applicable instructions — for example,
  "have another Claude implement this", "delegate this to Claude Code", or "run this queue through a
  separate Claude session." Do not trigger merely because the current orchestrator is Claude, and do
  not use when the user asks the current Claude to implement directly without delegation.
license: MIT
metadata:
  version: 0.5.2
---

# Claude Delegate

You are the **orchestrator**. Make a short plan and acceptance criteria, then hand a complete,
independently verifiable task to a separate Claude Code session early. Claude edits the tree;
you review the result, rerun the important gates, and land accepted work.

This skill is not a signal for the current Claude to implement directly. Use it when the human
requests delegation, including a standing preference in applicable instructions.

## When not to use this

- The human asked the current agent to implement the task directly.
- The task is small enough to do inline and the human did not request delegation.
- The `claude` CLI is missing or unauthenticated (`claude auth status`).
- The task needs a stronger host boundary than Claude Code's tool permissions and shell-only sandbox
  provide. Use an isolated container or VM for that requirement.

## Prerequisites

The relay requires Node 18+ and git. The orchestrator must be able to run shell commands and read
files. Claude's shell sandbox requires macOS, Linux, or WSL2; native Windows has no shell sandbox.

1. `claude --version` succeeds.
2. `claude auth status` reports an authenticated session. On macOS the live credentials sit in the
   login Keychain; when the orchestrator's own sandbox blocks Keychain access (Codex's sandbox
   does), `claude` falls back to a possibly stale credentials file and reports `loggedIn: false`
   even though the login is valid. Re-run the check — and the dispatch itself — with that sandbox
   escalated or outside it before concluding the CLI is unauthenticated.
3. The target repository is the directory passed with `--cd`.
4. On Linux/WSL2, Claude's sandbox dependencies are installed. The normal relay profile is
   configured to fail when the sandbox is unavailable instead of silently running shell commands
   unsandboxed. Existing merged settings can still affect the effective boundary.

## The loop

### 1. Plan briefly and write the brief

The separate session has no orchestrator chat history. It receives the brief on stdin and can inspect
the target working tree.

Decide the target behavior, scope, and acceptance criteria without doing the implementer's
investigation first. Give Claude enough context to implement and verify the whole bounded task;
include exact gate commands and a report contract. Claude Code automatically discovers the target
project's `CLAUDE.md` and normal local Claude configuration because the relay does not use `--bare`.
It does **not** generically auto-load `AGENTS.md`. Read `AGENTS.md` yourself and copy every load-bearing constraint and the real gate
commands into the brief. Tell the implementer not to commit. Keep one task per brief.

Template and details: [references/writing-the-brief.md](references/writing-the-brief.md).

### 2. Preflight Claude's Bash environment

Before a write-capable task dispatch on macOS, Linux, or WSL2, invoke the **same registered relay and workspace** with
`--preflight-only` and a short timeout (90 seconds or less). With codex-background, use a separate
`delegate_run` and a fresh runId; pass `relayArgs: ["--preflight-only"]`. Its brief path is required
by the adapter but is ignored by this relay mode. For shell use:

```bash
node "<skill-dir>/scripts/relay.mjs" --preflight-only --cd /path/to/repo
```

The relay starts a fresh Claude process with the normal permission and sandbox settings, allows only
Bash, requests one `pwd` call, and stops at its tool result. It caps the probe at two turns, $0.50,
and 90 seconds. **Check `result.json` for `preflight.passed === true`**, not just a zero Claude exit
or `PRECHECK_OK` in its final text. If the probe fails, times out, or lacks a Bash result, stop this
delegation round and report the error. Do not dispatch the implementation, retry unsandboxed, or
spend the task timeout investigating the sandbox. A shell command run by Codex is not this preflight.

The probe uses a separate Claude session. Keep its session ID out of the task run; resume only the
task's session when review finds concrete fixes.

### 3. Dispatch

For **Codex orchestration of background work**, install and configure the separate
**codex-background** support utility first. If its MCP tools or registry are missing, stop and
report the setup requirement; do not silently fall back to model-driven polling. Then invoke its
registered `delegate_run` MCP tool once in place of the shell example below.
Follow that skill for structured common arguments, unchanged provider flags and reattachment by
runId. Keep the request pending until completion, then continue this skill's review and landing
steps. Other orchestrators keep the shell workflow below.

```bash
node "<skill-dir>/scripts/relay.mjs" --brief brief.txt --cd /path/to/repo
# review/diagnosis only:                 add --read-only
# continue the recorded session:         add --session <id>
# choose limits:                         add --max-turns 40 --max-budget-usd 10
# set the auto-compact window:           add --autocompact 400k
# hard relay deadline:                   add --timeout 2h
# inspect every option:                  node .../relay.mjs --help
```

`<skill-dir>` is this installed skill directory, the folder containing this `SKILL.md`.

The relay runs `claude -p --output-format stream-json --verbose`, sends the brief through stdin, and
writes artifacts under the system temp directory by default. It never uses `--bg` or `--bare`, and it
never commits. See [references/dispatch-and-poll.md](references/dispatch-and-poll.md).

`--autocompact <auto|tokens>` passes Claude Code's auto-compact window setting on every new or resumed invocation, and needs Claude Code `2.1.221` or newer — older builds fail the dispatch with `unknown option '--autocompact'`. The installed CLI owns the accepted range (`auto`, or 100k–1M tokens); the relay records the requested value but does not claim that Claude applied or enforced it.

### 4. Wait

Codex using codex-background keeps the original MCP request pending: no model-driven sleep,
status or log polling. Use `delegate_wait` only to reattach after an interrupted request.

The relay blocks until Claude exits. Use the orchestrator's background-command facility, or run it in
the foreground and wait. Completion means the process exited and `result.json` exists.

- A pre-run usage error exits 2 and writes no `result.json`.
- A missing `claude` exits 127 and writes `status: "claude_unavailable"`.
- Timeout and caught relay signals terminate the whole implementer process tree and preserve an
  outcome artifact.

Read `finalMessage`, `touchedFiles`, `resultSubtype`, and the raw artifact paths from `result.json`.

### 5. Review

Treat the implementer's report and gate outcomes as claims:

- Review edits to existing tests before a green gate means anything.
- Re-run the project's actual gates yourself.
- Read the complete diff against the brief, starting with `touchedFiles`.
- Inspect untracked and staged content as well as the ordinary diff.
- Run relevant guard skills if installed.

Full checklist: [references/review-and-land.md](references/review-and-land.md).

### 6. Land

The **orchestrator commits** only after the gates pass and the diff holds. For rework, resume the same
task Claude session with a delta brief listing only concrete review findings, affected behavior,
and the gates to rerun. Do not send a new broad task or use a new session for ordinary corrections:

```bash
echo "Keep the implementation, replace the mocked DB test with the migrated fixture, and remove the
unused import." | node "<skill-dir>/scripts/relay.mjs" --session <id> --cd /path/to/repo
```

Review a resumed run exactly like the first run.

## Permission profiles

The normal profile is deliberately explicit:

- `acceptEdits` permission mode.
- Built-in tools restricted to Read, Glob, Grep, Edit, Write, and the platform shell.
- On macOS, Linux, and WSL2, Claude's shell sandbox is enabled with startup failure on missing
  dependencies and no unsandboxed retry. Commands that stay sandboxed are auto-approved so ordinary
  gates can run headlessly. The sandbox governs shell processes and their children only; merged
  local or managed sandbox settings can add effective paths or exclusions.
- In a detected Docker, Podman, or Kubernetes container, the relay adds Claude's documented
  `enableWeakerNestedSandbox` compatibility setting and allows Unix sockets for that child invocation.
  This reuses the outer container's `/proc` and skips Claude's broken nested seccomp helper while
  retaining bwrap filesystem isolation and network proxying. Sandboxed commands can see container
  process information and connect to Unix sockets visible inside the container. The settings live
  only in the relay's per-run `profile.json`; ordinary Claude sessions are unaffected.
- Configured MCP discovery and Claude.ai connectors are disabled, all MCP tools are denied, and
  skills, commands, and Claude's Agent tool are unavailable to the child. Project `CLAUDE.md`, hooks,
  normal authentication, session persistence, and other local settings still load.
- String rules deny common direct shell forms of `git commit`, `git push`, and nested `claude`, plus
  any command containing `claude-delegate`. Aliases, scripts, and wrappers can bypass them, so they
  are only a speed bump; the brief's no-commit instruction and orchestrator review remain the boundary.

Native Windows does not support Claude's shell sandbox. The relay restricts the tool surface and
pre-approves PowerShell so the run remains non-interactive, but that shell is not OS-isolated. Native
`claude.exe` and npm `claude.cmd` launch paths are implemented. The `claude.cmd` path has a recorded
native Windows run (see the README's Verification status); a directly spawned `claude.exe` is not yet
verified.

`--read-only` uses `plan` mode with only Read, Glob, and Grep. It removes edit, write, and shell paths,
then compares parsed git porcelain and fingerprints the working-tree identity and index entries of
Git-visible paths that were already dirty.
`readOnlyViolation` is `true` when either signal proves a change, `false` when coverage is complete and
detects none, and `null` when coverage is incomplete. This is a reporting tripwire, not an OS boundary:
ignored paths and perfect restores are outside it, local hooks can write, and concurrent changes cannot
be attributed to Claude.

`--dangerously-skip-permissions` is an explicit opt-in to Claude's `bypassPermissions` mode. The
restricted tool surface, direct commit/push deny rules, and supported-platform shell sandbox remain,
but direct file tools can cross normal permission boundaries. Use it only with the human's explicit
acceptance.

## Complementary to native Claude features

Claude subagents, agent teams, and background sessions are useful when the current Claude environment
is already the orchestrator and native coordination is the goal. This skill is complementary: it
provides a cross-orchestrator contract — self-contained brief → dispatch → artifacts → review → land
— and keeps the commit with the orchestrator.

## References

- [references/writing-the-brief.md](references/writing-the-brief.md) — context, `CLAUDE.md` versus
  `AGENTS.md`, real gates, report contract, and delta briefs.
- [references/dispatch-and-poll.md](references/dispatch-and-poll.md) — flags, profiles, artifacts,
  `result.json`, polling, and failure recovery.
- [references/review-and-land.md](references/review-and-land.md) — generated-code review, the commit
  boundary, and session rework.
- [references/multi-task-queues.md](references/multi-task-queues.md) — sequential queues, progress
  tracking, constraint carry-forward, and final coherence.
