# Fresh Noir test session

Use this lane from Codex or Claude for an exact, very simple test. The runner:

1. Executes the caller's exact argv once using `shell=False` in the named directory.
2. Records stdout, stderr, exit status and elapsed time under the chosen devlog path.
3. Starts a new `claude --print` session, pinned to Noir, with a new UUID and isolated
   temporary configuration. No resume, conversation import, plugins, hooks, skills,
   MCP or agent tools. A compact, bounded test-output excerpt is data in its brief.
4. Saves the independent model report and observed model IDs. The test process's
   outcome stays authoritative even if the model's prose disagrees.

```bash
python3 /absolute/skill/scripts/run_simple_test.py \
  --cwd /absolute/small-project \
  --artifacts /absolute/devlog/noir-smoke-unique \
  --test-timeout 120 --session-timeout 180 \
  -- python3 -m unittest test_parser.TestParser.test_empty
```

The artifact directory must be new and outside the tested directory; the runner
does not scan or copy the project. Temporary diagnostic scripts also belong in the
host's devlog. Tests themselves may create their usual fixtures/cache; choose a
safe test directory and pass its applicable rules in `--brief-file` if needed.
Do not give the runner a shell wrapper that violates the authorized scope.

The Claude CLI must be installed with this version's `--bare` and print-mode flags.
Credentials/provider URL come from the active process or the explicitly named
`--settings-file` (default: `~/.claude/settings.json`), in memory only. No secrets
are printed or written by the runner. The default model is the local gateway ID
`anthropic/a7n-noir[1m]`; use `BAH_NOIR_MODEL` or `--model` for the host's Noir ID.
No automatic model fallback, credential changes or CLI installation occurs.

Exit 0 means the command exited 0 **and** a valid Noir result was received, not that
all desired coverage ran. Test failure retains its nonzero exit status; a test
timeout returns 124; a failed/unavailable/invalid model session after a passing test
returns 2. `run.json` distinguishes the two outcomes and holds artifact filenames.
Interrupted runs return 130 and stop only their own child process group; partial
artifacts with no `state: complete` are not a completed result.
Read the actual output for skips and empty suites. Output excerpts are capped;
the strongest brain reads the raw artifact when diagnosis needs the omitted detail.

This runner deliberately keeps command execution deterministic. Noir summarizes
the actual evidence; it cannot edit code, expand test scope, or turn a failed test
green by changing assertions. Ordinary tests go to Sol/Sonnet, and all failure
analysis goes to the Astra/Opus brain.
