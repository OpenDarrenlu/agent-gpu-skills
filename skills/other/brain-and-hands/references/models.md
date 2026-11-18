# Model families and host adapters

Resolve only the family the user requested:

| Tier | Codex family | Claude alias | Local example verified 2026-11-18 |
|---|---|---|---|
| Brain / debugging | `astra` | `opus` | `gpt-6-astra` |
| Tests / worker / research / operator | `sol` | `sonnet` | `gpt-6.1-sol` |
| Read-only scout | `terra` | `haiku` | `gpt-6.1-terra` |
| Very simple tests, either host | fresh **Claude** session | a7n Noir | `anthropic/a7n-noir[1m]` (local gateway ID; real session verified 2026-11-18) |

Examples are observations, not universal pins. `scripts/route_task.py` reads only
the explicitly named Codex model-cache file and picks the greatest numeric version
in a family. It never substitutes Luna or a different family. Missing catalogs or
families return an actionable error. Use `--models-file` on another host.
Claude's `opus`, `sonnet`, `haiku` aliases resolve through the active provider; a
gateway may pin them. Inspect the actual model returned by the session rather than
claiming the alias guarantees the newest globally released model.

```bash
python3 scripts/route_task.py --host codex --task test
python3 scripts/route_task.py --host codex --task debug
python3 scripts/route_task.py --host claude --task simple-test
```

Explicit `--model` pins win within the requested family. Optional environment
overrides: `BAH_CODEX_ASTRA_MODEL`, `BAH_CODEX_SOL_MODEL`, `BAH_CODEX_TERRA_MODEL`,
`BAH_CLAUDE_OPUS_MODEL`, `BAH_CLAUDE_SONNET_MODEL`, `BAH_CLAUDE_HAIKU_MODEL`, and
`BAH_NOIR_MODEL`. An override is a deliberate pin, so the resolver reports it as
such. The Noir override must remain a Noir identifier.

## Codex

Keep the existing Astra main session. For hands, resolve the model first, then use
the native subagent tool with that exact model only if its available choices expose
it. Tool choices can differ from the CLI catalog; Terra's presence in a cache does
not prove the native subagent tool supports it. A same-model inherited hand is not
a Sol/Terra hand. Do not create sidebar chats with create_thread for internal work.

When an authorized hand needs a CLI-only model, use a fresh `codex exec --model`
session with an absolute `--cd`, a complete brief on stdin, appropriate permissions,
`--ephemeral`, and recorded output. Do not `resume`, `fork` or use `--last` for a
cold hand. Inspect `codex exec --help` for the installed version. Do not weaken
permissions merely to make a test pass. Debug stays with the Astra brain.

## Claude

The optional plugin exposes `scout` (Haiku), `worker`, `tester`, `researcher`,
`operator` (Sonnet), and `debugger` (Opus). Model aliases remain version-flexible.
`grunt` is retained for compatibility as a **Noir test-only** role; it is not a
general-purpose free audit/implementation hand. Inherit the strongest main session
for diagnosis. Use `debugger` only when a separate strong analysis is useful.

For very simple tests, use the standalone Noir runner even inside Claude; this is
an independent session, not an in-conversation Agent call. It disables skills,
hooks, MCP and nested hands for that child only, retaining provider credentials in
memory. It does not change global settings. Check CLI/result artifacts for the
actual model and report an unavailable gateway ID without silently using Gris.
