# Model families and host adapters

Resolve only the family the user requested:

| Tier | Codex family | Claude alias | Cursor slug | Local example verified 2026-11-18 |
|---|---|---|---|---|
| Brain / debugging | `astra` | `opus` or `fable` | current Fable or Opus session; a full slug launches a brain subagent | `gpt-6-astra` |
| Tests / worker / research / operator | `sol` | `sonnet` | latest `claude-sonnet-*` in the session list, or an explicit hand slug | `gpt-6.1-sol` |
| Read-only scout | `terra` | `haiku` | latest `claude-haiku-*`, or an explicit hand slug when Haiku is absent | `gpt-6.1-terra` |
| Very simple tests, any host | fresh **Claude** session | a7n Noir | same fresh Claude session | `anthropic/a7n-noir[1m]` (local gateway ID; real session verified 2026-11-18) |

Examples are observations, not universal pins. `scripts/route_task.py` reads only
the explicitly named Codex model-cache file and picks the greatest numeric version
in a family. It never substitutes Luna or a different family. Missing catalogs or
families return an actionable error. Use `--models-file` on another host.
Claude's `opus`, `fable`, `sonnet`, and `haiku` aliases resolve through the active
provider; a gateway may pin them. Inspect the actual model returned by the session
rather than claiming the alias guarantees the newest globally released model.
`fable` is a brain alias, accepted for `--task debug` beside `opus`. It is not a hand.

```bash
python3 scripts/route_task.py --host codex --task test
python3 scripts/route_task.py --host codex --task debug
python3 scripts/route_task.py --host claude --task debug --model fable
python3 scripts/route_task.py --host claude --task simple-test
python3 scripts/route_task.py --host cursor --task test --models-file "$MODELS"
```

Explicit `--model` pins win within the requested role. Optional environment
overrides: `BAH_CODEX_ASTRA_MODEL`, `BAH_CODEX_SOL_MODEL`, `BAH_CODEX_TERRA_MODEL`,
`BAH_CLAUDE_OPUS_MODEL`, `BAH_CLAUDE_FABLE_MODEL`, `BAH_CLAUDE_SONNET_MODEL`,
`BAH_CLAUDE_HAIKU_MODEL`, `BAH_CURSOR_BRAIN_MODEL`, `BAH_CURSOR_SONNET_MODEL`,
`BAH_CURSOR_HAIKU_MODEL`, `BAH_CURSOR_MODELS_FILE`, and `BAH_NOIR_MODEL`. An override
is a deliberate pin, so the resolver reports it as such. The Noir override must
remain a Noir identifier. When both Claude brain overrides are set, Opus wins over
Fable; an explicit `--model` wins over both.

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

For very simple tests, use the standalone Noir runner even inside Claude or Cursor;
this is an independent session, not an in-conversation Agent call. It disables skills,
hooks, MCP and nested hands for that child only, retaining provider credentials in
memory. It does not change global settings. Check CLI/result artifacts for the
actual model and report an unavailable gateway ID without silently using Gris.

## Cursor

The brain stays the current Fable or Opus session. `--task debug` without `--model`
returns that session. A full `claude-fable-*` or `claude-opus-*` pin launches a
brain subagent and must appear in the session's available model list.

Hands require that list as `--models-file` (or `BAH_CURSOR_MODELS_FILE`), using the
same `{"models":[{"slug":"..."}]}` shape as the Codex cache. The default ordinary
hand is the latest `claude-sonnet-*` slug, and the default scout is the latest
`claude-haiku-*` slug. If that family is absent, the resolver fails and names the
other non-brain slugs; it does not pick one. An explicit `--model` may then be any
non-brain slug in the list, including `deepseek-v41-flash-1017-high`,
`grok-4.7-high-fast`, Gemini, and Muse. Fable and Opus slugs are rejected as hands.

Without a catalog, an explicit hand pin is accepted only when it is recognizable
as Sonnet, Haiku, Sol, Terra, DeepSeek, Grok, Gemini, or Muse. Pass the Task tool
that exact `model` value. A missing `model` on a Cursor subagent is not a hand.

In multitask mode the parent follows the Cursor multitask section of `SKILL.md`.
A subagent whose brief is specified and whose logic is not especially complex uses a
hand slug and acts as a hand, including work a step above an ordinary test or edit.
Hand subagents follow the brief and do not launch further subagents. A Fable or Opus
subagent is reserved for design, ambiguous diagnosis, and the final diff check. An
exact, very simple test still uses the Noir runner in `noir-session.md` rather than a
Cursor Task subagent. The workspace rule
`/root/code/.cursor/rules/brain-and-hands-multitask.mdc` points Cursor sessions at this
skill.
