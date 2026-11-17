# brain-and-hands: Opus-as-brain subagent delegation for Claude Code

A Claude Code plugin that cuts cost and context bloat by making a strong main session (Opus) the **brain** and cheaper models (Sonnet, Haiku) the **hands**. Works as a token-saving multi-agent workflow for Claude Code: model-pinned subagents, a delegation skill, and safe stop points for computer use.

The main session decides, writes a complete brief, and verifies. Subagents pinned to cheaper models do the high-volume, low-judgment work (screenshots, bulk reads, test runs, research) and report back as text.

## Why

Two reasons it can cost less:

1. **Rate.** Haiku and Sonnet cost much less per token than Opus.
2. **Context.** A subagent's raw output (screenshots, logs, file dumps) never enters the main session's context, which is re-read on every turn.

It is not free. Briefing, the subagent's cold start, and verification all cost tokens, and total tokens usually go up even when the bill goes down. Short or judgment-heavy tasks are cheaper inline, and the bundled skill says so. Computer use is the best fit: image-heavy, repetitive, and almost no judgment.

## What's inside

| Piece | What it does |
|---|---|
| `agents/scout` | Haiku, read-only. Searches and summarises many files. |
| `agents/worker` | Sonnet. Executes a fully specified change, build or test run. |
| `agents/researcher` | Sonnet. Sourced web and document research. |
| `agents/operator` | Sonnet. Drives screens and consoles; stops before irreversible actions. |
| `skills/delegate` | The doctrine: what to delegate, the brief template, how to verify. |
| `hooks/session-start.sh` | A short reminder at session start (about 70 words). |

## Install

```
/plugin marketplace add prograsec/brain-and-hands
/plugin install brain-and-hands@brain-and-hands
```

Then start sessions on an Opus-class model. The delegation rules are written for a strong main session; subagents ignore them.

## How it behaves

- Delegates wide searches, bulk reads, test and build runs, mechanical edits, research, and anything involving screenshots.
- Does short questions, judgment-heavy work and tightly coupled edits inline.
- Every brief names the goal, exact inputs, what "done" looks like, stop points, and a text-only report format.
- Irreversible or outward-facing actions (publish, submit, send, delete, pay) are reported back unperformed unless you approved that exact action.
- The main session re-checks surprising claims and does its own pass before any go/no-go decision. Cheaper models can pick wrongly with confidence.

## Tuning

- Edit the `model:` line in any `agents/*.md` to change which model a role uses.
- Do not set `CLAUDE_CODE_SUBAGENT_MODEL`; it forces every subagent onto one model and removes the Haiku/Sonnet split.
- If a delegated task came back wrong or cost more than doing it inline, adjust the "Delegate or do it inline" section of `skills/delegate/SKILL.md`.

## Not claimed

No benchmark numbers are included. Savings depend on your workload. Track your own with `/cost` before and after.

## License

MIT
