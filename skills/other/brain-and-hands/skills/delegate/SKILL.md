---
name: delegate
description: >
  Choose brain-and-hands roles for an authorized Claude or Codex workflow, especially
  independent tests and debugging: Noir for very simple tests, Sonnet/Sol or another
  available hand model for ordinary test execution, and Opus/Fable/Astra for code
  and failure analysis. Cursor multitask subagents use the same roles.
---

# Brain and hands plugin entry

Read [the shared skill](../../SKILL.md) before choosing or launching a hand. It is
the canonical workflow for both hosts; do not duplicate its rules in this entry.
Resolve references and scripts from the plugin root, `${CLAUDE_PLUGIN_ROOT}`, when
the plugin loader presents this skill through a different working directory.

An already launched hand completes its brief without further delegation. A multitask
subagent with a specified brief and no especially complex logic uses a hand model
and is a hand, even when the work is a step above an ordinary hand. A Fable or Opus
subagent stays a brain for design and failure analysis, and may launch hands. A very
simple test uses the
root's `scripts/run_simple_test.py` to create an independent Claude session on
Noir, even when the main session itself is Claude or Cursor. Debugging belongs
to the Opus, Fable, or Astra brain after it reads actual test evidence.
