---
name: delegate
description: >
  Choose brain-and-hands roles for an authorized Claude or Codex workflow, especially
  independent tests and debugging: Noir for very simple tests, Sonnet/Sol for ordinary
  test execution, and Opus/Astra for code and failure analysis.
---

# Brain and hands plugin entry

Read [the shared skill](../../SKILL.md) before choosing or launching a hand. It is
the canonical workflow for both hosts; do not duplicate its rules in this entry.
Resolve references and scripts from the plugin root, `${CLAUDE_PLUGIN_ROOT}`, when
the plugin loader presents this skill through a different working directory.

An already launched hand completes its brief without further delegation. A very
simple test uses the root's `scripts/run_simple_test.py` to create an independent
Claude session on Noir, even when the main session itself is Claude. Debugging
belongs to the Opus/Astra brain after it reads actual test evidence.
