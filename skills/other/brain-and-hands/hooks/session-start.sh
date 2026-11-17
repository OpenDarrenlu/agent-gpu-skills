#!/usr/bin/env bash
# Injects the short delegation rule at session start. The full doctrine lives in the
# `delegate` skill; the rule is inlined here because the skill is rarely loaded on its own.
cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "brain-and-hands: if you are the main (strong-model) session you are the brain: decide, brief, verify. Work inline by default; delegation costs briefing, wake-up and verification turns. Delegate a whole task only if its result is checkable (a test passes, or every claim cites path:line), it is self-contained, and inline would be large (about 100k tokens of raw material, 30+ tool calls, anything with screenshots) or can run in the background while you do other work. Cheapest hand first: brain-and-hands:grunt (free, slow: run it in the background, in about 4 shards), then :scout (Haiku, read-only), :worker / :researcher (Sonnet), :operator (Sonnet, screens). After launching hands never poll or sleep; end your turn or do other work, and make your last message the complete answer. Verify before relying on a report: run the check, spot-read cited lines, confirm audit candidates yourself. Load the `delegate` skill for the brief template. If you are a subagent yourself, ignore this."
  }
}
EOF
