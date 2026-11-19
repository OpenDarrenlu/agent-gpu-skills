#!/usr/bin/env bash
# Reminder only: no model sessions or commands are launched by this hook.
cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "brain-and-hands: preserve the user's strong Astra, Opus, or Fable main session as the brain. For an authorized split, ordinary tests use Sol or the latest Sonnet, and Cursor may instead pin another available hand model such as DeepSeek V4 Flash or Grok 4.7. Very simple exact-command tests use a fresh independent Claude session on a7n Noir via the plugin root's scripts/run_simple_test.py. The brain reads actual test evidence and code to diagnose bugs; cheap test hands do not decide root causes. In Cursor multitask mode, pass this skill path and an explicit model into every Task prompt. A specified brief whose logic is not especially complex uses a hand model and that subagent acts as a hand, even a step above an ordinary hand. A Fable or Opus subagent is reserved for design, ambiguous diagnosis, and the final diff check; a hand executes its brief without delegating. Read the delegate skill for the shared workflow before launching a hand. Respect host authorization and scoped paths; do not revive disabled relays. No fixed sharding; serialize shared GPU/port tests. Use completion notifications or bounded waits and finish the required work before the final answer. Verify actual exit statuses, artifacts and diffs; old Gris prices/accuracy do not establish Noir quality or cost."
  }
}
EOF
