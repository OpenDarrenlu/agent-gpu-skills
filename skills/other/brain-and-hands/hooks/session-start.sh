#!/usr/bin/env bash
# Reminder only: no model sessions or commands are launched by this hook.
cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "brain-and-hands: preserve the user's strong Astra/Opus main session as the brain. For an authorized split, ordinary tests use Sol/Sonnet and very simple exact-command tests use a fresh independent Claude session on a7n Noir via the plugin root's scripts/run_simple_test.py. Astra/Opus reads actual test evidence and code to diagnose bugs; cheap test hands do not decide root causes. Read the delegate skill for the shared workflow before launching a hand. Respect host authorization and scoped paths; do not revive disabled relays. No recursive delegation or fixed sharding; serialize shared GPU/port tests. Use completion notifications or bounded waits and finish the required work before the final answer. Verify actual exit statuses, artifacts and diffs; old Gris prices/accuracy do not establish Noir quality or cost."
  }
}
EOF
