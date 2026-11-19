#!/usr/bin/env bash
# Link the shared global agent config into Claude Code and Codex.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CANONICAL="$SCRIPT_DIR/AGENTS.md"
FORCE=0

if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--force]" >&2
  exit 2
fi

if [[ ! -f "$CANONICAL" ]]; then
  echo "missing canonical config: $CANONICAL" >&2
  exit 1
fi

for target in "$HOME/.claude/CLAUDE.md" "$HOME/.codex/AGENTS.md"; do
  mkdir -p -- "$(dirname -- "$target")"

  if [[ -L "$target" ]]; then
    current=$(readlink -f -- "$target" || true)
    if [[ "$current" == "$CANONICAL" ]]; then
      echo "already linked: $target"
      continue
    fi
    if [[ "$FORCE" -ne 1 ]]; then
      echo "refuse to replace existing symlink: $target -> $current" >&2
      echo "rerun with --force after checking it" >&2
      exit 1
    fi
    mv -f -- "$target" "$target.bak.$(date +%Y%m%d%H%M%S)"
  elif [[ -e "$target" ]]; then
    mv -f -- "$target" "$target.bak.$(date +%Y%m%d%H%M%S)"
  fi

  ln -s -- "$CANONICAL" "$target"
  if ! cmp -s -- "$CANONICAL" "$target"; then
    echo "verification failed: $target does not match canonical file" >&2
    exit 1
  fi
  echo "linked: $target -> $CANONICAL"
done
