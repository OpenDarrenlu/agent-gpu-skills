#!/bin/bash
# 为 Claude Code 启用 brain-and-hands 插件（user scope，幂等）：把本仓库的
# skills/other/brain-and-hands 注册为本地 marketplace 并安装。插件提供 delegate skill、
# scout/worker/researcher/operator/grunt 子 agent 和 SessionStart 提示。
#
# 用法: bash install-brain-and-hands.sh [--uninstall]
# 只调用 `claude plugin` 子命令，不解析、不编辑 ~/.claude/settings.json。

set -eu

DIR="$(cd "$(dirname "$0")" && pwd)/skills/other/brain-and-hands"
ID="brain-and-hands@brain-and-hands"

command -v claude >/dev/null || { echo "未找到 claude CLI，跳过" >&2; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
    claude plugin uninstall "$ID" --scope user || true
    claude plugin marketplace remove brain-and-hands || true
    exit 0
fi

claude plugin marketplace list 2>/dev/null | grep -q "brain-and-hands" \
    || claude plugin marketplace add "$DIR" --scope user
claude plugin list 2>/dev/null | grep -q "$ID" \
    || claude plugin install "$ID" --scope user
claude plugin list | grep -A4 "$ID"
