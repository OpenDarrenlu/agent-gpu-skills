#!/bin/bash
# 单独安装 Codex/Claude 共用技能；--agent claude 同时启用官方本地插件。
# --dest 隔离安装不调用 Claude、不改宿主配置，也不枚举其他技能目录。

set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
DIR="$ROOT/skills/other/brain-and-hands"
ID="brain-and-hands@brain-and-hands"
AGENT=claude
DEST=""
UNINSTALL=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --agent) AGENT="${2:?--agent 缺少值}"; shift 2 ;;
        --dest) DEST="${2:?--dest 缺少值}"; shift 2 ;;
        --uninstall) UNINSTALL=true; shift ;;
        --help|-h) echo '用法: bash install-brain-and-hands.sh [--agent codex|claude] [--dest DIR] [--uninstall]'; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done
case "$AGENT" in
    codex) DEFAULT_DEST="${CODEX_HOME:-$HOME/.codex}/skills" ;;
    claude) DEFAULT_DEST="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills" ;;
    *) echo "不支持的 agent: $AGENT" >&2; exit 2 ;;
esac

if [ "$AGENT" = claude ] && [ -z "$DEST" ]; then
    command -v claude >/dev/null || { echo "未找到 claude CLI；可用 --dest 仅安装技能" >&2; exit 1; }
fi
TARGET="${DEST:-$DEFAULT_DEST}/brain-and-hands"
case "$TARGET" in
    /*) ;;
    *) echo "目标目录必须为绝对路径" >&2; exit 2 ;;
esac
if [ -L "$TARGET" ] || { [ -e "$TARGET" ] && [ ! -d "$TARGET" ]; }; then
    echo "保留现有目标（非托管目录）: $TARGET" >&2; exit 1
fi
if [ -d "$TARGET" ] && [ "$(cat "$TARGET/.brain-and-hands-source" 2>/dev/null || true)" != "$DIR" ]; then
    echo "保留现有非本安装器管理的技能: $TARGET" >&2; exit 1
fi

if [ "$UNINSTALL" = true ]; then
    if [ "$AGENT" = claude ] && [ -z "$DEST" ]; then
        claude plugin uninstall "$ID" --scope user
        claude plugin marketplace remove brain-and-hands
    fi
    # 已核对来源标记，仅删除本安装器创建的单个技能目录。
    if [ -d "$TARGET" ]; then rm -r -- "$TARGET"; fi
    exit 0
fi

command -v python3 >/dev/null || { echo "未找到 python3（技能脚本及安装器需要 Python 3）" >&2; exit 1; }
mkdir -p "$TARGET"
# 已知小插件目录；保留相对引用、许可和 Codex UI 元数据。
python3 - "$DIR" "$TARGET" <<'PY'
from pathlib import Path
import shutil
import sys
source, target = map(Path, sys.argv[1:])
if source.resolve() == target.resolve():
    raise SystemExit("不能将技能安装到源码目录自身")
shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".claude-plugin"))
PY
printf '%s\n' "$DIR" > "$TARGET/.brain-and-hands-source"
echo "已安装 $AGENT skill: $TARGET/SKILL.md"

if [ "$AGENT" = claude ] && [ -z "$DEST" ]; then
    claude plugin marketplace list 2>/dev/null | grep -q 'brain-and-hands' \
        || claude plugin marketplace add "$DIR" --scope user
    if claude plugin list 2>/dev/null | grep -q "$ID"; then
        claude plugin update "$ID" --scope user
    else
        claude plugin install "$ID" --scope user
    fi
    claude plugin list | grep -A4 "$ID"
fi
