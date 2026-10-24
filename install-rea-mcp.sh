#!/bin/bash
# 为 Codex 注册 REA MCP：rea setup --client codex --skill=false --yes --json。
#
# 范围（刻意收窄）:
#   - 只在清单 sources/external-sources.conf 声明了 rea-agents 运行时时执行；
#   - 只注册 codex（其他 agent 只得到 CLI 与 skill，不会被报告为已注册 MCP）；
#   - 只使用 REA 自身的 setup：保留其他配置并做备份（见 repos/rea/docs/installation.md）。
#     本脚本不解析、不编辑 ~/.codex/config.toml，也不读取任何密钥；
#   - 不使用 --all-detected、--install-hopper，不安装 Hopper/Ghidra/IDA 等原生 provider。
#
# 用法:
#   bash install-rea-mcp.sh --agent codex [--dry-run]
#   --dry-run  只输出 REA 的计划（rea setup --client codex --skill=false --dry-run --json），不写配置。
#
# 退出码: 需要注册但 rea 不可用、Node 不兼容或 rea setup 失败时为 1。

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$SCRIPT_DIR/sources/external-sources.conf"
AGENT=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)   AGENT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

if [ "$AGENT" != codex ]; then
    echo "REA MCP: 仅注册到 codex。agent=${AGENT:-未指定} 跳过（REA CLI 与 skill 仍由 install-external.sh/install.sh 安装，但不会报告为已注册 MCP）"
    exit 0
fi

# 清单中的 rea-agents 运行时规格，形如 npm:rea-agents@6.7.0:rea
RUNTIME_SPEC="$(awk -F'|' '$1 !~ /^#/ && $6 ~ /^npm:rea-agents@/ {print $6; exit}' "$CONF" 2>/dev/null)"
if [ -z "$RUNTIME_SPEC" ]; then
    echo "REA MCP: 清单中没有 rea-agents 运行时，跳过"
    exit 0
fi
REA_BIN="${RUNTIME_SPEC##*:}"

# shellcheck source=scripts/node-preflight.sh
. "$SCRIPT_DIR/scripts/node-preflight.sh"

if ! command -v "$REA_BIN" >/dev/null 2>&1; then
    echo "  FAIL: 找不到 '$REA_BIN'（清单要求 ${RUNTIME_SPEC}）。先运行 bash install-external.sh"
    exit 1
fi
node_preflight rea-agents || exit 1

if [ "$DRY_RUN" = true ]; then
    echo "REA MCP 计划（不写配置）:"
    "$REA_BIN" setup --client codex --skill=false --dry-run --json </dev/null
    rc=$?
else
    echo "REA MCP: 注册到 codex（rea setup --client codex --skill=false --yes --json）"
    "$REA_BIN" setup --client codex --skill=false --yes --json </dev/null
    rc=$?
fi

if [ $rc -ne 0 ]; then
    echo "  FAIL: rea setup 失败（退出码 ${rc}）。查看上方 JSON 中的 status；needs_confirmation 表示需要人工确认"
    exit 1
fi
echo "REA MCP: 完成"
