#!/bin/bash
# 一条命令完成外部 repo 获取与 skill 安装。
# 用法: bash bootstrap.sh [--agent cursor|claude|codex|gemini] [install.sh 其他选项]

set -e

if [ "${BASH_VERSINFO[0]}" -lt 3 ]; then
    echo "错误: 需要 Bash 3.0+ (当前 Bash $BASH_VERSION)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AGENT="cursor"
RUN_UPDATE=true
INSTALL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --no-update)
            RUN_UPDATE=false
            shift
            ;;
        -h|--help)
            echo "用法: bash bootstrap.sh [--agent cursor|claude|codex|gemini] [install.sh 其他选项]"
            echo ""
            echo "示例:"
            echo "  bash bootstrap.sh --agent codex"
            echo "  bash bootstrap.sh --agent cursor --copy"
            echo "  bash bootstrap.sh --agent claude --no-nvidia-skills"
            echo ""
            echo "选项:"
            echo "  --no-update       跳过 update-repos.sh，只运行 install.sh"
            echo "  其他参数会原样传给 install.sh，例如 --copy、--no-veloq、--no-nvidia-skills、--no-cursor-skills"
            exit 0
            ;;
        *)
            INSTALL_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "================================"
echo "Bootstrap agent-gpu-skills"
echo "================================"
echo "Agent: $AGENT"
echo ""

if [ "$RUN_UPDATE" = true ]; then
    echo "================================"
    echo "更新 submodules 与外部 repos"
    echo "================================"
    git -C "$SCRIPT_DIR" submodule update --init --recursive || true
    bash "$SCRIPT_DIR/update-repos.sh"
fi

echo "================================"
echo "安装 skills"
echo "================================"
bash "$SCRIPT_DIR/install.sh" --agent "$AGENT" "${INSTALL_ARGS[@]}"

echo ""
echo "Bootstrap 完成。若 agent 已经运行，请重启或新开会话以加载新增/更新的 skills。"
