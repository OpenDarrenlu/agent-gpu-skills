#!/bin/bash
# 一条命令完成 submodule / 外部来源获取、运行时 CLI 安装与 skill 安装。
# 用法: bash bootstrap.sh [--agent cursor|claude|codex|gemini|kimi] [选项] [install.sh 其他选项]
#
# 每一步都会执行并记录退出码；任一步失败时最终返回非零，并在末尾说明。不会因为前一步失败而跳过 install.sh，
# 这样缺失项会在同一份报告里列出来。

if [ "${BASH_VERSINFO[0]}" -lt 3 ]; then
    echo "错误: 需要 Bash 3.0+ (当前 Bash $BASH_VERSION)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AGENT="cursor"
RUN_UPDATE=true
INTEGRATIONS=true
REA_MCP=true
DEST_DIR=""
INSTALL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --dest)
            DEST_DIR="$2"
            INSTALL_ARGS+=("--dest" "$2")
            shift 2
            ;;
        --no-update)
            RUN_UPDATE=false
            shift
            ;;
        --no-integrations)
            INTEGRATIONS=false
            shift
            ;;
        --no-rea-mcp)
            REA_MCP=false
            shift
            ;;
        -h|--help)
            echo "用法: bash bootstrap.sh [--agent cursor|claude|codex|gemini|kimi] [选项] [install.sh 其他选项]"
            echo ""
            echo "示例:"
            echo "  bash bootstrap.sh --agent codex"
            echo "  bash bootstrap.sh --agent cursor --copy"
            echo "  bash bootstrap.sh --agent claude --no-nvidia-skills"
            echo "  bash bootstrap.sh --agent kimi"
            echo "  bash bootstrap.sh --agent codex --dest /tmp/agent-gpu-skills-test --no-veloq   # 隔离验证，不写真实 HOME"
            echo ""
            echo "选项:"
            echo "  --no-update       跳过 update-repos.sh（不拉取已有仓库的新版本）。仍会补齐缺失的 submodule 与外部来源"
            echo "  --no-integrations 跳过 sources/external-sources.conf 中的外部来源：不获取源码、不安装运行时 CLI、不安装其 skill、不注册 REA MCP"
            echo "  --no-rea-mcp      跳过 REA MCP 注册（只对 codex 生效；使用 REA 自身的 rea setup --client codex --skill=false --yes）"
            echo "  --dest DIR        隔离安装：只写 DIR 下的 skill 目录。不初始化 submodule、不运行 update-repos.sh、"
            echo "                    不获取外部来源、不安装运行时 CLI（这些会联网或写入真实 HOME）"
            echo "  其他参数会原样传给 install.sh，例如 --copy、--no-veloq、--no-nvidia-skills、--no-amd-skills、--no-cursor-skills、--no-ccfa-skills"
            echo ""
            echo "退出码: 任一步骤（仓库更新、外部来源与运行时、install.sh 的缺失检查）失败时为非零。"
            exit 0
            ;;
        *)
            INSTALL_ARGS+=("$1")
            shift
            ;;
    esac
done

# FETCH: 是否获取外部来源与安装其运行时（--dest 与 --no-integrations 都会关闭）。
# INTEGRATIONS: 是否让 install.sh 处理外部来源的 skill（--dest 时仍处理已存在的来源）。
FETCH=true
if [ -n "$DEST_DIR" ]; then
    RUN_UPDATE=false
    FETCH=false
fi
if [ "$INTEGRATIONS" = false ]; then
    FETCH=false
    INSTALL_ARGS+=("--no-integrations")
fi

echo "================================"
echo "Bootstrap agent-gpu-skills"
echo "================================"
echo "Agent: $AGENT"
if [ -n "$DEST_DIR" ]; then
    echo "模式: --dest 隔离安装（${DEST_DIR}）；不初始化 submodule，不更新仓库，不获取外部来源，不安装运行时 CLI"
fi
if [ "$INTEGRATIONS" = false ] && [ -z "$DEST_DIR" ]; then
    echo "模式: --no-integrations（跳过外部来源与其运行时 CLI）"
fi
if [ "$INTEGRATIONS" = true ] && [ -n "$DEST_DIR" ]; then
    echo "提示: --dest 下只安装已经存在于 repos/ 中的外部来源 skill，不获取、不安装运行时"
fi
echo ""

STATUS=0

if [ -z "$DEST_DIR" ] && [ -f "$SCRIPT_DIR/.gitmodules" ]; then
    echo "================================"
    echo "初始化 submodules（缺失时总是执行，即使 --no-update）"
    echo "================================"
    git -C "$SCRIPT_DIR" submodule update --init --recursive || \
        echo "warn: submodule 初始化未完成；install.sh 会列出缺失的组件"
    echo ""
fi

if [ "$RUN_UPDATE" = true ]; then
    echo "================================"
    echo "更新仓库（update-repos.sh：源码、外部来源与 runtime）"
    echo "================================"
    UPDATE_ARGS=()
    if [ "$FETCH" = false ]; then
        UPDATE_ARGS+=("--no-integrations")
    fi
    bash "$SCRIPT_DIR/update-repos.sh" ${UPDATE_ARGS[@]+"${UPDATE_ARGS[@]}"} || STATUS=$?
    echo ""
elif [ "$FETCH" = true ]; then
    echo "================================"
    echo "补齐外部来源（--no-update：只 clone 缺失项，不拉取已有 checkout；安装运行时 CLI）"
    echo "================================"
    bash "$SCRIPT_DIR/install-external.sh" || STATUS=$?
    echo ""
fi

echo "================================"
echo "安装 skills"
echo "================================"
bash "$SCRIPT_DIR/install.sh" --agent "$AGENT" "${INSTALL_ARGS[@]}" || STATUS=$?
echo ""

# REA MCP：只在非 --dest、未 --no-integrations、未 --no-rea-mcp 时执行；只注册 codex。
if [ "$FETCH" = true ] && [ "$REA_MCP" = true ]; then
    echo "================================"
    echo "REA MCP（仅 codex；由清单中的 rea-agents 运行时决定是否执行）"
    echo "================================"
    bash "$SCRIPT_DIR/install-rea-mcp.sh" --agent "$AGENT" || STATUS=$?
    echo ""
fi

# 路由清单：在所有来源、CLI 与 skill 准备完成之后重新生成，使新来源在同一次运行中即可被路由器找到。
# --dest 隔离安装不修改受 git 跟踪的 inventory.tsv，只提示。
echo "================================"
echo "路由清单（agent-skill-router/references/inventory.tsv）"
echo "================================"
if [ -n "$DEST_DIR" ]; then
    echo "--dest 模式：不修改受 git 跟踪的清单；路由器读取的是已提交的清单。真实安装时会自动重新生成。"
elif [ -f "$SCRIPT_DIR/skills/routers/agent-skill-router/scripts/build_inventory.py" ]; then
    python3 "$SCRIPT_DIR/skills/routers/agent-skill-router/scripts/build_inventory.py" || STATUS=$?
else
    echo "FAIL: 缺少 build_inventory.py，路由清单未生成"
    STATUS=1
fi
echo ""
if [ $STATUS -ne 0 ]; then
    echo "Bootstrap 未完成（退出码 ${STATUS}）：请查看上方 FAIL / BLOCKED / 缺失 项。不要把本次运行视为成功安装。"
    exit $STATUS
fi
echo "Bootstrap 完成。若 agent 已经运行，请重启或新开会话以加载新增/更新的 skills。"
