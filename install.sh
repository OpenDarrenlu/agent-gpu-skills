#!/bin/bash
# GPU Skill 安装脚本
# 用法: bash install.sh [--agent cursor|claude|codex|gemini] [--copy] [--no-veloq] [--no-nvidia-skills]
#
# 默认安装到 Cursor。使用 --agent 选择目标工具。
#
# 安装模式（默认混合模式）:
#   - skill 目录: 真实目录（多数工具不识别软链接目录）
#   - SKILL.md: 复制真实文件
#   - repos、references 等子目录/文件: 软链接到项目目录
#
# --copy  全量复制模式（适用于无法软链接的场景）

set -e

# Bash 版本检查: declare -A 需要 4.0+
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    for brew_bash in /opt/homebrew/bin/bash /usr/local/bin/bash; do
        if [ -x "$brew_bash" ]; then
            exec "$brew_bash" "$0" "$@"
        fi
    done
    echo "错误: 需要 Bash 4.0+ (当前 Bash $BASH_VERSION)"
    echo "macOS 用户请运行: brew install bash"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AGENT="cursor"
COPY_MODE=false
INSTALL_VELOQ=true
INSTALL_NVIDIA=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --agent)    AGENT="$2"; shift 2 ;;
        --copy)     COPY_MODE=true; shift ;;
        --no-veloq) INSTALL_VELOQ=false; shift ;;
        --no-nvidia-skills) INSTALL_NVIDIA=false; shift ;;
        -h|--help)
            echo "用法: bash install.sh [--agent cursor|claude|codex|gemini] [--copy] [--no-veloq] [--no-nvidia-skills]"
            echo ""
            echo "首次安装:"
            echo "  bash update-repos.sh    # 获取源码 repo (含 veloq 二进制 + NVIDIA skills)"
            echo "  bash install.sh         # 安装到 Cursor (默认，已验证)"
            echo ""
            echo "安装到其他工具 (未验证，如遇问题让对应 AI 协助排查):"
            echo "  bash install.sh --agent claude   # Claude Code (~/.claude/skills/)"
            echo "  bash install.sh --agent codex    # Codex (~/.codex/skills/)"
            echo "  bash install.sh --agent gemini   # Gemini CLI (~/.gemini/skills/)"
            echo ""
            echo "选项:"
            echo "  --copy            全量复制（适用于无法软链接的场景）"
            echo "  --no-veloq        跳过 VeloQ（profile 查询 CLI + nsys/ncu-profile-analysis skill）"
            echo "  --no-nvidia-skills 跳过 NVIDIA 官方 skills（200+ 个，可能较多）"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

get_skill_dir() {
    case $1 in
        cursor) echo "${HOME}/.cursor/skills" ;;
        claude) echo "${HOME}/.claude/skills" ;;
        codex)  echo "${HOME}/.codex/skills" ;;
        gemini) echo "${HOME}/.gemini/skills" ;;
        *)      echo "Unknown agent: $1" >&2; return 1 ;;
    esac
}

if [ ! -d "$SCRIPT_DIR/cuda_skill" ]; then
    echo "错误: 未找到 cuda_skill/ 目录"
    echo "请在项目根目录下运行此脚本"
    exit 1
fi

# 使用平行数组替代关联数组（兼容 Bash 3.2）
SKILL_NAMES=(
    cuda-skill
    triton-skill
    cutlass-skill
    sglang-skill
    nv-gpu-kernel-performance-modeling
    colfax-research-skill
    gpu-communication-libraries
    ncu-persistent-kernel-diagnosis
    persistent-kernel-scheduling
    persistent-kernel-utilization
    ncu-report-skill
)
SKILL_DIRS=(
    cuda_skill
    triton_skill
    cutlass_skill
    sglang_skill
    nv-gpu-kernel-performance-modeling
    colfax-research-skill
    gpu-communication-libraries
    ncu-persistent-kernel-diagnosis
    persistent-kernel-scheduling
    persistent-kernel-utilization
    ncu-report-skill
)

# 检查是否为本地 skill（避免 NVIDIA skills 覆盖本地）
is_local_skill() {
    local name="$1"
    for n in "${SKILL_NAMES[@]}"; do
        [ "$n" = "$name" ] && return 0
    done
    return 1
}

install_to_agent() {
    local agent=$1
    local SKILL_DIR
    SKILL_DIR=$(get_skill_dir "$agent")

    echo "================================"
    echo "安装到 $agent ($SKILL_DIR)"
    echo "================================"
    echo ""

    mkdir -p "$SKILL_DIR"

    for i in "${!SKILL_NAMES[@]}"; do
        skill_name="${SKILL_NAMES[$i]}"
        src_dir="${SKILL_DIRS[$i]}"
        src_path="$SCRIPT_DIR/$src_dir"
        target="$SKILL_DIR/$skill_name"

        echo "--- $skill_name ---"

        # 清理旧安装
        if [ "$skill_name" = "triton-skill" ]; then
            old_target="$SKILL_DIR/triton-gluon-skill"
            if [ -L "$old_target" ] || [ -d "$old_target" ]; then
                echo "  移除旧版: triton-gluon-skill"
                rm -rf "$old_target"
            fi
        fi

        if [ -L "$target" ]; then
            rm "$target"
        elif [ -d "$target" ]; then
            rm -rf "$target"
        fi

        if [ ! -d "$src_path" ]; then
            echo "  跳过: $src_dir/ 不存在"
            continue
        fi

        if [ "$COPY_MODE" = true ]; then
            cp -r "$src_path" "$target"
            echo "  已复制: $src_path -> $target"
        else
            mkdir -p "$target"
            cp "$src_path/SKILL.md" "$target/SKILL.md"
            echo "  已复制: SKILL.md"

            for item in "$src_path"/*; do
                basename="$(basename "$item")"
                [ "$basename" = "SKILL.md" ] && continue
                [[ "$basename" == update-*.sh ]] && continue
                [[ "$basename" == *.skill ]] && continue
                ln -sf "$item" "$target/$basename"
                echo "  已链接: $basename"
            done
        fi
    done
    echo ""

    # NVIDIA skills 自动遍历安装
    if [ "$INSTALL_NVIDIA" = true ]; then
        local nvidia_bases=(
            "$SCRIPT_DIR/repos/nvidia-skills/skills"
            "$SCRIPT_DIR/repos/nvidia-skills/plugins/nvidia-skills/skills"
        )
        local nvidia_installed=0
        local nvidia_skipped=0

        for nvidia_base in "${nvidia_bases[@]}"; do
            [ -d "$nvidia_base" ] || continue

            for skill_dir in "$nvidia_base"/*; do
                [ -d "$skill_dir" ] || continue
                [ -f "$skill_dir/SKILL.md" ] || continue

                local skill_name
                skill_name="$(basename "$skill_dir")"
                local target="$SKILL_DIR/$skill_name"

                # 如果与本地 skill 同名，优先保留本地（避免覆盖）
                if is_local_skill "$skill_name"; then
                    nvidia_skipped=$((nvidia_skipped + 1))
                    continue
                fi

                # 清理旧安装
                if [ -L "$target" ]; then
                    rm "$target"
                elif [ -d "$target" ]; then
                    rm -rf "$target"
                fi

                if [ "$COPY_MODE" = true ]; then
                    cp -r "$skill_dir" "$target"
                else
                    mkdir -p "$target"
                    cp "$skill_dir/SKILL.md" "$target/SKILL.md"
                    for item in "$skill_dir"/*; do
                        local basename_item
                        basename_item="$(basename "$item")"
                        [ "$basename_item" = "SKILL.md" ] && continue
                        [ -L "$target/$basename_item" ] && rm "$target/$basename_item"
                        ln -sf "$item" "$target/$basename_item" 2>/dev/null || true
                    done
                fi
                nvidia_installed=$((nvidia_installed + 1))
            done
        done

        if [ $nvidia_installed -gt 0 ]; then
            echo "--- NVIDIA skills ($nvidia_installed 个) ---"
            echo "  已安装 $nvidia_installed 个 NVIDIA 官方 skill"
            echo "  (跳过与本地 skill 同名的 $nvidia_skipped 个)"
            echo ""
        fi
    fi
}

install_to_agent "$AGENT"

# VeloQ：profile 查询 CLI + 两个 profiling skill。不 vendored，委托其官方安装器。
# 非致命：失败只 warn，不影响上面已装好的 skill。
if [ "$INSTALL_VELOQ" = true ] && [ -f "$SCRIPT_DIR/install-veloq.sh" ]; then
    echo "================================"
    echo "VeloQ (profile 查询 CLI + skill)"
    echo "================================"
    bash "$SCRIPT_DIR/install-veloq.sh" --agent "$AGENT" || \
        echo "提示: VeloQ 安装未完成（用 --no-veloq 可跳过；或参考 VeloQ README 手动安装）"
fi

# 验证
echo "================================"
echo "验证"
echo "================================"
echo ""

verify_agent() {
    local agent=$1
    local SKILL_DIR
    SKILL_DIR=$(get_skill_dir "$agent")
    local PASS=0 FAIL=0

    echo "--- $agent ($SKILL_DIR) ---"

    check() {
        if [ -e "$1" ]; then
            echo "  OK: $2"
            PASS=$((PASS + 1))
        else
            echo "  缺失: $2"
            FAIL=$((FAIL + 1))
        fi
    }

    for i in "${!SKILL_NAMES[@]}"; do
        skill_name="${SKILL_NAMES[$i]}"
        check "$SKILL_DIR/$skill_name/SKILL.md" "$skill_name/SKILL.md"
    done

    REFS="$SCRIPT_DIR/cuda_skill/references"
    check "$REFS/ptx-docs" "CUDA 文档: ptx-docs"
    check "$REFS/cuda-guide" "CUDA 文档: cuda-guide"
    check "$REFS/cuda-runtime-docs" "CUDA 文档: cuda-runtime-docs"
    check "$REFS/cuda-driver-docs" "CUDA 文档: cuda-driver-docs"

    local TRITON_REPO="$SKILL_DIR/triton-skill/repos/triton"
    check "$TRITON_REPO/python/tutorials" "Triton 教程"
    check "$TRITON_REPO/python/tutorials/gluon" "Gluon 教程"

    local CUTLASS_REPO="$SKILL_DIR/cutlass-skill/repos/cutlass"
    check "$CUTLASS_REPO/python/CuTeDSL" "CuTeDSL source"
    check "$CUTLASS_REPO/include/cute" "CuTe headers"

    local SGLANG_REPO="$SKILL_DIR/sglang-skill/repos/sglang"
    check "$SGLANG_REPO/python/sglang/srt" "SGLang SRT core"
    check "$SGLANG_REPO/sgl-kernel/csrc" "sgl-kernel CUDA source"

    local PERF_SKILL="$SKILL_DIR/nv-gpu-kernel-performance-modeling"
    check "$PERF_SKILL/research" "性能建模: research"

    local COLFAX_SKILL="$SKILL_DIR/colfax-research-skill"
    check "$COLFAX_SKILL/colfax_knowledge_base/metadata.json" "Colfax: 文章索引 metadata.json"
    check "$COLFAX_SKILL/colfax_knowledge_base/articles" "Colfax: articles"
    check "$COLFAX_SKILL/scripts/update_kb.py" "Colfax: 更新脚本"

    if [ "$INSTALL_VELOQ" = true ]; then
        check "$SKILL_DIR/nsys-profile-analysis/SKILL.md" "VeloQ: nsys-profile-analysis"
        check "$SKILL_DIR/ncu-profile-analysis/SKILL.md" "VeloQ: ncu-profile-analysis"
        if command -v veloq >/dev/null 2>&1; then
            echo "  OK: veloq 二进制 ($(veloq --version 2>/dev/null | head -1))"
            PASS=$((PASS + 1))
        else
            echo "  缺失: veloq 二进制 (PATH 未找到；见 install-veloq.sh)"
            FAIL=$((FAIL + 1))
        fi
    fi

    # NVIDIA skills 验证
    if [ "$INSTALL_NVIDIA" = true ]; then
        local nvidia_bases=(
            "$SCRIPT_DIR/repos/nvidia-skills/skills"
            "$SCRIPT_DIR/repos/nvidia-skills/plugins/nvidia-skills/skills"
        )
        local nvidia_ok=0 nvidia_missing=0
        for nvidia_base in "${nvidia_bases[@]}"; do
            [ -d "$nvidia_base" ] || continue
            for skill_dir in "$nvidia_base"/*; do
                [ -d "$skill_dir" ] || continue
                [ -f "$skill_dir/SKILL.md" ] || continue
                local skill_name
                skill_name="$(basename "$skill_dir")"
                if [ -e "$SKILL_DIR/$skill_name/SKILL.md" ]; then
                    nvidia_ok=$((nvidia_ok + 1))
                else
                    nvidia_missing=$((nvidia_missing + 1))
                fi
            done
        done
        if [ $nvidia_ok -gt 0 ] || [ $nvidia_missing -gt 0 ]; then
            echo "  OK: NVIDIA skills $nvidia_ok 个已安装, $nvidia_missing 个缺失"
            PASS=$((PASS + 1))
        fi
    fi

    echo "  验证: $PASS 通过, $FAIL 失败"
    echo ""

    if [ $FAIL -gt 0 ]; then
        echo "  提示: 缺失路径可能影响 skill 搜索功能."
        echo "    - CUDA 文档: 运行 'uv run scrape_docs.py all --force'"
        echo "    - 源码 repo: 运行 'bash update-repos.sh'"
        echo ""
    fi
}

verify_agent "$AGENT"

echo "安装完成."
