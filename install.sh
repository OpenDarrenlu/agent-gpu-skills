#!/bin/bash
# Agent Skill 安装脚本
# 用法: bash install.sh [--agent cursor|claude|codex|gemini|kimi] [--dest DIR] [--copy] [--no-veloq] [--no-nvidia-skills] [--no-amd-skills] [--no-cursor-skills] [--no-ccfa-skills]
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

# Bash 版本检查: 脚本只使用普通数组，macOS 自带 Bash 3.2 可运行。
if [ "${BASH_VERSINFO[0]}" -lt 3 ]; then
    echo "错误: 需要 Bash 3.0+ (当前 Bash $BASH_VERSION)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AGENT="cursor"
DEST_DIR=""
COPY_MODE=false
INSTALL_VELOQ=true
INSTALL_NVIDIA=true
INSTALL_AMD=true
INSTALL_CURSOR_SKILLS=true
INSTALL_CCFA_SKILLS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --agent)    AGENT="$2"; shift 2 ;;
        --dest)     DEST_DIR="$2"; shift 2 ;;
        --copy)     COPY_MODE=true; shift ;;
        --no-veloq) INSTALL_VELOQ=false; shift ;;
        --no-nvidia-skills) INSTALL_NVIDIA=false; shift ;;
        --no-amd-skills) INSTALL_AMD=false; shift ;;
        --no-cursor-skills) INSTALL_CURSOR_SKILLS=false; shift ;;
        --no-ccfa-skills) INSTALL_CCFA_SKILLS=false; shift ;;
        -h|--help)
            echo "用法: bash install.sh [--agent cursor|claude|codex|gemini|kimi] [--dest DIR] [--copy] [--no-veloq] [--no-nvidia-skills] [--no-amd-skills] [--no-cursor-skills] [--no-ccfa-skills]"
            echo ""
            echo "首次安装:"
            echo "  bash update-repos.sh    # 获取源码 repo (含 veloq 二进制 + NVIDIA/AMD/Cursor/CCFA skills)"
            echo "  bash install.sh         # 安装到 Cursor (默认，已验证)"
            echo ""
            echo "安装到其他工具 (未验证，如遇问题让对应 AI 协助排查):"
            echo "  bash install.sh --agent claude   # Claude Code (~/.claude/skills/)"
            echo "  bash install.sh --agent codex    # Codex (~/.codex/skills/)"
            echo "  bash install.sh --agent gemini   # Gemini CLI (~/.gemini/skills/)"
            echo "  bash install.sh --agent kimi     # Kimi Code CLI (~/.agents/skills/)"
            echo ""
            echo "选项:"
            echo "  --dest DIR        安装到自定义 skill 目录（便于隔离安装/CI 验证）"
            echo "  --copy            全量复制（适用于无法软链接的场景）"
            echo "  --no-veloq        跳过 VeloQ（profile 查询 CLI + nsys/ncu-profile-analysis skill）"
            echo "  --no-nvidia-skills 跳过 NVIDIA 官方 skills（200+ 个，可能较多）"
            echo "  --no-amd-skills   跳过 AMD 官方 skills（来自 amd/skills）"
            echo "  --no-cursor-skills 跳过 Saddss/cursor-skills 仓库中的 skills"
            echo "  --no-ccfa-skills  跳过 mikubaka88/CCFA-Skills 论文研究 skills"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

get_skill_dir() {
    if [ -n "$DEST_DIR" ]; then
        echo "$DEST_DIR"
        return 0
    fi
    case $1 in
        cursor) echo "${HOME}/.cursor/skills" ;;
        claude) echo "${HOME}/.claude/skills" ;;
        codex)  echo "${HOME}/.codex/skills" ;;
        gemini) echo "${HOME}/.gemini/skills" ;;
        kimi)   echo "${HOME}/.agents/skills" ;;
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
    deepgemm-skill
    sglang-skill
    nv-gpu-kernel-performance-modeling
    colfax-research-skill
    nccl-skill
    gpu-communication-libraries
    ncu-persistent-kernel-diagnosis
    persistent-kernel-scheduling
    persistent-kernel-utilization
    ncu-report-skill
    amd-gpu-docs
    amd-instinct-isa
    amd-instinct-cdna4-isa
    gpu-performance-router
    gpu-kernel-authoring-router
    llm-serving-router
    gpu-development-catchall
    iket-profiling
)
SKILL_DIRS=(
    cuda_skill
    triton_skill
    cutlass_skill
    deepgemm-skill
    sglang_skill
    nv-gpu-kernel-performance-modeling
    colfax-research-skill
    nccl_skill
    gpu-communication-libraries
    ncu-persistent-kernel-diagnosis
    persistent-kernel-scheduling
    persistent-kernel-utilization
    ncu-report-skill
    amd-gpu-docs
    amd-instinct-isa
    amd-instinct-cdna4-isa
    gpu-performance-router
    gpu-kernel-authoring-router
    llm-serving-router
    gpu-development-catchall
    iket-profiling
)

# 检查是否为本地 skill（避免外部 catalog 覆盖本地路由/参考 skill）
is_local_skill() {
    local name="$1"
    for n in "${SKILL_NAMES[@]}"; do
        [ "$n" = "$name" ] && return 0
    done
    return 1
}

ensure_submodule_path() {
    local label="$1"
    local path="$2"
    local marker="$3"

    [ -e "$SCRIPT_DIR/$marker" ] && return 0

    if git -C "$SCRIPT_DIR" config -f .gitmodules --get "submodule.$path.url" >/dev/null 2>&1; then
        echo "  初始化 submodule: $label ($path)"
        if ! git -C "$SCRIPT_DIR" submodule update --init "$path"; then
            echo "  warn: 无法初始化 $label submodule；可手动运行 'git submodule update --init --recursive'" >&2
            return 1
        fi
    fi
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
        ensure_submodule_path "nvidia-skills" "repos/nvidia-skills" "repos/nvidia-skills/skills" || true

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

    # Cursor skills 自动遍历安装
    if [ "$INSTALL_CURSOR_SKILLS" = true ]; then
        ensure_submodule_path "cursor-skills" "repos/cursor-skills" "repos/cursor-skills/skills" || true

        local cursor_base="$SCRIPT_DIR/repos/cursor-skills/skills"
        local cursor_installed=0
        local cursor_skipped=0

        if [ -d "$cursor_base" ]; then
            for skill_dir in "$cursor_base"/*; do
                [ -d "$skill_dir" ] || continue
                [ -f "$skill_dir/SKILL.md" ] || continue

                local skill_name
                skill_name="$(basename "$skill_dir")"
                local target="$SKILL_DIR/$skill_name"

                # 如果与本地 skill 同名，优先保留本地（避免覆盖）
                if is_local_skill "$skill_name"; then
                    cursor_skipped=$((cursor_skipped + 1))
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
                cursor_installed=$((cursor_installed + 1))
            done

            if [ $cursor_installed -gt 0 ]; then
                echo "--- Cursor skills ($cursor_installed 个) ---"
                echo "  已安装 $cursor_installed 个 Saddss/cursor-skills skill"
                echo "  (跳过与本地 skill 同名的 $cursor_skipped 个)"
                echo ""
            fi
        else
            echo "--- Cursor skills ---"
            echo "  跳过: 未找到 repos/cursor-skills/skills"
            echo "  提示: 运行 'git submodule update --init repos/cursor-skills' 或 'bash update-repos.sh cursor-skills'"
            echo ""
        fi
    fi

    # CCFA Skills 自动遍历安装。上游完整安装语义是复制所有 ccf-* 目录：
    # 其中 17 个目录是 runtime skill，ccf-latex-templates 是项目脚手架依赖。
    if [ "$INSTALL_CCFA_SKILLS" = true ]; then
        ensure_submodule_path "ccfa-skills" "repos/ccfa-skills" "repos/ccfa-skills/ccf-common/SKILL.md" || true

        local ccfa_base="$SCRIPT_DIR/repos/ccfa-skills"
        local ccfa_installed=0
        local ccfa_support_installed=0
        local ccfa_skipped=0

        if [ -d "$ccfa_base" ]; then
            for ccfa_dir in "$ccfa_base"/ccf-*; do
                [ -d "$ccfa_dir" ] || continue

                local entry_name
                entry_name="$(basename "$ccfa_dir")"
                local target="$SKILL_DIR/$entry_name"

                if [ -f "$ccfa_dir/SKILL.md" ] && is_local_skill "$entry_name"; then
                    ccfa_skipped=$((ccfa_skipped + 1))
                    continue
                fi

                if [ -L "$target" ]; then
                    rm "$target"
                elif [ -d "$target" ]; then
                    rm -rf "$target"
                elif [ -e "$target" ]; then
                    rm "$target"
                fi

                if [ -f "$ccfa_dir/SKILL.md" ]; then
                    if [ "$COPY_MODE" = true ]; then
                        cp -r "$ccfa_dir" "$target"
                    else
                        mkdir -p "$target"
                        cp "$ccfa_dir/SKILL.md" "$target/SKILL.md"
                        for item in "$ccfa_dir"/*; do
                            local basename_item
                            basename_item="$(basename "$item")"
                            [ "$basename_item" = "SKILL.md" ] && continue
                            [ -L "$target/$basename_item" ] && rm "$target/$basename_item"
                            ln -sf "$item" "$target/$basename_item" 2>/dev/null || true
                        done
                    fi
                    ccfa_installed=$((ccfa_installed + 1))
                else
                    # 支撑目录不参与 skill 发现，混合模式可直接链接整个目录。
                    if [ "$COPY_MODE" = true ]; then
                        cp -r "$ccfa_dir" "$target"
                    else
                        ln -s "$ccfa_dir" "$target"
                    fi
                    ccfa_support_installed=$((ccfa_support_installed + 1))
                fi
            done

            echo "--- CCFA Skills ($ccfa_installed 个) ---"
            echo "  已安装 $ccfa_installed 个 CCFA runtime skill"
            echo "  已安装 $ccfa_support_installed 个 CCFA 支撑目录"
            echo "  (跳过与本地 skill 同名的 $ccfa_skipped 个)"
            echo ""
        else
            echo "--- CCFA Skills ---"
            echo "  跳过: 未找到 repos/ccfa-skills"
            echo "  提示: 运行 'git submodule update --init repos/ccfa-skills' 或 'bash update-repos.sh ccfa-skills'"
            echo ""
        fi
    fi

    # AMD 官方 skills 自动遍历安装（amd/skills）
    if [ "$INSTALL_AMD" = true ]; then
        ensure_submodule_path "amd-skills" "repos/amd-skills" "repos/amd-skills/skills" || true

        local amd_base="$SCRIPT_DIR/repos/amd-skills/skills"
        local amd_installed=0
        local amd_skipped=0

        if [ -d "$amd_base" ]; then
            for skill_dir in "$amd_base"/*; do
                [ -d "$skill_dir" ] || continue
                [ -f "$skill_dir/SKILL.md" ] || continue

                local skill_name
                skill_name="$(basename "$skill_dir")"
                local target="$SKILL_DIR/$skill_name"

                # 本地 skill 优先，避免上游新增同名目录覆盖仓库路由。
                if is_local_skill "$skill_name"; then
                    amd_skipped=$((amd_skipped + 1))
                    continue
                fi

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
                amd_installed=$((amd_installed + 1))
            done

            echo "--- AMD skills ($amd_installed 个) ---"
            echo "  已安装 $amd_installed 个 amd/skills 官方 skill"
            echo "  (跳过与本地 skill 同名的 $amd_skipped 个)"
            echo ""
        else
            echo "--- AMD skills ---"
            echo "  跳过: 未找到 repos/amd-skills/skills"
            echo "  提示: 运行 'git submodule update --init repos/amd-skills' 或 'bash update-repos.sh amd-skills'"
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

    local DEEPGEMM_REPO="$SKILL_DIR/deepgemm-skill/repos/deepgemm"
    check "$DEEPGEMM_REPO/csrc/apis/gemm.hpp" "DeepGEMM grouped GEMM API"
    check "$DEEPGEMM_REPO/deep_gemm/include/deep_gemm/scheduler/gemm.cuh" "DeepGEMM grouped scheduler"
    check "$DEEPGEMM_REPO/tests/test_fp8_fp4.py" "DeepGEMM grouped GEMM tests"
    check "$DEEPGEMM_REPO/third-party/cutlass/include/cute" "DeepGEMM CUTLASS submodule"

    local SGLANG_REPO="$SKILL_DIR/sglang-skill/repos/sglang"
    check "$SGLANG_REPO/python/sglang/srt" "SGLang SRT core"
    check "$SGLANG_REPO/python/sglang/kernels/aot/csrc" "sgl-kernel CUDA source (kernels/aot)"

    local NCCL_REPO="$SKILL_DIR/nccl-skill/repos/nccl"
    check "$NCCL_REPO/src/nccl.h.in" "NCCL public API header"
    check "$NCCL_REPO/src/device" "NCCL device kernels"
    check "$NCCL_REPO/plugins/net" "NCCL net plugin headers"
    check "$SKILL_DIR/nccl-skill/references" "NCCL references"

    local PERF_SKILL="$SKILL_DIR/nv-gpu-kernel-performance-modeling"
    check "$PERF_SKILL/research" "性能建模: research"

    local COLFAX_SKILL="$SKILL_DIR/colfax-research-skill"
    check "$COLFAX_SKILL/colfax_knowledge_base/metadata.json" "Colfax: 文章索引 metadata.json"
    check "$COLFAX_SKILL/colfax_knowledge_base/articles" "Colfax: articles"
    check "$COLFAX_SKILL/scripts/update_kb.py" "Colfax: 更新脚本"

    local NCU_SKILL="$SKILL_DIR/ncu-report-skill"
    check "$NCU_SKILL/references/ProfilingGuide.md" "NCU ProfilingGuide (参考文档)"

    if [ "$INSTALL_VELOQ" = true ]; then
        if [ -e "$SKILL_DIR/nsys-profile-analysis/SKILL.md" ]; then
            echo "  OK: VeloQ: nsys-profile-analysis"
            PASS=$((PASS + 1))
        else
            echo "  可选缺失: VeloQ: nsys-profile-analysis (用 --no-veloq 可跳过)"
        fi
        if [ -e "$SKILL_DIR/ncu-profile-analysis/SKILL.md" ]; then
            echo "  OK: VeloQ: ncu-profile-analysis"
            PASS=$((PASS + 1))
        else
            echo "  可选缺失: VeloQ: ncu-profile-analysis (用 --no-veloq 可跳过)"
        fi
        if command -v veloq >/dev/null 2>&1; then
            echo "  OK: veloq 二进制 ($(veloq --version 2>/dev/null | head -1))"
            PASS=$((PASS + 1))
        else
            echo "  可选缺失: veloq 二进制 (PATH 未找到；不影响其他 skills；见 install-veloq.sh 或用 --no-veloq)"
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

    # AMD skills 验证
    if [ "$INSTALL_AMD" = true ]; then
        local amd_base="$SCRIPT_DIR/repos/amd-skills/skills"
        local amd_ok=0 amd_missing=0
        if [ -d "$amd_base" ]; then
            for skill_dir in "$amd_base"/*; do
                [ -d "$skill_dir" ] || continue
                [ -f "$skill_dir/SKILL.md" ] || continue
                local skill_name
                skill_name="$(basename "$skill_dir")"
                if is_local_skill "$skill_name"; then
                    continue
                fi
                if [ -e "$SKILL_DIR/$skill_name/SKILL.md" ]; then
                    amd_ok=$((amd_ok + 1))
                else
                    amd_missing=$((amd_missing + 1))
                fi
            done
        fi
        if [ $amd_ok -gt 0 ] || [ $amd_missing -gt 0 ]; then
            echo "  OK: AMD skills $amd_ok 个已安装, $amd_missing 个缺失"
            PASS=$((PASS + 1))
        fi
    fi

    # Cursor skills 验证
    if [ "$INSTALL_CURSOR_SKILLS" = true ]; then
        local cursor_base="$SCRIPT_DIR/repos/cursor-skills/skills"
        local cursor_ok=0 cursor_missing=0
        if [ -d "$cursor_base" ]; then
            for skill_dir in "$cursor_base"/*; do
                [ -d "$skill_dir" ] || continue
                [ -f "$skill_dir/SKILL.md" ] || continue
                local skill_name
                skill_name="$(basename "$skill_dir")"
                if is_local_skill "$skill_name"; then
                    continue
                fi
                if [ -e "$SKILL_DIR/$skill_name/SKILL.md" ]; then
                    cursor_ok=$((cursor_ok + 1))
                else
                    cursor_missing=$((cursor_missing + 1))
                fi
            done
        fi
        if [ $cursor_ok -gt 0 ] || [ $cursor_missing -gt 0 ]; then
            echo "  OK: Cursor skills $cursor_ok 个已安装, $cursor_missing 个缺失"
            PASS=$((PASS + 1))
        fi
    fi

    # CCFA Skills 验证（包含无 SKILL.md 的 ccf-latex-templates 支撑目录）
    if [ "$INSTALL_CCFA_SKILLS" = true ]; then
        local ccfa_base="$SCRIPT_DIR/repos/ccfa-skills"
        local ccfa_ok=0 ccfa_missing=0 ccfa_support_ok=0
        if [ -d "$ccfa_base" ]; then
            for ccfa_dir in "$ccfa_base"/ccf-*; do
                [ -d "$ccfa_dir" ] || continue
                local entry_name
                entry_name="$(basename "$ccfa_dir")"
                if [ -f "$ccfa_dir/SKILL.md" ]; then
                    if [ -e "$SKILL_DIR/$entry_name/SKILL.md" ]; then
                        ccfa_ok=$((ccfa_ok + 1))
                    else
                        ccfa_missing=$((ccfa_missing + 1))
                    fi
                elif [ -e "$SKILL_DIR/$entry_name" ]; then
                    ccfa_support_ok=$((ccfa_support_ok + 1))
                else
                    ccfa_missing=$((ccfa_missing + 1))
                fi
            done
        else
            ccfa_missing=$((ccfa_missing + 1))
        fi
        if [ $ccfa_missing -eq 0 ] && [ $ccfa_ok -gt 0 ]; then
            echo "  OK: CCFA Skills $ccfa_ok 个、支撑目录 $ccfa_support_ok 个已安装"
            PASS=$((PASS + 1))
        else
            echo "  缺失: CCFA Skills $ccfa_missing 项（已安装 runtime skill $ccfa_ok 个）"
            FAIL=$((FAIL + 1))
        fi
    fi

    echo "  验证: $PASS 通过, $FAIL 失败"
    echo ""

    if [ $FAIL -gt 0 ]; then
        echo "  提示: 缺失路径可能影响 skill 搜索功能."
        echo "    - CUDA 文档: 运行 'python3 scrape_docs.py all --force'"
        echo "    - 源码 repo: 运行 'bash update-repos.sh'"
        echo ""
    fi
}

verify_agent "$AGENT"

echo "安装完成."
