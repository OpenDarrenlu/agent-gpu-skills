#!/bin/bash
# 统一获取/更新所有源码仓库
# 用法: bash update-repos.sh [repo_name]
#
# 不带参数: 更新所有 repo
# 带参数:   只更新指定 repo (triton / cutlass / sglang / nvidia-skills / cursor-skills)
#
# repo 存放在各自 skill 目录的 repos/ 下:
#   triton_skill/repos/triton/
#   cutlass_skill/repos/cutlass/
#   sglang_skill/repos/sglang/
#   repos/nvidia-skills/  (NVIDIA skills 完整仓库)
#   repos/cursor-skills/  (Saddss/cursor-skills submodule)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

clone_or_update() {
    local name="$1"
    local skill_dir="$2"
    local url="$3"
    local branch="$4"
    shift 4
    local sparse_dirs=("$@")

    local repos_dir="$SCRIPT_DIR/$skill_dir/repos"
    local repo_dir="$repos_dir/$name"

    mkdir -p "$repos_dir"

    echo ""
    echo "=== $name ==="

    if [ -d "$repo_dir/.git" ]; then
        echo "  更新中..."
        cd "$repo_dir"
        git pull --ff-only origin "$branch" 2>/dev/null || git pull origin "$branch"
        echo "  更新完成."
    else
        echo "  首次 clone (sparse checkout)..."
        git clone --filter=blob:none --no-checkout --depth 1 --branch "$branch" "$url" "$repo_dir"
        cd "$repo_dir"
        git sparse-checkout init --cone
        git sparse-checkout set "${sparse_dirs[@]}"
        git checkout "$branch"
        echo "  Clone 完成."
    fi

    du -sh "$repo_dir" 2>/dev/null | awk '{print "  大小: "$1}'
}

# 完整 clone（不 sparse checkout）用于纯文本 skill 仓库
clone_full_repo() {
    local name="$1"
    local repo_dir="$2"
    local url="$3"
    local branch="$4"

    mkdir -p "$(dirname "$repo_dir")"

    echo ""
    echo "=== $name ==="

    if [ -d "$repo_dir/.git" ]; then
        echo "  更新中..."
        cd "$repo_dir"
        git pull --ff-only origin "$branch" 2>/dev/null || git pull origin "$branch"
        echo "  更新完成."
    else
        echo "  首次 clone..."
        git clone --depth 1 --branch "$branch" "$url" "$repo_dir"
        echo "  Clone 完成."
    fi

    du -sh "$repo_dir" 2>/dev/null | awk '{print "  大小: "$1}'
}

# Git submodule 仓库
update_submodule() {
    local name="$1"
    local path="$2"

    echo ""
    echo "=== $name ==="
    if ! git -C "$SCRIPT_DIR" submodule update --init --remote "$path"; then
        local url
        local https_url
        url="$(git -C "$SCRIPT_DIR" config -f .gitmodules --get "submodule.$path.url" 2>/dev/null || true)"
        case "$url" in
            git@github.com:*)
                https_url="https://github.com/${url#git@github.com:}"
                echo "  SSH 更新失败，尝试 HTTPS: $https_url"
                git -C "$SCRIPT_DIR" config "submodule.$path.url" "$https_url"
                git -C "$SCRIPT_DIR" submodule update --init --remote "$path"
                ;;
            *)
                return 1
                ;;
        esac
    fi
    echo "  Submodule 更新完成."
    du -sh "$SCRIPT_DIR/$path" 2>/dev/null | awk '{print "  大小: "$1}'
}

# Triton sparse checkout 目录
triton_dirs=(
    "python/tutorials"
    "python/triton_kernels"
    "python/triton/language"
    "python/triton/experimental/gluon"
    "python/triton/runtime"
    "python/triton/compiler"
    "python/triton/tools"
    "python/examples"
    "include"
    "lib"
)

# CUTLASS sparse checkout 目录
cutlass_dirs=(
    "python/CuTeDSL"
    "python/pycute"
    "python/cutlass_library"
    "examples"
    "include"
    "tools/library"
    "tools/util"
)

# SGLang sparse checkout 目录
sglang_dirs=(
    "python/sglang/srt"
    "python/sglang/jit_kernel"
    "python/sglang/lang"
    "sgl-kernel/csrc"
    "sgl-kernel/include"
    "sgl-kernel/python"
    "sgl-kernel/tests"
    "sgl-kernel/benchmark"
    "examples"
    "benchmark"
    "docs"
    "test"
)

# VeloQ：装/更新 profile 查询 CLI 二进制（skill 由 install.sh 负责）。非致命。
update_veloq() {
    echo ""
    echo "=== veloq ==="
    if command -v veloq >/dev/null 2>&1; then
        echo "  已安装: $(veloq --version 2>/dev/null | head -1)，尝试 self-update..."
        if veloq self-update --no-skills >/dev/null 2>&1; then
            echo "  完成: $(veloq --version 2>/dev/null | head -1)"
        else
            echo "  跳过（网络/限流；二进制保持不变）"
        fi
    else
        bash "$SCRIPT_DIR/install-veloq.sh" --no-skills || echo "  跳过（见 install-veloq.sh）"
    fi
}

TARGET="${1:-all}"

case "$TARGET" in
    triton)
        clone_or_update "triton" "triton_skill" "https://github.com/triton-lang/triton.git" "main" "${triton_dirs[@]}"
        ;;
    cutlass)
        clone_or_update "cutlass" "cutlass_skill" "https://github.com/NVIDIA/cutlass.git" "main" "${cutlass_dirs[@]}"
        ;;
    sglang)
        clone_or_update "sglang" "sglang_skill" "https://github.com/sgl-project/sglang.git" "main" "${sglang_dirs[@]}"
        ;;
    veloq)
        update_veloq
        ;;
    nvidia-skills)
        clone_full_repo "nvidia-skills" "$SCRIPT_DIR/repos/nvidia-skills" "https://github.com/NVIDIA/skills.git" "main"
        ;;
    cursor-skills)
        update_submodule "cursor-skills" "repos/cursor-skills"
        ;;
    all)
        clone_or_update "triton" "triton_skill" "https://github.com/triton-lang/triton.git" "main" "${triton_dirs[@]}"
        clone_or_update "cutlass" "cutlass_skill" "https://github.com/NVIDIA/cutlass.git" "main" "${cutlass_dirs[@]}"
        clone_or_update "sglang" "sglang_skill" "https://github.com/sgl-project/sglang.git" "main" "${sglang_dirs[@]}"
        update_veloq
        clone_full_repo "nvidia-skills" "$SCRIPT_DIR/repos/nvidia-skills" "https://github.com/NVIDIA/skills.git" "main"
        update_submodule "cursor-skills" "repos/cursor-skills"
        ;;
    *)
        echo "未知 repo: $TARGET"
        echo "用法: bash update-repos.sh [triton|cutlass|sglang|veloq|nvidia-skills|cursor-skills|all]"
        exit 1
        ;;
esac

echo ""
echo "=== 总览 ==="
for sk in triton_skill cutlass_skill sglang_skill; do
    if [ -d "$SCRIPT_DIR/$sk/repos" ]; then
        du -sh "$SCRIPT_DIR/$sk/repos/"*/ 2>/dev/null
    fi
done
if [ -d "$SCRIPT_DIR/repos/nvidia-skills" ]; then
    du -sh "$SCRIPT_DIR/repos/nvidia-skills" 2>/dev/null
    # 统计 NVIDIA skill 数量
    nvidia_count=0
    for base in "$SCRIPT_DIR/repos/nvidia-skills/skills" "$SCRIPT_DIR/repos/nvidia-skills/plugins/nvidia-skills/skills"; do
        if [ -d "$base" ]; then
            for d in "$base"/*/; do
                [ -f "$d/SKILL.md" ] && nvidia_count=$((nvidia_count + 1))
            done
        fi
    done
    echo "  NVIDIA skills 数量: $nvidia_count"
fi
if [ -d "$SCRIPT_DIR/repos/cursor-skills" ]; then
    du -sh "$SCRIPT_DIR/repos/cursor-skills" 2>/dev/null
    cursor_count=0
    if [ -d "$SCRIPT_DIR/repos/cursor-skills/skills" ]; then
        for d in "$SCRIPT_DIR/repos/cursor-skills/skills"/*/; do
            [ -f "$d/SKILL.md" ] && cursor_count=$((cursor_count + 1))
        done
    fi
    echo "  Cursor skills 数量: $cursor_count"
fi
