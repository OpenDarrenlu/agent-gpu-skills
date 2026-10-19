#!/bin/bash
# 从 GitHub 获取/更新 verl 源码（sparse checkout，只拉需要的目录）
# 用法: bash update-verl.sh [--full] [--with-recipe]
#
# 默认使用 sparse checkout，目标目录与 update-repos.sh 一致: repos/verl/
# --full         拉取完整仓库（depth=1）
# --with-recipe  额外初始化 recipe/ submodule（verl-project/verl-recipe）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repos/verl"
REPO_URL="https://github.com/verl-project/verl.git"
BRANCH="main"

FULL_MODE=false
WITH_RECIPE=false
for arg in "$@"; do
    case "$arg" in
        --full) FULL_MODE=true ;;
        --with-recipe) WITH_RECIPE=true ;;
        *) echo "未知参数: $arg"; exit 1 ;;
    esac
done

# sparse checkout 的目录列表（与 update-repos.sh 的 verl_dirs 保持一致）
# 注: recipe/ 是上游 submodule，cone 模式只保留占位目录，需 --with-recipe 才会拉取内容。
SPARSE_DIRS=(
    "verl"
    "examples"
    "recipe"
    "docs"
    "scripts"
    "tests"
)

mkdir -p "$SCRIPT_DIR/repos"

if [ -d "$REPO_DIR/.git" ]; then
    echo "更新 verl 源码..."
    cd "$REPO_DIR"
    git pull --ff-only origin "$BRANCH" 2>/dev/null || git pull origin "$BRANCH"
    if git sparse-checkout list >/dev/null 2>&1; then
        git sparse-checkout init --cone >/dev/null 2>&1 || true
        git sparse-checkout set "${SPARSE_DIRS[@]}"
    fi
    echo "更新完成."
else
    echo "首次 clone verl 源码..."

    if [ "$FULL_MODE" = true ]; then
        echo "模式: 完整 clone（depth=1）"
        git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    else
        echo "模式: sparse checkout（只拉关键目录）"
        git clone --filter=blob:none --no-checkout --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
        cd "$REPO_DIR"
        git sparse-checkout init --cone
        git sparse-checkout set "${SPARSE_DIRS[@]}"
        git checkout "$BRANCH"
    fi

    echo "Clone 完成."
fi

if [ "$WITH_RECIPE" = true ]; then
    echo "初始化 recipe submodule..."
    git -C "$REPO_DIR" submodule update --init --depth 1 recipe
fi

# 验证
echo ""
echo "--- 验证 ---"
PASS=0
FAIL=0

check() {
    if [ -e "$1" ]; then
        echo "  OK: $2"
        PASS=$((PASS + 1))
    else
        echo "  缺失: $2"
        FAIL=$((FAIL + 1))
    fi
}

check "$REPO_DIR/verl/trainer/main_ppo.py" "PPO 入口 main_ppo.py"
check "$REPO_DIR/verl/trainer/ppo/core_algos.py" "算法注册表 core_algos.py"
check "$REPO_DIR/verl/trainer/ppo/v1" "V1 trainer"
check "$REPO_DIR/verl/trainer/config/ppo_trainer.yaml" "Hydra 主配置"
check "$REPO_DIR/verl/single_controller/ray/base.py" "single controller (Ray)"
check "$REPO_DIR/verl/workers/engine_workers.py" "engine workers"
check "$REPO_DIR/verl/workers/engine" "model engines"
check "$REPO_DIR/verl/workers/rollout" "rollout backends"
check "$REPO_DIR/verl/checkpoint_engine" "checkpoint engine"
check "$REPO_DIR/verl/experimental/agent_loop" "agent loop"
check "$REPO_DIR/examples" "Examples"
check "$REPO_DIR/docs" "Documentation"

echo ""
echo "验证: $PASS 通过, $FAIL 失败"

if git -C "$REPO_DIR" rev-parse --short HEAD >/dev/null 2>&1; then
    echo "HEAD: $(git -C "$REPO_DIR" log -1 --format='%h %ci')"
fi
du -sh "$REPO_DIR" 2>/dev/null | awk '{print "仓库大小: "$1}'
