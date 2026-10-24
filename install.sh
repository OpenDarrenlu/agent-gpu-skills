#!/bin/bash
# Agent Skill 安装脚本
# 用法: bash install.sh [--agent cursor|claude|codex|gemini|kimi] [--dest DIR] [--copy] [--no-veloq] [--no-tilelang-skills] [--no-nvidia-skills] [--no-amd-skills] [--no-cursor-skills] [--no-ccfa-skills] [--no-integrations]
#
# 默认安装到 Cursor。使用 --agent 选择目标工具。
#
# 安装模式（默认混合模式）:
#   - skill 目录: 真实目录（多数工具不识别软链接目录）
#   - SKILL.md: 复制真实文件
#   - repos、references 等子目录/文件: 软链接到项目目录
#
# --copy  全量复制模式（适用于无法软链接的场景）
#
# --dest DIR 是隔离安装：只写入 DIR 下的 skill 目录，跳过 VeloQ 等会写入真实 HOME 的步骤，
#            外部来源缺失只给出 WARN。真实安装（不带 --dest）中任何已请求的组件缺失都返回非零。

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
INSTALL_TILELANG=true
INSTALL_NVIDIA=true
INSTALL_AMD=true
INSTALL_CURSOR_SKILLS=true
INSTALL_CCFA_SKILLS=true
INSTALL_INTEGRATIONS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --agent)    AGENT="$2"; shift 2 ;;
        --dest)     DEST_DIR="$2"; shift 2 ;;
        --copy)     COPY_MODE=true; shift ;;
        --no-veloq) INSTALL_VELOQ=false; shift ;;
        --no-tilelang-skills) INSTALL_TILELANG=false; shift ;;
        --no-nvidia-skills) INSTALL_NVIDIA=false; shift ;;
        --no-amd-skills) INSTALL_AMD=false; shift ;;
        --no-cursor-skills) INSTALL_CURSOR_SKILLS=false; shift ;;
        --no-ccfa-skills) INSTALL_CCFA_SKILLS=false; shift ;;
        --no-integrations) INSTALL_INTEGRATIONS=false; shift ;;
        -h|--help)
            echo "用法: bash install.sh [--agent cursor|claude|codex|gemini|kimi] [--dest DIR] [--copy] [--no-veloq] [--no-tilelang-skills] [--no-nvidia-skills] [--no-amd-skills] [--no-cursor-skills] [--no-ccfa-skills] [--no-integrations]"
            echo ""
            echo "首次安装（一条命令，推荐）:"
            echo "  bash bootstrap.sh --agent codex  # 获取 submodule/外部来源、安装运行时 CLI，再安装 skills"
            echo ""
            echo "只安装 skill（不获取源码、不安装运行时）:"
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
            echo "  --dest DIR        隔离安装：只写 DIR 下的 skill 目录；跳过 VeloQ 与运行时 CLI，不写入真实 HOME"
            echo "  --copy            全量复制（适用于无法软链接的场景）"
            echo "  --no-veloq        跳过 VeloQ（profile 查询 CLI + nsys/ncu-profile-analysis skill）"
            echo "  --no-tilelang-skills 跳过 sablin39/tilelang-cuda-skills 中的 TileLang skills"
            echo "  --no-nvidia-skills 跳过 NVIDIA 官方 skills（200+ 个，可能较多）"
            echo "  --no-amd-skills   跳过 AMD 官方 skills（来自 amd/skills）"
            echo "  --no-cursor-skills 跳过 Saddss/cursor-skills 仓库中的 skills"
            echo "  --no-ccfa-skills  跳过 mikubaka88/CCFA-Skills 论文研究 skills"
            echo "  --no-integrations 跳过 sources/external-sources.conf 中的外部来源（如 mattpocock-skills）"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

DEST_MODE=false
if [ -n "$DEST_DIR" ]; then
    DEST_MODE=true
fi

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

if [ ! -d "$SCRIPT_DIR/skills/kernel/cuda-skill" ]; then
    echo "错误: 未找到 skills/kernel/cuda-skill/ 目录"
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
    verl-skill
    train-inference-consistency
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
    gpu-framework-router
    llm-serving-router
    gpu-development-catchall
    agent-skill-router
    iket-profiling
    tilelang-optimization-loop
    designing-with-tilefoundry
    debugging-tilelang-programs
    optimizing-tilelang-programs
    profiling-tilelang-programs
    testing-fwd-bwd-kernels
    torch-profiling-tilelang-programs
    writing-tilelang-design-docs
    writing-tilelang-kernels
    giantpandallm-knowledge
    zhihu-knowledge-base
    salary-negotiation-skill
    tech-doc-style-chinese
    zotero-cli
    ponytail
    ponytail-review
    ponytail-audit
    ponytail-debt
    ponytail-gain
    ponytail-help
    claude-delegate
    codex-background
)
SKILL_DIRS=(
    skills/kernel/cuda-skill
    skills/kernel/triton-skill
    skills/kernel/cutlass-skill
    skills/kernel/deepgemm-skill
    skills/framework/sglang-skill
    skills/framework/verl-skill
    skills/framework/train-inference-consistency
    skills/kernel/nv-gpu-kernel-performance-modeling
    skills/kernel/colfax-research-skill
    skills/other/nccl-skill
    skills/other/gpu-communication-libraries
    skills/kernel/ncu-persistent-kernel-diagnosis
    skills/kernel/persistent-kernel-scheduling
    skills/kernel/persistent-kernel-utilization
    skills/kernel/ncu-report-skill
    skills/other/amd-gpu-docs
    skills/other/amd-instinct-isa
    skills/other/amd-instinct-cdna4-isa
    skills/routers/gpu-performance-router
    skills/routers/gpu-kernel-authoring-router
    skills/routers/gpu-framework-router
    skills/routers/llm-serving-router
    skills/routers/gpu-development-catchall
    skills/routers/agent-skill-router
    skills/kernel/iket-profiling
    repos/tilelang-cuda-skills/skills/tilelang-optimization-loop
    repos/tilelang-cuda-skills/skills/designing-with-tilefoundry
    repos/tilelang-cuda-skills/skills/tilelang/debugging-tilelang-programs
    repos/tilelang-cuda-skills/skills/tilelang/optimizing-tilelang-programs
    repos/tilelang-cuda-skills/skills/tilelang/profiling-tilelang-programs
    repos/tilelang-cuda-skills/skills/tilelang/testing-fwd-bwd-kernels
    repos/tilelang-cuda-skills/skills/tilelang/torch-profiling-tilelang-programs
    repos/tilelang-cuda-skills/skills/tilelang/writing-tilelang-design-docs
    repos/tilelang-cuda-skills/skills/tilelang/writing-tilelang-kernels
    skills/other/giantpandallm-knowledge
    skills/other/zhihu-knowledge-base
    skills/other/salary-negotiation-skill
    skills/other/tech-doc-style-chinese
    skills/other/zotero-cli
    repos/ponytail/skills/ponytail
    repos/ponytail/skills/ponytail-review
    repos/ponytail/skills/ponytail-audit
    repos/ponytail/skills/ponytail-debt
    repos/ponytail/skills/ponytail-gain
    repos/ponytail/skills/ponytail-help
    skills/other/claude-delegate
    skills/other/codex-background
)

# 检查是否为本地 skill（避免外部 catalog 覆盖本地路由/参考 skill）
is_local_skill() {
    local name="$1"
    for n in "${SKILL_NAMES[@]}"; do
        [ "$n" = "$name" ] && return 0
    done
    return 1
}

# 用户级保留名：由本机其他项目管理的软链接（research_ltz），任何 catalog 都不得覆盖或删除。
PROTECTED_SKILL_NAMES="claude-delegate codex-background"

is_protected_name() {
    local p
    for p in $PROTECTED_SKILL_NAMES; do
        [ "$p" = "$1" ] && return 0
    done
    return 1
}

# 本次安装已占用的 skill 名 -> 来源。先到先得：本地 skill 最先登记，其余按安装顺序登记。
CLAIMED_NAMES=()
CLAIMED_SOURCES=()

claimed_source() {
    local i
    for i in "${!CLAIMED_NAMES[@]}"; do
        if [ "${CLAIMED_NAMES[$i]}" = "$1" ]; then
            echo "${CLAIMED_SOURCES[$i]}"
            return 0
        fi
    done
    return 1
}

claim_skill() {
    CLAIMED_NAMES+=("$1")
    CLAIMED_SOURCES+=("$2")
}

# 指向本仓库的顶层 symlink 才是本安装器创建的；其他 symlink 视为用户配置。
is_managed_link() {
    case "$(readlink "$1")" in
        "$SCRIPT_DIR"/*) return 0 ;;
    esac
    return 1
}

# 替换 $SKILL_DIR/<name> 之前的检查。返回 1 表示必须保留现有条目。
can_replace_target() {
    local name="$1" target="$2"
    if is_protected_name "$name"; then
        echo "  保留: $name 为用户级 skill，不由安装器覆盖"
        return 1
    fi
    if [ -L "$target" ] && ! is_managed_link "$target"; then
        echo "  保留: $target 是非本仓库 symlink，未覆盖"
        return 1
    fi
    return 0
}

# 受保护名（用户级 skill）的本地安装策略。参数: 名称  仓库内 bundled 源目录  目标 $SKILL_DIR/<name>
#   - 目标已存在（任何 symlink 或目录）：原样保留，不替换，不改动任何用户级 symlink。
#   - 目标不存在且为真实安装：若 $HOME/.agents/skills/<name>/SKILL.md 存在，只创建指向它的别名 symlink。
#   - 目标不存在且为 --dest 隔离安装：不读取真实 HOME，直接复制仓库内的 bundled 版本。
#   - 目标不存在且没有用户级副本：复制仓库内的 bundled 版本（新机器的便携安装）。
# 名称无论哪种情况都会被本地登记，因此 catalog 永远不能占用这些名称。
install_protected_local() {
    local name="$1" src="$2" target="$3"
    local user_copy="$HOME/.agents/skills/$name" item basename_item

    claim_skill "$name" "local"

    if [ -e "$target" ] && [ "$target" -ef "$src" ]; then
        echo "  拒绝: 目标与源相同，避免自链接"
        return 0
    fi
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo "  保留: $target 已存在（用户级 skill，未替换）"
        return 0
    fi
    if [ "$DEST_MODE" = false ] && [ -f "$user_copy/SKILL.md" ]; then
        ln -s "$user_copy" "$target"
        echo "  别名: ${target} -> ${user_copy}（复用已有用户级版本，未改动原链接）"
        return 0
    fi
    if [ ! -f "$src/SKILL.md" ]; then
        echo "  缺失: $src 不存在，且没有可复用的用户级版本"
        return 0
    fi
    if [ "$COPY_MODE" = true ]; then
        mkdir -p "$target"
        for item in "$src"/*; do
            cp -R "$item" "$target/"
        done
    else
        mkdir -p "$target"
        cp "$src/SKILL.md" "$target/SKILL.md"
        for item in "$src"/*; do
            basename_item="$(basename "$item")"
            [ "$basename_item" = "SKILL.md" ] && continue
            ln -sf "$item" "$target/$basename_item" 2>/dev/null || true
        done
    fi
    echo "  已安装仓库内 bundled 版本: $name"
}

# 把一个 catalog 中的 skill 目录放入 $SKILL_DIR/<name>（混合模式或 --copy）。
# 用法: place_catalog_skill <catalog> <源 skill 目录> <安装名>
# 返回 0 = 已安装；1 = 跳过（同名已被先到的来源占用、保留名或用户 symlink）。
place_catalog_skill() {
    local catalog="$1" src="$2" name="$3"
    local target="$SKILL_DIR/$name" item basename_item

    if claimed_source "$name" >/dev/null; then
        return 1
    fi
    if is_protected_name "$name"; then
        echo "  跳过: ${name} 为受保护的用户级名称，catalog 不得占用（${catalog}）"
        return 1
    fi
    can_replace_target "$name" "$target" || return 1

    if [ -L "$target" ]; then
        rm "$target"
    elif [ -d "$target" ]; then
        rm -rf "$target"
    fi

    if [ "$COPY_MODE" = true ]; then
        # 只复制可见的 skill 载荷。隐藏项（.git、.claude、.agents）属于仓库或 agent 配置，不进入用户 skill 目录；
        # 根目录即 skill 的仓库（如 wx-cli）因此不会把 Git 元数据复制过去。
        mkdir -p "$target"
        for item in "$src"/*; do
            cp -R "$item" "$target/"
        done
    else
        mkdir -p "$target"
        cp "$src/SKILL.md" "$target/SKILL.md"
        for item in "$src"/*; do
            basename_item="$(basename "$item")"
            [ "$basename_item" = "SKILL.md" ] && continue
            [ -L "$target/$basename_item" ] && rm "$target/$basename_item"
            ln -sf "$item" "$target/$basename_item" 2>/dev/null || true
        done
    fi
    claim_skill "$name" "$catalog"
    return 0
}

# 运行时命令是否可用：PATH 中的命令（需 --version 成功），或真实安装时由 install-external.sh 生成的
# managed launcher（$HOME/.local/bin/<bin>，带标记）。--dest 隔离安装不读取真实 HOME，只看 PATH。
MANAGED_LAUNCHER_MARKER="agent-gpu-skills-managed"
runtime_usable() {
    local bin="$1" launcher="$HOME/.local/bin/$1"
    if command -v "$bin" >/dev/null 2>&1 && "$bin" --version >/dev/null 2>&1 </dev/null; then
        return 0
    fi
    if [ "$DEST_MODE" = false ] && [ -f "$launcher" ] && [ ! -L "$launcher" ] \
        && grep -q "$MANAGED_LAUNCHER_MARKER" "$launcher" 2>/dev/null \
        && "$launcher" --version >/dev/null 2>&1 </dev/null; then
        return 0
    fi
    return 1
}

# 外部来源的 skill 名：取 SKILL.md frontmatter 的 name，缺省时取目录的真实名称。
# 用于 skills_dir 为 "." 的仓库根 SKILL.md（wx-cli），目录基名此时是 "."。build_inventory.py 使用同一规则。
frontmatter_skill_name() {
    local dir="$1" name
    name="$(sed -n 's/^name:[[:space:]]*//p' "$dir/SKILL.md" | head -1 | tr -d "\"' \r")"
    case "$name" in
        ''|*[!A-Za-z0-9._-]*) basename "$(cd "$dir" && pwd)" ;;
        *) echo "$name" ;;
    esac
}

# sources/external-sources.conf 中 mode=skills 的来源（sepia、wx-cli、rea、mattpocock-skills）。
# skills_dir 可用逗号列出多个目录；递归查找 SKILL.md；同名先到先得。
# 缺失、pending 与运行时检查由 verify_conf_sources 报告。
install_conf_catalogs() {
    local conf="$SCRIPT_DIR/sources/external-sources.conf"
    local name url ref mode skills_dir runtime provenance sd root skill_md skill_dir skill_name
    local installed skipped
    [ -f "$conf" ] || return 0

    while IFS='|' read -r name url ref mode skills_dir runtime provenance; do
        case "$name" in ''|'#'*) continue ;; esac
        [ "$mode" = none ] && continue

        echo "--- 外部来源: $name ---"
        if [ "$mode" = pending ]; then
            echo "  跳过: $name 尚未检视（mode=pending），不能安装；见 sources/external-sources.conf"
            continue
        fi

        installed=0
        skipped=0
        for sd in $(echo "$skills_dir" | tr ',' ' '); do
            root="$SCRIPT_DIR/repos/$name/$sd"
            if [ ! -d "$root" ]; then
                echo "  跳过: $root 不存在（运行 bash install-external.sh 获取）"
                continue
            fi
            # -H：源根若是 symlink（如 repos/rea/runtime-skills 指向 npm 全局包）也要进入；不跟随子目录中的 symlink。
            while IFS= read -r skill_md; do
                skill_dir="$(dirname "$skill_md")"
                skill_name="$(frontmatter_skill_name "$skill_dir")"
                if place_catalog_skill "$name" "$skill_dir" "$skill_name"; then
                    installed=$((installed + 1))
                else
                    skipped=$((skipped + 1))
                fi
            done < <(find -H "$root" -name SKILL.md -not -path '*/node_modules/*' | sort)
            if [ -z "$(find -H "$root" -name SKILL.md 2>/dev/null | head -1)" ]; then
                echo "  跳过: $root 中没有 SKILL.md（源为空或链接失效）"
            fi
        done

        echo "  已安装 $installed 个 skill（跳过同名/保留 $skipped 个）"
    done < "$conf"
}

ensure_submodule_path() {
    local label="$1"
    local path="$2"
    local marker="$3"

    [ -e "$SCRIPT_DIR/$marker" ] && return 0

    # --dest 隔离安装不联网：缺失的 submodule 只报告，不获取。
    if [ "$DEST_MODE" = true ]; then
        echo "  跳过 submodule 获取（--dest 不联网）: $label ($path)"
        return 1
    fi

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
    if [ "$DEST_MODE" = true ]; then
        echo "模式: --dest 隔离安装（只写 skill 目录；不运行 VeloQ，不安装运行时 CLI）"
    fi
    echo ""

    mkdir -p "$SKILL_DIR"

    if [ "$INSTALL_TILELANG" = true ]; then
        ensure_submodule_path "tilelang-cuda-skills" "repos/tilelang-cuda-skills" "repos/tilelang-cuda-skills/skills" || true
        if [ -d "$SCRIPT_DIR/repos/tilelang-cuda-skills" ] && [ "$DEST_MODE" = false ]; then
            git -C "$SCRIPT_DIR" submodule update --init --recursive repos/tilelang-cuda-skills >/dev/null 2>&1 || \
                echo "  warn: 无法初始化 tilelang-cuda-skills 的嵌套 submodule" >&2
        fi
    fi

    ensure_submodule_path "ponytail" "repos/ponytail" "repos/ponytail/skills/ponytail/SKILL.md" || true

    for i in "${!SKILL_NAMES[@]}"; do
        skill_name="${SKILL_NAMES[$i]}"
        src_dir="${SKILL_DIRS[$i]}"
        src_path="$SCRIPT_DIR/$src_dir"
        target="$SKILL_DIR/$skill_name"

        if [ "$INSTALL_TILELANG" = false ] && [[ "$src_dir" == repos/tilelang-cuda-skills/* ]]; then
            continue
        fi
        # codex-background 是 claude-delegate 在 Codex 中的 MCP 支撑 skill，其他 agent 不安装。
        if [ "$skill_name" = "codex-background" ] && [ "$agent" != "codex" ]; then
            continue
        fi

        echo "--- $skill_name ---"

        # 清理旧安装
        if [ "$skill_name" = "triton-skill" ]; then
            old_target="$SKILL_DIR/triton-gluon-skill"
            if [ -L "$old_target" ] || [ -d "$old_target" ]; then
                echo "  移除旧版: triton-gluon-skill"
                rm -rf "$old_target"
            fi
        fi

        if is_protected_name "$skill_name"; then
            install_protected_local "$skill_name" "$src_path" "$target"
            continue
        fi

        can_replace_target "$skill_name" "$target" || continue

        if [ -L "$target" ]; then
            rm "$target"
        elif [ -d "$target" ]; then
            rm -rf "$target"
        fi

        if [ ! -d "$src_path" ]; then
            echo "  跳过: $src_dir/ 不存在"
            continue
        fi
        claim_skill "$skill_name" "local"

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

    # Kernel Design Agents 的 KernelWiki 是外部 skill，但属于本仓库的 kernel
    # 路由面；本地 ncu-report-skill 仍然优先，避免重复覆盖。
    local kernelwiki_src="$SCRIPT_DIR/kernel-design-agents/skills/KernelWiki"
    local kernelwiki_target="$SKILL_DIR/KernelWiki"
    if [ ! -f "$kernelwiki_src/SKILL.md" ] && [ "$DEST_MODE" = false ]; then
        echo "--- KernelWiki ---"
        echo "  初始化 kernel-design-agents 及其嵌套 skills..."
        git -C "$SCRIPT_DIR" submodule update --init --recursive kernel-design-agents || true
    fi
    if [ -f "$kernelwiki_src/SKILL.md" ]; then
        echo "--- KernelWiki ---"
        if place_catalog_skill "kernel-design-agents" "$kernelwiki_src" "KernelWiki"; then
            echo "  已安装 KernelWiki"
        else
            echo "  跳过 KernelWiki（名称已被占用或受保护）"
        fi
    else
        echo "  可选缺失: KernelWiki（运行 git submodule update --init --recursive kernel-design-agents）"
    fi

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

                # 同名先到先得：本地 skill 最先登记；同一 catalog 内的重复目录（如 nvidia-skill-finder）只装第一个
                if place_catalog_skill "nvidia-skills" "$skill_dir" "$skill_name"; then
                    nvidia_installed=$((nvidia_installed + 1))
                else
                    nvidia_skipped=$((nvidia_skipped + 1))
                fi
            done
        done

        if [ $nvidia_installed -gt 0 ]; then
            echo "--- NVIDIA skills ($nvidia_installed 个) ---"
            echo "  已安装 $nvidia_installed 个 NVIDIA 官方 skill"
            echo "  (跳过同名或受保护的 $nvidia_skipped 个)"
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

                # 同名先到先得：本地 skill 优先，cursor-skills 先于 AMD/外部来源登记
                if place_catalog_skill "cursor-skills" "$skill_dir" "$skill_name"; then
                    cursor_installed=$((cursor_installed + 1))
                else
                    cursor_skipped=$((cursor_skipped + 1))
                fi
            done

            if [ $cursor_installed -gt 0 ]; then
                echo "--- Cursor skills ($cursor_installed 个) ---"
                echo "  已安装 $cursor_installed 个 Saddss/cursor-skills skill"
                echo "  (跳过同名或受保护的 $cursor_skipped 个)"
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

                if [ -f "$ccfa_dir/SKILL.md" ]; then
                    if place_catalog_skill "ccfa-skills" "$ccfa_dir" "$entry_name"; then
                        ccfa_installed=$((ccfa_installed + 1))
                    else
                        ccfa_skipped=$((ccfa_skipped + 1))
                    fi
                    continue
                fi

                # 支撑目录不参与 skill 发现，混合模式可直接链接整个目录。
                can_replace_target "$entry_name" "$target" || continue
                if [ -L "$target" ]; then
                    rm "$target"
                elif [ -d "$target" ]; then
                    rm -rf "$target"
                elif [ -e "$target" ]; then
                    rm "$target"
                fi
                if [ "$COPY_MODE" = true ]; then
                    cp -r "$ccfa_dir" "$target"
                else
                    ln -s "$ccfa_dir" "$target"
                fi
                ccfa_support_installed=$((ccfa_support_installed + 1))
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

                # 本地与先安装的 catalog 优先，避免上游新增同名目录覆盖仓库路由或 cursor 改写版。
                if place_catalog_skill "amd-skills" "$skill_dir" "$skill_name"; then
                    amd_installed=$((amd_installed + 1))
                else
                    amd_skipped=$((amd_skipped + 1))
                fi
            done

            echo "--- AMD skills ($amd_installed 个) ---"
            echo "  已安装 $amd_installed 个 amd/skills 官方 skill"
            echo "  (跳过同名或受保护的 $amd_skipped 个)"
            echo ""
        else
            echo "--- AMD skills ---"
            echo "  跳过: 未找到 repos/amd-skills/skills"
            echo "  提示: 运行 'git submodule update --init repos/amd-skills' 或 'bash update-repos.sh amd-skills'"
            echo ""
        fi
    fi

    # sources/external-sources.conf 中的外部 skill 来源（最后安装：同名时本地与上面的 catalog 优先）
    if [ "$INSTALL_INTEGRATIONS" = true ]; then
        echo "--- 外部来源（sources/external-sources.conf）---"
        install_conf_catalogs
        echo ""
    fi
}

install_to_agent "$AGENT"

# VeloQ：profile 查询 CLI + 两个 profiling skill。不 vendored，委托其官方安装器。
# 非致命：失败只 warn，不影响上面已装好的 skill。
# --dest 隔离安装时跳过：install-veloq.sh 写入真实 HOME（~/.local/bin 与 agent skills），不随 --dest 重定向。
if [ "$INSTALL_VELOQ" = true ] && [ "$DEST_MODE" = true ]; then
    echo "提示: --dest 隔离安装已跳过 VeloQ（会写入真实 HOME；需要时去掉 --dest 或加 --no-veloq）"
    echo ""
elif [ "$INSTALL_VELOQ" = true ] && [ -f "$SCRIPT_DIR/install-veloq.sh" ]; then
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

# 验证 sources/external-sources.conf 中 mode=skills 的来源是否都已可见于 $SKILL_DIR。
# 真实安装中缺失计为 FAIL（已请求的组件缺失）；--dest 隔离安装只提示。
# 依赖调用方 verify_agent 的 PASS/FAIL 局部变量（bash 动态作用域）。
verify_conf_sources() {
    local conf="$SCRIPT_DIR/sources/external-sources.conf"
    local name url ref mode skills_dir runtime provenance root skill_md skill_name sd bin found_here
    local ok=0 missing=0 rt_ok=0 rt_missing=0
    [ -f "$conf" ] || return 0

    while IFS='|' read -r name url ref mode skills_dir runtime provenance; do
        case "$name" in ''|'#'*) continue ;; esac
        if [ "$mode" = pending ] || [ "$runtime" = pending ]; then
            missing=$((missing + 1))
            echo "  缺失: $name 仍是 pending（未检视）"
            continue
        fi
        if [ "$mode" = skills ]; then
            for sd in $(echo "$skills_dir" | tr ',' ' '); do
                root="$SCRIPT_DIR/repos/$name/$sd"
                if [ ! -d "$root" ]; then
                    missing=$((missing + 1))
                    echo "  缺失: $root 不存在（运行 bash install-external.sh）"
                    continue
                fi
                found_here=0
                while IFS= read -r skill_md; do
                    found_here=$((found_here + 1))
                    skill_name="$(frontmatter_skill_name "$(dirname "$skill_md")")"
                    if [ -e "$SKILL_DIR/$skill_name/SKILL.md" ]; then
                        ok=$((ok + 1))
                    else
                        missing=$((missing + 1))
                        echo "  缺失: $SKILL_DIR/$skill_name/SKILL.md"
                    fi
                done < <(find -H "$root" -name SKILL.md -not -path '*/node_modules/*' | sort)
                # 源目录存在但没有任何 SKILL.md 不是成功：这意味着请求的 skill 没有被获取到。
                if [ $found_here -eq 0 ]; then
                    missing=$((missing + 1))
                    echo "  缺失: $root 中没有 SKILL.md（源为空或链接失效）"
                fi
            done
        fi
        # 运行时命令：PATH 中可用，或真实安装时的 managed launcher 可用（只检查，不安装）。
        if [ "$runtime" != none ]; then
            bin="${runtime##*:}"
            if runtime_usable "$bin"; then
                rt_ok=$((rt_ok + 1))
            else
                rt_missing=$((rt_missing + 1))
                echo "  缺失: 运行时命令 '${bin}'（${runtime}）不在 PATH 或 --version 失败"
            fi
        fi
    done < "$conf"

    if [ $((missing + rt_missing)) -eq 0 ]; then
        echo "  OK: 外部来源 skill $ok 个已安装，运行时命令 $rt_ok 个可用"
        PASS=$((PASS + 1))
    elif [ "$DEST_MODE" = true ]; then
        echo "  可选缺失: 外部来源 $missing 项 skill、$rt_missing 项运行时（--dest 隔离安装不计为失败；已安装 $ok 个 skill）"
    else
        echo "  缺失: 外部来源 $missing 项 skill、$rt_missing 项运行时未就绪（已安装 $ok 个 skill）。运行 bash bootstrap.sh"
        FAIL=$((FAIL + 1))
    fi
}

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
        src_dir="${SKILL_DIRS[$i]}"
        if [ "$INSTALL_TILELANG" = false ] && [[ "$src_dir" == repos/tilelang-cuda-skills/* ]]; then
            continue
        fi
        if [ "$skill_name" = "codex-background" ] && [ "$agent" != "codex" ]; then
            continue
        fi
        check "$SKILL_DIR/$skill_name/SKILL.md" "$skill_name/SKILL.md"
    done

    REFS="$SCRIPT_DIR/skills/kernel/cuda-skill/references"
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

    local VERL_REPO="$SKILL_DIR/verl-skill/repos/verl"
    check "$VERL_REPO/verl/trainer/ppo/core_algos.py" "verl PPO/GRPO 算法注册表"
    check "$VERL_REPO/verl/trainer/ppo/v1" "verl V1 trainer"
    check "$VERL_REPO/verl/workers/engine_workers.py" "verl engine workers"
    check "$VERL_REPO/verl/checkpoint_engine" "verl checkpoint engine"

    local TIC_SKILL="$SKILL_DIR/train-inference-consistency"
    check "$TIC_SKILL/workflow/README.md" "训推一致: C0-C10 workflow"
    check "$TIC_SKILL/guides/bi-kernels.md" "训推一致: BI kernel guide"
    local VLLM_REPO="$TIC_SKILL/repos/vllm"
    check "$VLLM_REPO/docs/features/batch_invariance.md" "vLLM batch invariance 文档"
    check "$VLLM_REPO/vllm/model_executor/determinism/batch_invariant.py" "vLLM batch invariant 实现"
    check "$VLLM_REPO/tests/v1/determinism" "vLLM batch invariance 测试"

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

    local ZHIHU_SKILL="$SKILL_DIR/zhihu-knowledge-base"
    check "$ZHIHU_SKILL/index/zhihu_articles.db" "知乎知识库: SQLite 索引"
    check "$ZHIHU_SKILL/articles" "知乎知识库: articles"
    check "$ZHIHU_SKILL/scripts/zhihu_search.py" "知乎知识库: 搜索脚本"

    local TECHDOC_SKILL="$SKILL_DIR/tech-doc-style-chinese"
    check "$TECHDOC_SKILL/references/gpu-report-structures.md" "中文技术写作: 报告结构模板"
    check "$TECHDOC_SKILL/assets/report-template.html" "中文技术写作: HTML 报告模板"
    check "$TECHDOC_SKILL/scripts/lint_copy_rules.py" "中文技术写作: 检查器"

    check "$SKILL_DIR/claude-delegate/scripts/relay.mjs" "Claude 委托: relay 脚本"
    if [ "$agent" = "codex" ]; then
        local CODEX_BG_SKILL="$SKILL_DIR/codex-background"
        check "$CODEX_BG_SKILL/scripts/bootstrap.mjs" "Codex 后台: MCP 注册脚本"
        check "$CODEX_BG_SKILL/scripts/server.mjs" "Codex 后台: MCP server"
    fi

    if [ "$INSTALL_TILELANG" = true ]; then
        local TILELANG_REPO="$SCRIPT_DIR/repos/tilelang-cuda-skills"
        check "$TILELANG_REPO/skills/tilelang-optimization-loop/SKILL.md" "TileLang: optimization loop"
        check "$TILELANG_REPO/skills/tilelang/writing-tilelang-kernels/SKILL.md" "TileLang: kernel authoring"
        check "$TILELANG_REPO/skills/tilelang/profiling-tilelang-programs/SKILL.md" "TileLang: profiling"
        check "$TILELANG_REPO/skills/tilelang/debugging-tilelang-programs/references/compiler-tools.md" "TileLang: compiler inspection"
    fi

    local NCU_SKILL="$SKILL_DIR/ncu-report-skill"
    check "$NCU_SKILL/references/ProfilingGuide.md" "NCU ProfilingGuide (参考文档)"
    if [ -e "$SKILL_DIR/KernelWiki/SKILL.md" ]; then
        echo "  OK: KernelWiki（kernel 知识库）"
        PASS=$((PASS + 1))
    else
        echo "  可选缺失: KernelWiki（运行 git submodule update --init --recursive kernel-design-agents）"
    fi

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

    if [ "$INSTALL_INTEGRATIONS" = true ]; then
        verify_conf_sources
    fi

    echo "  验证: $PASS 通过, $FAIL 失败"
    VERIFY_FAIL=$FAIL
    echo ""

    if [ $FAIL -gt 0 ]; then
        echo "  提示: 缺失路径可能影响 skill 搜索功能."
        echo "    - CUDA 文档: 运行 'python3 scrape_docs.py all --force'"
        echo "    - 源码 repo: 运行 'bash update-repos.sh'"
        echo ""
    fi
}

verify_agent "$AGENT"

# codex-background 的 bootstrap 会写入全局 Codex 配置，安装器只提示、不自动执行。
# 上游 bootstrap 仅支持 Linux/Windows；macOS 上的补丁版配置由用户单独维护，不能被提示覆盖。
if [ "$AGENT" = "codex" ]; then
    echo "提示: 在 Codex 中使用 claude-delegate 需要 codex-background MCP server。"
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "  macOS: 上游 codex-background 的 bootstrap 不支持 macOS。若已有补丁版用户级配置，保持现有 MCP 注册，不要替换。"
    else
        echo "  手动注册（会写入全局 Codex 配置，需自行确认）并重启 Codex:"
        echo "    node \"$(get_skill_dir codex)/codex-background/scripts/bootstrap.mjs\""
    fi
    echo ""
fi

if [ "$VERIFY_FAIL" -gt 0 ]; then
    echo "安装未完成: $VERIFY_FAIL 项已请求的组件缺失（见上方“缺失”）。"
    exit 1
fi

echo "安装完成."
