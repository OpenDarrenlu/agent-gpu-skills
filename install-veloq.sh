#!/bin/bash
# 安装/更新 VeloQ —— agent-friendly 的 GPU profile 查询 CLI（nsys/ncu → JSON）
# 及其两个 profiling skill（nsys-profile-analysis / ncu-profile-analysis）。
#
# VeloQ 不随本仓库 vendored；本脚本是最小封装，按以下优先级取得 VeloQ：
#   1. 本地 VeloQ 源码 checkout（$VELOQ_SRC，或自动探测 ../VeloQ、~/workspace/VeloQ）
#   2. 已在 PATH 的 veloq（用其自带 `veloq self-update`）
#   3. VeloQ 官方安装脚本（curl）
#   4. Rust fallback: cargo-binstall / cargo install
# 任一步失败都不致命，只 warn —— 不影响本仓库其他 skill 的安装。
#
# 用法:
#   bash install-veloq.sh [--agent cursor|claude|codex|gemini] [--no-binary] [--no-skills]
#
# 选项:
#   --agent       目标工具（决定 skill 安装目录），默认 cursor
#   --no-binary   只装 skill，不装/更新 veloq 二进制
#   --no-skills   只装/更新二进制，不装 skill
#
# 环境变量:
#   VELOQ_SRC     本地 VeloQ 源码 checkout 路径（优先用它的 .claude/skills 与 target/release/veloq）

set -euo pipefail

AGENT="cursor"
DO_BINARY=true
DO_SKILLS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)     AGENT="$2"; shift 2 ;;
        --no-binary) DO_BINARY=false; shift ;;
        --no-skills) DO_SKILLS=false; shift ;;
        -h|--help)   sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

agent_root() {
    case "$1" in
        cursor) echo "${HOME}/.cursor" ;;
        claude) echo "${HOME}/.claude" ;;
        codex)  echo "${HOME}/.codex" ;;
        gemini) echo "${HOME}/.gemini" ;;
        *) echo "未知 agent: $1" >&2; return 1 ;;
    esac
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(agent_root "$AGENT")"
SKILLS_DIR="$ROOT/skills"
VELOQ_SKILLS=(nsys-profile-analysis ncu-profile-analysis)
INSTALLER_URL="https://raw.githubusercontent.com/lucifer1004/veloq/main/scripts/install.sh"

# 探测本地 VeloQ 源码（带 .claude/skills/ 的 checkout）
detect_src() {
    if [ -n "${VELOQ_SRC:-}" ] && [ -d "$VELOQ_SRC/.claude/skills" ]; then
        ( cd "$VELOQ_SRC" && pwd ); return 0
    fi
    local c
    for c in "$SCRIPT_DIR/../VeloQ" "$SCRIPT_DIR/../veloq" "$HOME/workspace/VeloQ"; do
        if [ -d "$c/.claude/skills" ]; then ( cd "$c" && pwd ); return 0; fi
    done
    return 1
}
SRC="$(detect_src || true)"

# 从本地源码安装 skill：复制 SKILL.md（多数工具不识别软链接的 SKILL.md），
# references 等子目录软链接回源码以跟随更新 —— 与本仓库 install.sh 的混合模式一致。
install_skills_from_src() {
    local src="$1" s from to item b
    mkdir -p "$SKILLS_DIR"
    for s in "${VELOQ_SKILLS[@]}"; do
        from="$src/.claude/skills/$s"
        to="$SKILLS_DIR/$s"
        [ -d "$from" ] || continue
        if [ -L "$to" ]; then rm "$to"; elif [ -d "$to" ]; then rm -rf "$to"; fi
        mkdir -p "$to"
        cp "$from/SKILL.md" "$to/SKILL.md"
        for item in "$from"/*; do
            b="$(basename "$item")"
            [ "$b" = "SKILL.md" ] && continue
            ln -sf "$item" "$to/$b"
        done
        echo "  已安装 skill: $s  (SKILL.md 复制；references 软链接 → $from)"
    done
}

install_binary_from_cargo() {
    if command -v cargo-binstall >/dev/null 2>&1; then
        echo "  尝试 cargo-binstall veloq..."
        cargo-binstall -y veloq && return 0
    fi

    if command -v cargo >/dev/null 2>&1; then
        if cargo binstall --version >/dev/null 2>&1; then
            echo "  尝试 cargo binstall veloq..."
            cargo binstall -y veloq && return 0
        fi

        echo "  尝试 cargo install veloq..."
        cargo install veloq && return 0
    fi

    return 1
}

# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------
if $DO_SKILLS; then
    echo "=== VeloQ skills → $SKILLS_DIR ==="
    if [ -n "$SRC" ]; then
        echo "源: 本地 VeloQ 源码 $SRC"
        install_skills_from_src "$SRC"
    elif command -v veloq >/dev/null 2>&1; then
        echo "源: veloq self-update --no-binary --skills-dir $ROOT"
        if ! veloq self-update --no-binary --skills-dir "$ROOT" >/dev/null 2>&1; then
            echo "  warn: veloq self-update 失败（多为网络/GitHub 限流），稍后可重试" >&2
        else
            echo "  完成"
        fi
    else
        echo "源: VeloQ 官方安装脚本（curl）"
        if ! curl -fsSL "$INSTALLER_URL" | bash -s -- --no-binary --skills-dir "$ROOT"; then
            echo "  warn: 远程安装失败，请检查网络或克隆 VeloQ 后设 VELOQ_SRC 重试" >&2
        fi
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# binary
# ---------------------------------------------------------------------------
if $DO_BINARY; then
    echo "=== VeloQ binary ==="
    if command -v veloq >/dev/null 2>&1; then
        echo "  已安装: $(command -v veloq)  ($(veloq --version 2>/dev/null | head -1))"
        echo "  （升级用: veloq self-update）"
    elif [ -n "$SRC" ] && [ -x "$SRC/target/release/veloq" ]; then
        mkdir -p "$HOME/.local/bin"
        cp "$SRC/target/release/veloq" "$HOME/.local/bin/veloq"
        echo "  已从本地构建产物安装: $HOME/.local/bin/veloq"
    elif [ -n "$SRC" ] && command -v cargo >/dev/null 2>&1; then
        echo "  从本地源码构建 veloq（cargo build --release -p veloq）..."
        if ( cd "$SRC" && cargo build --release -p veloq ) && [ -x "$SRC/target/release/veloq" ]; then
            mkdir -p "$HOME/.local/bin"
            cp "$SRC/target/release/veloq" "$HOME/.local/bin/veloq"
            echo "  已安装: $HOME/.local/bin/veloq"
        else
            echo "  warn: 本地构建失败，可改用官方安装脚本或 'cargo binstall veloq'" >&2
        fi
    else
        echo "  安装官方预编译二进制（curl install.sh --no-skills）..."
        if ! curl -fsSL "$INSTALLER_URL" | bash -s -- --no-skills; then
            echo "  warn: 远程安装失败，尝试 Rust/cargo fallback..." >&2
            if ! install_binary_from_cargo; then
                echo "  warn: VeloQ 二进制未安装。可稍后手动运行 'cargo binstall veloq'、设置 VELOQ_SRC，或用 --no-veloq 跳过" >&2
            fi
        fi
    fi
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *) command -v veloq >/dev/null 2>&1 || echo "  提示: 确保二进制目录（如 ~/.local/bin 或 ~/.cargo/bin）在 PATH 中" ;;
    esac
    echo ""
fi
