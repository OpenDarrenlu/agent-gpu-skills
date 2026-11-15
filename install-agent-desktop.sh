#!/bin/bash
# 独立安装 vercel-labs/agent-desktop：上游 skill 与匹配的 npm 运行时（agent-desktop@1.3.1）。
#
# 与统一安装流程分开：install.sh、bootstrap.sh、install-external.sh 不调用本脚本，
# sources/external-sources.conf 也不登记它。只在显式运行时执行。
#
# 固定版本（本脚本是唯一的 pin 来源）:
#   npm 运行时: agent-desktop@1.3.1（npm 官方 registry）
#   skill 源码: vercel-labs/agent-desktop 提交 b0b7277bea09b770ef9d1fb45aad667bc432cd57（tag v1.3.1）
#               skill 与运行时必须是同一发布版本，所以取 release 提交而不是 main
#   许可: Apache-2.0（上游 LICENSE，随 skill 一并复制）
#
# 用法:
#   bash install-agent-desktop.sh [--agent cursor|claude|codex|gemini|kimi] [--dest DIR]
#     --agent  默认 codex（~/.codex/skills）；目录映射与 install.sh 相同
#     --dest   隔离安装：写 DIR/agent-desktop；checkout 缺失时还会创建 repos/agent-desktop；
#              不安装 npm 运行时，不写 HOME，不运行 npm
#
# 真实安装直接写入的路径:
#   <skill 目录>/agent-desktop/                           上游 skill、LICENSE 与标记文件
#   ~/.local/share/agent-gpu-skills/agent-desktop/1.3.1/  npm 私有前缀（--prefix，不使用 npm -g）
#   ~/.local/bin/agent-desktop                            带标记的 launcher（不修改 PATH 或 shell 配置）
#   repos/agent-desktop/                                  上游 checkout；缺失时创建并只获取固定提交
#   npm 还会写入它的下载缓存（~/.npm）
#
# npm 使用 --ignore-scripts：不运行上游 postinstall，bin 经 Node launcher 运行（每次调用约多 125 ms）。
#
# 安全规则:
#   - 目标已存在且不带本脚本标记（或是 symlink）时失败并保持原样，不删除、不覆盖。
#   - 所有前置检查（checkout、上游文件集合、Node/npm、写入目标归属）在写入前完成。
#   - 不请求 macOS 辅助功能或屏幕录制权限，不运行 agent-desktop 的任何子命令。
#
# 退出码: 0 成功；1 任何失败（前置条件缺失、冲突、校验不符或 npm 安装失败）。

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG="agent-desktop"
NPM_VERSION="1.3.1"
NPM_SPEC="${PKG}@${NPM_VERSION}"
UPSTREAM_URL="https://github.com/vercel-labs/agent-desktop.git"
UPSTREAM_SHA="b0b7277bea09b770ef9d1fb45aad667bc432cd57"
SKILL_SRC="skills/agent-desktop"
# 上游 skill 的完整文件集合（相对 ${SKILL_SRC}）。上游目录与此不一致时失败，需要人工更新本脚本。
SKILL_FILES=(
    SKILL.md
    references/commands-interaction.md
    references/commands-observation.md
    references/commands-system.md
    references/workflows.md
)
OWNER_TAG="agent-desktop-standalone-installer"
OWNER_MARKER=".${OWNER_TAG}"
CHECKOUT="$SCRIPT_DIR/repos/agent-desktop"
RUNTIME_DIR="$HOME/.local/share/agent-gpu-skills/${PKG}/${NPM_VERSION}"
LAUNCHER="$HOME/.local/bin/${PKG}"

AGENT="codex"
DEST_DIR=""
STAGE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)   AGENT="$2"; shift 2 ;;
        --dest)    DEST_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

# skill 目录：--dest 时即 DIR；否则与 install.sh 的 get_skill_dir 相同。
skill_root() {
    if [ -n "$DEST_DIR" ]; then
        echo "$DEST_DIR"
        return 0
    fi
    case "$AGENT" in
        cursor) echo "${HOME}/.cursor/skills" ;;
        claude) echo "${HOME}/.claude/skills" ;;
        codex)  echo "${HOME}/.codex/skills" ;;
        gemini) echo "${HOME}/.gemini/skills" ;;
        kimi)   echo "${HOME}/.agents/skills" ;;
        *)      echo "未知 agent: $AGENT" >&2; return 1 ;;
    esac
}

SKILL_ROOT="$(skill_root)" || exit 1
SKILL_TARGET="$SKILL_ROOT/${PKG}"

# shellcheck source=scripts/node-preflight.sh
. "$SCRIPT_DIR/scripts/node-preflight.sh"

# 由本脚本生成的目录：带标记的真实目录。symlink 与无标记目录一律视为用户所有。
is_owned() {
    [ -d "$1" ] && [ ! -L "$1" ] && [ -f "$1/$OWNER_MARKER" ]
}

# 返回 0：路径不存在，或是本脚本生成的目录；否则打印 BLOCKED 并返回 1，不做任何修改。
check_owned_dir() {
    local path="$1"
    if [ ! -e "$path" ] && [ ! -L "$path" ]; then
        return 0
    fi
    if is_owned "$path"; then
        return 0
    fi
    echo "  BLOCKED: ${path} 已存在，且不是本脚本生成的副本（用户目录或 symlink），未修改"
    echo "           请先移走或重命名该路径，然后重新运行"
    return 1
}

# launcher 只有带标记的普通文件才视为本脚本生成。
check_launcher() {
    if [ ! -e "$LAUNCHER" ] && [ ! -L "$LAUNCHER" ]; then
        return 0
    fi
    if [ ! -L "$LAUNCHER" ] && [ -f "$LAUNCHER" ] && grep -q "$OWNER_TAG" "$LAUNCHER" 2>/dev/null; then
        return 0
    fi
    echo "  BLOCKED: ${LAUNCHER} 已存在，且不是本脚本生成的 launcher，未修改"
    echo "           请先移走或重命名该文件，然后重新运行（本脚本不覆盖无关的 agent-desktop 命令）"
    return 1
}

# 上游 checkout：缺失时创建并只获取固定提交；已存在时必须是 origin 一致、无未提交改动的独立 git checkout。
prepare_checkout() {
    local origin
    if [ ! -e "$CHECKOUT" ]; then
        echo "  创建 checkout: ${CHECKOUT}（origin ${UPSTREAM_URL}）"
        mkdir -p "$CHECKOUT" || return 1
        git -C "$CHECKOUT" init -q </dev/null || return 1
        git -C "$CHECKOUT" remote add origin "$UPSTREAM_URL" </dev/null || return 1
    fi
    # 必须有自己的 .git：否则 git 会解析到外层仓库，检查的就不是这个 checkout。
    if [ ! -e "$CHECKOUT/.git" ]; then
        echo "  BLOCKED: ${CHECKOUT} 存在但不是独立 git checkout（缺少 .git），未修改"
        return 1
    fi
    origin="$(git -C "$CHECKOUT" remote get-url origin 2>/dev/null </dev/null)"
    if [ "$origin" != "$UPSTREAM_URL" ]; then
        echo "  BLOCKED: ${CHECKOUT} 的 origin 为 '${origin}'，不是 ${UPSTREAM_URL}，未修改"
        return 1
    fi
    if [ -n "$(git -C "$CHECKOUT" status --porcelain 2>/dev/null </dev/null)" ]; then
        echo "  BLOCKED: ${CHECKOUT} 有未提交或未跟踪的改动，未修改"
        return 1
    fi
    if ! git -C "$CHECKOUT" cat-file -e "${UPSTREAM_SHA}^{commit}" 2>/dev/null </dev/null; then
        echo "  fetch: 只获取固定提交 ${UPSTREAM_SHA:0:12}"
        git -C "$CHECKOUT" fetch -q --depth 1 origin "$UPSTREAM_SHA" </dev/null || {
            echo "  FAIL: 无法获取 ${UPSTREAM_SHA}（网络或权限问题）"
            return 1
        }
    fi
    return 0
}

# 只读校验固定提交：skill 文件集合必须与 SKILL_FILES 完全一致，LICENSE 必须是 Apache-2.0。
verify_upstream_tree() {
    local listed expected
    listed="$(git -C "$CHECKOUT" ls-tree -r --name-only "$UPSTREAM_SHA" -- "$SKILL_SRC" </dev/null)" || return 1
    listed="$(printf '%s\n' "$listed" | sed "s|^${SKILL_SRC}/||")"
    expected="$(printf '%s\n' "${SKILL_FILES[@]}")"
    if [ "$listed" != "$expected" ]; then
        echo "  FAIL: ${UPSTREAM_SHA:0:12} 中的 ${SKILL_SRC} 文件与本脚本不一致（上游已变化或 pin 有误）:"
        printf '%s\n' "$listed" | sed 's/^/           /'
        return 1
    fi
    if ! git -C "$CHECKOUT" cat-file blob "$UPSTREAM_SHA:LICENSE" </dev/null 2>/dev/null | head -n 2 | grep -q "Apache License"; then
        echo "  FAIL: ${UPSTREAM_SHA:0:12} 的 LICENSE 不是 Apache-2.0"
        return 1
    fi
    return 0
}

# npm 与 Node 前置检查（真实安装才需要）。
check_runtime_tools() {
    command -v npm >/dev/null 2>&1 || { echo "  BLOCKED: 缺少 npm（需要 Node 与 npm）"; return 1; }
    node_preflight "$PKG" || return 1
    return 0
}

# 从固定提交写入临时目录 STAGE（位于 skill 目录内，保证 mv 是同一文件系统内的改名）。
stage_skill() {
    local f
    STAGE="$(mktemp -d "${SKILL_ROOT}/.agent-desktop-stage.XXXXXX")" || return 1
    chmod 755 "$STAGE" || return 1
    for f in "${SKILL_FILES[@]}"; do
        mkdir -p "$STAGE/$(dirname "$f")" || return 1
        git -C "$CHECKOUT" cat-file blob "${UPSTREAM_SHA}:${SKILL_SRC}/${f}" </dev/null > "$STAGE/$f" || return 1
    done
    git -C "$CHECKOUT" cat-file blob "${UPSTREAM_SHA}:LICENSE" </dev/null > "$STAGE/LICENSE" || return 1
    cat > "$STAGE/$OWNER_MARKER" <<EOF
managed-by: install-agent-desktop.sh
package: ${NPM_SPEC}
upstream: ${UPSTREAM_URL}
commit: ${UPSTREAM_SHA} (tag v${NPM_VERSION})
license: Apache-2.0 (LICENSE)
EOF
}

install_skill() {
    mkdir -p "$SKILL_ROOT" || return 1
    stage_skill || return 1
    if is_owned "$SKILL_TARGET"; then
        rm -rf "$SKILL_TARGET" || return 1
    fi
    if [ -e "$SKILL_TARGET" ] || [ -L "$SKILL_TARGET" ]; then
        echo "  FAIL: ${SKILL_TARGET} 在写入前被占用，未修改"
        return 1
    fi
    mv "$STAGE" "$SKILL_TARGET" || return 1
    STAGE=""
    echo "  skill: ${SKILL_TARGET}（${UPSTREAM_SHA:0:12}，LICENSE 与标记文件已包含）"
    return 0
}

# 只读校验安装结果：包版本正确，且当前平台的原生二进制存在。不执行 agent-desktop。
check_runtime_package() {
    local root="$1" version status
    version="$(node -e 'process.stdout.write(require(process.argv[1]).version)' "$root/package.json" 2>/dev/null </dev/null)" || version=""
    if [ "$version" != "$NPM_VERSION" ]; then
        echo "  FAIL: 安装的 ${PKG} 版本为 '${version}'，应为 ${NPM_VERSION}"
        return 1
    fi
    node -e '
        const path = require("path");
        const root = process.argv[1];
        const { resolve } = require(path.join(root, "lib", "platform.js"));
        const entry = resolve(process.platform, process.arch);
        if (!entry || !entry.released) process.exit(3);
        process.exit(require("fs").existsSync(path.join(root, "bin", entry.binaryName)) ? 0 : 4);
    ' "$root" </dev/null
    status=$?
    case $status in
        0) ;;
        3) echo "  FAIL: 当前平台 $(uname -sm) 不在 ${PKG} 支持列表中"; return 1 ;;
        *) echo "  FAIL: 包内缺少当前平台的原生二进制（查看上方 npm 输出中的 postinstall 信息）"; return 1 ;;
    esac
    if [ ! -x "$RUNTIME_DIR/node_modules/.bin/${PKG}" ]; then
        echo "  FAIL: 缺少 ${RUNTIME_DIR}/node_modules/.bin/${PKG}"
        return 1
    fi
    return 0
}

# npm 私有前缀安装（不使用 -g）。RUNTIME_DIR 已通过前置检查，因此只会是本脚本的目录。
install_runtime() {
    mkdir -p "$RUNTIME_DIR" || return 1
    printf 'managed-by: install-agent-desktop.sh\n' > "$RUNTIME_DIR/$OWNER_MARKER" || return 1
    # 不运行 postinstall：npm 11 拒绝在项目级安装中使用 --allow-scripts（EALLOWSCRIPTS）。
    # 跳过后上游 bin 经 Node launcher 运行，每次调用约多 125 ms（上游 postinstall 的提示）。
    echo "  npm: install --prefix ${RUNTIME_DIR} --ignore-scripts ${NPM_SPEC}（不使用 -g）"
    npm install --prefix "$RUNTIME_DIR" --ignore-scripts --no-save --no-audit --no-fund "$NPM_SPEC" </dev/null || {
        echo "  FAIL: npm install 失败（见上方 npm 输出）"
        return 1
    }
    check_runtime_package "$RUNTIME_DIR/node_modules/$PKG" || return 1
    echo "  runtime: ${RUNTIME_DIR}/node_modules/.bin/${PKG}（${NPM_SPEC}）"
    return 0
}

# 生成带标记的 launcher：把安装时的 node 目录前置到 PATH，然后 exec 私有前缀中的真实命令（参数逐个保留）。
write_launcher() {
    local node_dir tmp
    node_dir="$(dirname "$(command -v node)")"
    mkdir -p "$(dirname "$LAUNCHER")" || return 1
    tmp="${LAUNCHER}.tmp.$$"
    {
        printf '#!/bin/bash\n'
        printf '# %s: generated by install-agent-desktop.sh; do not edit.\n' "$OWNER_TAG"
        printf 'PATH=%q:"$PATH"\n' "$node_dir"
        printf 'export PATH\n'
        printf 'exec %q "$@"\n' "$RUNTIME_DIR/node_modules/.bin/${PKG}"
    } > "$tmp" || { rm -f "$tmp"; return 1; }
    chmod 755 "$tmp" && mv "$tmp" "$LAUNCHER" || { rm -f "$tmp"; return 1; }
    echo "  launcher: ${LAUNCHER} -> ${RUNTIME_DIR}/node_modules/.bin/${PKG}（node: ${node_dir}）"
    return 0
}

cleanup() {
    if [ -n "$STAGE" ]; then
        rm -rf "$STAGE"
    fi
}
trap cleanup EXIT

command -v git >/dev/null 2>&1 || { echo "BLOCKED: 缺少 git" >&2; exit 1; }

echo "=== ${NPM_SPEC}（上游 ${UPSTREAM_SHA:0:12}，Apache-2.0）==="
if [ -n "$DEST_DIR" ]; then
    echo "模式: --dest 隔离安装（写 skill 目录；checkout 缺失时创建 repos/agent-desktop；不安装 npm 运行时，不写 HOME）"
else
    echo "模式: 真实安装（agent=${AGENT}，skill 目录 ${SKILL_ROOT}）"
fi

# 第一阶段：前置检查。除 checkout 缺失时的创建与获取外，不写入任何路径。
prepare_checkout || exit 1
verify_upstream_tree || exit 1
check_owned_dir "$SKILL_TARGET" || exit 1
if [ -z "$DEST_DIR" ]; then
    check_runtime_tools || exit 1
    check_owned_dir "$RUNTIME_DIR" || exit 1
    check_launcher || exit 1
fi

# 第二阶段：写入。
install_skill || exit 1
if [ -z "$DEST_DIR" ]; then
    install_runtime || exit 1
    write_launcher || exit 1
    case ":${PATH}:" in
        *":${HOME}/.local/bin:"*) ;;
        *) echo "提示: ${HOME}/.local/bin 不在 PATH 中；请自行加入（本脚本不修改 shell 配置）" ;;
    esac
fi

echo ""
echo "完成。"
echo "macOS: 运行 agent-desktop 的应用需要辅助功能权限（系统设置 > 隐私与安全性 > 辅助功能）。"
echo "       本脚本没有请求任何权限；授权与检查见 docs/agent-desktop/INSTALL.md。"
exit 0
