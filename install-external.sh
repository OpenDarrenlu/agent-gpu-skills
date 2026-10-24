#!/bin/bash
# 获取 sources/external-sources.conf 中的外部来源，并安装其运行时 CLI。
# skill 目录由 install.sh 负责；本脚本只处理 checkout（repos/<name>/）、runtime 可执行文件，
# 以及 runtime-skills（npm 包附带的 skills 目录）的链接。
#
# 用法:
#   bash install-external.sh [--update] [--no-runtime]
#
#   默认（ensure）: checkout 缺失时才 clone；已存在时不拉取，因此 --no-update 也能补齐缺失来源。
#   --update      : 拉取远端并切换到 ref（UNPINNED 则 pull --ff-only，并给出警告）。
#   --no-runtime  : 只获取源码，不安装运行时 CLI。
#
# 退出码: 任一来源 FAIL / BLOCKED 时为 1。缺失必须可见，不能静默跳过。
# runtime 安装会写入用户全局位置（uv tool / npm -g），因此 --dest 隔离安装和 --no-integrations 不调用本脚本。

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$SCRIPT_DIR/sources/external-sources.conf"
UPDATE=false
RUNTIME=true
FAILS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --update)     UPDATE=true; shift ;;
        --no-runtime) RUNTIME=false; shift ;;
        -h|--help)    sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "$CONF" ]; then
    echo "错误: 缺少 $CONF" >&2
    exit 1
fi

# shellcheck source=scripts/node-preflight.sh
. "$SCRIPT_DIR/scripts/node-preflight.sh"

# 将 checkout 切换到固定 commit；ref 不在本地时先 fetch。
checkout_pinned() {
    local dir="$1" ref="$2"
    if ! git -C "$dir" cat-file -e "$ref^{commit}" 2>/dev/null; then
        git -C "$dir" fetch -q origin "$ref" </dev/null || return 1
    fi
    git -C "$dir" checkout -q --detach "$ref" </dev/null
}

# 获取源码。成功返回 0；失败（网络、权限、ref 不存在）返回 1。
fetch_source() {
    local url="$1" ref="$2" dir="$3"
    if [ ! -d "$dir/.git" ]; then
        echo "  clone: $url"
        git clone -q --template= --filter=blob:none "$url" "$dir" </dev/null || return 1
    elif [ "$UPDATE" = true ]; then
        git -C "$dir" fetch -q --tags origin </dev/null || return 1
    fi

    if [ "$ref" != "UNPINNED" ]; then
        checkout_pinned "$dir" "$ref" || return 1
    else
        echo "  WARN: 未固定 ref（UNPINNED），跟随默认分支；固定后请写入 sources/external-sources.conf"
        if [ "$UPDATE" = true ]; then
            git -C "$dir" pull -q --ff-only </dev/null || return 1
        fi
    fi
    return 0
}

# 安装运行时 CLI，链接 runtime-skills（如需要），并确认可执行名可见且 --version 成功。
# 参数: checkout 目录  runtime 规格  skills_dir
# 返回: 0 成功；2 缺少工具或规格不完整（BLOCKED）；1 安装或校验失败。
install_runtime() {
    local dir="$1" spec="$2" skills_dir="$3"
    local kind="${spec%%:*}"
    local rest="${spec#*:}"
    local bin="${rest##*:}"
    local target="${rest%:*}"
    local root pkg

    case "$kind" in
        uv)
            command -v uv >/dev/null 2>&1 || { echo "  BLOCKED: 缺少 uv（见 https://docs.astral.sh/uv/）"; return 2; }
            uv tool install --force "$dir/$target" </dev/null || return 1
            ;;
        npm)
            case "$target" in
                *@*) ;;
                *) echo "  BLOCKED: npm 规格必须固定版本（pkg@version），当前为 '$target'"; return 2 ;;
            esac
            command -v npm >/dev/null 2>&1 || { echo "  BLOCKED: 缺少 npm（需要 Node 24 或包要求的版本）"; return 2; }
            node_preflight "${target%@*}" || return 2
            npm install -g "$target" </dev/null || return 1
            if [ "$skills_dir" = runtime-skills ]; then
                pkg="${target%@*}"
                root="$(npm root -g </dev/null)" || return 1
                if [ ! -d "$root/$pkg/skills" ]; then
                    echo "  FAIL: 已安装的 ${pkg} 中没有 skills/ 目录（${root}/${pkg}）"
                    return 1
                fi
                ln -sfn "$root/$pkg/skills" "$dir/runtime-skills"
                echo "  runtime-skills -> $root/$pkg/skills"
            fi
            ;;
        *)
            echo "  BLOCKED: 未知 runtime 类型 '$kind'"
            return 2
            ;;
    esac

    if ! command -v "$bin" >/dev/null 2>&1; then
        echo "  FAIL: 安装完成但 PATH 中找不到 '$bin'（检查 npm 全局 bin 或 uv tool 目录是否在 PATH）"
        return 1
    fi
    if ! "$bin" --version >/dev/null 2>&1 </dev/null; then
        echo "  FAIL: '$bin --version' 失败（退出码非 0）"
        return 1
    fi
    echo "  runtime OK: $bin -> $(command -v "$bin") ($("$bin" --version 2>/dev/null </dev/null | head -1))"
    return 0
}

while IFS='|' read -r name url ref mode skills_dir runtime provenance; do
    case "$name" in ''|'#'*) continue ;; esac
    dir="$SCRIPT_DIR/repos/$name"

    echo ""
    echo "=== $name (mode=$mode, ref=$ref) ==="

    if ! fetch_source "$url" "$ref" "$dir"; then
        echo "  FAIL: 无法获取 ${url}（网络、权限或 ref 不存在）"
        FAILS=$((FAILS + 1))
        continue
    fi
    echo "  HEAD: $(git -C "$dir" rev-parse HEAD </dev/null)"

    if [ "$mode" = pending ]; then
        echo "  BLOCKED: mode=pending，manifest 尚未检视，不能安装。检视后在 sources/external-sources.conf 填写 mode/runtime"
        FAILS=$((FAILS + 1))
        continue
    fi

    if [ "$RUNTIME" = true ] && [ "$runtime" != none ]; then
        if [ "$runtime" = pending ]; then
            echo "  BLOCKED: runtime=pending，尚未确定运行时安装方式"
            FAILS=$((FAILS + 1))
            continue
        fi
        install_runtime "$dir" "$runtime" "$skills_dir" </dev/null
        rc=$?
        if [ $rc -eq 2 ]; then
            FAILS=$((FAILS + 1))
        elif [ $rc -ne 0 ]; then
            echo "  FAIL: runtime 安装失败（${runtime}）"
            FAILS=$((FAILS + 1))
        fi
    fi
done < "$CONF"

echo ""
if [ $FAILS -gt 0 ]; then
    echo "外部来源: $FAILS 项未就绪（见上方 FAIL/BLOCKED）"
    exit 1
fi
echo "外部来源: 全部就绪"
