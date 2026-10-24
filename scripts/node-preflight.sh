#!/bin/bash
# Node 版本预检（被 install-external.sh 与 install-rea-mcp.sh source）。
# 只检查 PATH 中的 node，不修改全局 Node，不切换版本。
#
# node_preflight <npm 包名>
#   返回 0: 满足该包的 engines；2: 不满足或找不到 node（已打印 BLOCKED 与修复指引）。
#
# 已知规则以“实际安装的已发布 npm 包”的 engines 为准（安装器从 npm 安装运行时，不从源码构建）:
#   rea-agents@6.7.0 : ^22.19.0 || ^24.11.0 || >=26.0.0   （已安装包 lib/node_modules/rea-agents/package.json）
#   @jackwener/wx-cli@0.3.0 / wx-cli : >=14
# 注意：源码 checkout（repos/rea/package.json）中同一版本号的 engines 更新（^22.23.4 || ^24.22.1 || >=26.12.1）
# 不适用于 npm 安装的运行时，因此这里不使用源码的 engines。

# 把 vX.Y.Z 转为可比较的整数 X*1e6 + Y*1e3 + Z。
node_version_number() {
    local v="${1#v}" major minor patch
    IFS=. read -r major minor patch <<< "$v"
    echo $(( ${major:-0} * 1000000 + ${minor:-0} * 1000 + ${patch:-0} ))
}

node_preflight() {
    local pkg="$1" v major n ok=false
    if ! command -v node >/dev/null 2>&1; then
        echo "  BLOCKED: 找不到 node（PATH 中无 node）。安装 Node 24.22.1 或更高的 24.x 后，把其 bin 目录放在 PATH 前面"
        return 2
    fi
    v="$(node -v 2>/dev/null)"
    if ! [[ "$v" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        echo "  BLOCKED: 无法读取 node 版本（node -v 输出为 '${v}'）"
        return 2
    fi
    major="${v#v}"
    major="${major%%.*}"
    n="$(node_version_number "$v")"

    case "$pkg" in
        rea-agents)
            # 已发布包的 engines: ^22.19.0 || ^24.11.0 || >=26.0.0
            if [ "$major" -eq 22 ] && [ "$n" -ge 22019000 ]; then ok=true
            elif [ "$major" -eq 24 ] && [ "$n" -ge 24011000 ]; then ok=true
            elif [ "$major" -ge 26 ]; then ok=true
            fi
            if [ "$ok" = false ]; then
                echo "  BLOCKED: 当前 node ${v} 不满足已安装 rea-agents 包的 engines（^22.19.0 || ^24.11.0 || >=26.0.0）"
                echo "           请把兼容版本放在 PATH 前面，例如: PATH=\$HOME/.nvm/versions/node/v24.22.1/bin:\$PATH"
                echo "           不会自动修改全局 Node。"
                return 2
            fi
            ;;
        wx-cli|@jackwener/wx-cli)
            if [ "$major" -ge 14 ]; then ok=true; fi
            if [ "$ok" = false ]; then
                echo "  BLOCKED: 当前 node ${v} 低于 wx-cli 要求的 >=14"
                return 2
            fi
            ;;
        *)
            ;;
    esac
    return 0
}
