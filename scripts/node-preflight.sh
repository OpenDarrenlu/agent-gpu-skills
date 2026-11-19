#!/bin/bash
# Node 版本预检（被 install-external.sh source）。
# 只检查 PATH 中的 node，不修改全局 Node，不切换版本。
#
# node_preflight <npm 包名>
#   返回 0: 满足该包的 engines；2: 不满足或找不到 node（已打印 BLOCKED 与修复指引）。
#
# 已知规则以“实际安装的已发布 npm 包”的 engines 为准（安装器从 npm 安装运行时，不从源码构建）:
#   agent-desktop@1.3.1 : >=18（已发布包 npm/package.json 的 engines，见 install-agent-desktop.sh）
# 未知包只检查 node 可用，不判断具体 engines。

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
        agent-desktop)
            if [ "$major" -ge 18 ]; then ok=true; fi
            if [ "$ok" = false ]; then
                echo "  BLOCKED: 当前 node ${v} 低于 agent-desktop 要求的 >=18"
                return 2
            fi
            ;;
        *)
            ;;
    esac
    return 0
}
