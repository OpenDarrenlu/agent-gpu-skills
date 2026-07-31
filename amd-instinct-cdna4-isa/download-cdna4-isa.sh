#!/bin/bash
# Fetch the AMD CDNA4 ISA PDF for a user's local, licensed reference copy.
# The document is intentionally not vendored by this repository.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$SCRIPT_DIR/references/amd-instinct-cdna4-instruction-set-architecture.pdf"
URL="https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf"
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=true; shift ;;
        -h|--help)
            echo "用法: bash download-cdna4-isa.sh [--force]"
            echo "下载官方 AMD CDNA4 ISA PDF 到本 skill 的 references/（不提交到 Git）。"
            exit 0
            ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

if [ -s "$DEST" ] && [ "$FORCE" = false ]; then
    echo "已存在: $DEST"
    echo "如需刷新，运行: bash download-cdna4-isa.sh --force"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
tmp="$DEST.part.$$"
trap 'unlink "$tmp" 2>/dev/null || true' EXIT INT TERM

echo "从 AMD 官方 URL 下载 CDNA4 ISA PDF..."
if ! curl --fail --location --retry 1 --retry-delay 2 \
    --connect-timeout 10 --max-time 45 --output "$tmp" "$URL"; then
    # Some regional CDNs answer HEAD/range requests but stall a normal full
    # download. Keep the TLS hostname/certificate and try each DNS A record
    # with one exact byte-range request.
    downloaded=false
    if command -v dig >/dev/null 2>&1; then
        echo "普通下载失败，尝试 AMD CDN 的其他地址..."
        for ip in $(dig +short www.amd.com A | awk '/^[0-9]+(\.[0-9]+){3}$/'); do
            size="$(curl --fail --silent --show-error --location --http1.1 \
                --connect-timeout 10 --max-time 30 \
                --resolve "www.amd.com:443:$ip" --head "$URL" | \
                awk 'tolower($1) == "content-length:" {gsub("\r", "", $2); n=$2} END {print n}')"
            case "$size" in
                ''|*[!0-9]*) continue ;;
            esac
            end=$((size - 1))
            if curl --fail --silent --show-error --location --http1.1 \
                --connect-timeout 10 --max-time 90 \
                --resolve "www.amd.com:443:$ip" --range "0-$end" \
                --output "$tmp" "$URL"; then
                downloaded=true
                break
            fi
        done
    fi
    if [ "$downloaded" != true ]; then
        echo "下载失败。可在浏览器打开官方 URL，下载后保存为: $DEST" >&2
        exit 1
    fi
fi
test -s "$tmp"
mv "$tmp" "$DEST"
trap - EXIT INT TERM
echo "完成: $DEST"
echo "注意: 该文件受 AMD Specification Agreement 约束，请勿提交或再分发。"
