#!/bin/bash
# Search a locally downloaded CDNA4 ISA PDF without retaining extracted text.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PDF="$SCRIPT_DIR/references/amd-instinct-cdna4-instruction-set-architecture.pdf"

if [ $# -lt 1 ]; then
    echo "用法: bash query-cdna4-isa.sh <regex>" >&2
    exit 2
fi
if [ ! -s "$PDF" ]; then
    echo "未找到本地 PDF: $PDF" >&2
    echo "先运行: bash download-cdna4-isa.sh" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "缺少 python3，无法生成带 PDF 页码的检索结果。" >&2
    exit 1
fi

pattern="$1"
if command -v pdftotext >/dev/null 2>&1; then
    mode="text"
    source="$(mktemp "${TMPDIR:-/tmp}/cdna4-isa.XXXXXX")"
    trap 'unlink "$source" 2>/dev/null || true' EXIT INT TERM
    pdftotext -layout "$PDF" "$source"
elif python3 -c 'import pdfplumber' >/dev/null 2>&1; then
    mode="pdfplumber"
    source="$PDF"
else
    echo "缺少 PDF 文本提取器。安装 pdftotext（macOS: 'brew install poppler'；Debian/Ubuntu: 'apt install poppler-utils'）或 Python pdfplumber。" >&2
    exit 1
fi

python3 - "$mode" "$source" "$pattern" <<'PY'
import re
import sys
from pathlib import Path

mode = sys.argv[1]
source = Path(sys.argv[2])
try:
    pattern = re.compile(sys.argv[3], re.IGNORECASE)
except re.error as exc:
    print(f"无效正则表达式: {exc}", file=sys.stderr)
    raise SystemExit(2)

if mode == "text":
    pages = source.read_text(errors="replace").split("\f")
else:
    import pdfplumber

    with pdfplumber.open(source) as pdf:
        pages = [page.extract_text(x_tolerance=1.5, y_tolerance=3) or "" for page in pdf.pages]

found = False
for page_number, page in enumerate(pages, 1):
    lines = page.splitlines()
    last_end = -1
    for index, line in enumerate(lines):
        if not pattern.search(line):
            continue
        start = max(0, index - 3)
        end = min(len(lines), index + 4)
        if start <= last_end:
            continue
        found = True
        print(f"===== PDF PAGE {page_number}; text lines {start + 1}-{end} =====")
        for line_number in range(start, end):
            marker = ">" if line_number == index else " "
            print(f"{marker}{line_number + 1:5d}: {lines[line_number]}")
        print()
        last_end = end - 1

raise SystemExit(0 if found else 1)
PY
