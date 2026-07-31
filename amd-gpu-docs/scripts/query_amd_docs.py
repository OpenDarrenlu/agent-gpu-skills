#!/usr/bin/env python3
"""Query the local AMD documentation cache and installed amd/skills."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SKILL_DIR.parent
CACHE_DIR = SKILL_DIR / "cache"
MANIFEST_FILE = CACHE_DIR / "manifest.json"
SOURCES_FILE = SKILL_DIR / "sources.json"
TEXT_SUFFIXES = {".md", ".rst", ".txt", ".py", ".sh", ".json", ".yaml", ".yml"}


def load_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def compile_pattern(query: str, regex: bool) -> re.Pattern[str]:
    expression = query if regex else re.escape(query)
    try:
        return re.compile(expression, re.IGNORECASE)
    except re.error as exc:
        raise SystemExit(f"invalid regular expression: {exc}") from exc


def manifest_files(manifest: dict) -> list[tuple[Path, str]]:
    chosen: list[tuple[Path, str]] = []
    for url, record in manifest.get("documents", {}).items():
        relative = record.get("source_path") or record.get("text_path")
        if not relative:
            continue
        path = SKILL_DIR / relative
        if path.is_file():
            chosen.append((path, record.get("source_url") or record.get("final_url") or url))
    return chosen


def local_skill_files() -> list[tuple[Path, str]]:
    config = load_json(SOURCES_FILE, {})
    results: list[tuple[Path, str]] = []
    if not isinstance(config, dict):
        return results
    for item in config.get("local_roots", []):
        root = (SKILL_DIR / item["relative_path"]).resolve()
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 5_000_000:
                results.append((path, f"local:{item['id']}/{path.relative_to(root)}"))
    return results


def fallback_cache_files() -> list[tuple[Path, str]]:
    results: list[tuple[Path, str]] = []
    for root in (CACHE_DIR / "sources", CACHE_DIR / "text"):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                results.append((path, str(path)))
    return results


def print_match(path: Path, source: str, lines: list[str], index: int, context: int, ordinal: int) -> None:
    start = max(0, index - context)
    end = min(len(lines), index + context + 1)
    print(f"[{ordinal}] {source}")
    print(f"    local: {path}")
    for line_number in range(start, end):
        marker = ">" if line_number == index else " "
        print(f"  {marker}{line_number + 1:6d}: {lines[line_number]}")
    print()


def search_text_files(
    files: list[tuple[Path, str]], pattern: re.Pattern[str], source_filter: str | None,
    limit: int, context: int,
) -> int:
    count = 0
    seen_files: set[Path] = set()
    for path, source in files:
        resolved = path.resolve()
        if resolved in seen_files:
            continue
        seen_files.add(resolved)
        if source_filter and source_filter.lower() not in (source + " " + str(path)).lower():
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except (OSError, UnicodeError):
            continue
        last_printed = -1
        for index, line in enumerate(lines):
            if not pattern.search(line) or index <= last_printed:
                continue
            count += 1
            print_match(path, source, lines, index, context, count)
            last_printed = index + context
            if count >= limit:
                return count
    return count


def search_pdfs(manifest: dict, pattern: re.Pattern[str], source_filter: str | None, limit: int, start: int) -> int:
    try:
        import pdfplumber
    except ImportError:
        print("PDF query skipped: install Python pdfplumber", file=sys.stderr)
        return start

    count = start
    for url, record in manifest.get("pdfs", {}).items():
        path = SKILL_DIR / record.get("path", "")
        if not path.is_file():
            continue
        if source_filter and source_filter.lower() not in (url + " " + str(path)).lower():
            continue
        try:
            with pdfplumber.open(path) as pdf:
                for page_number, page in enumerate(pdf.pages, 1):
                    text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
                    matching = [line for line in text.splitlines() if pattern.search(line)]
                    if not matching:
                        continue
                    count += 1
                    print(f"[{count}] {url}")
                    print(f"    local: {path}")
                    print(f"    PDF page: {page_number}")
                    for line in matching[:5]:
                        print(f"  > {line}")
                    print()
                    if count >= limit:
                        return count
        except Exception as exc:
            print(f"PDF query failed for {path}: {exc}", file=sys.stderr)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="literal text by default")
    parser.add_argument("--regex", action="store_true")
    parser.add_argument("--source", help="filter by source URL, id, or local path")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--include-pdfs", action="store_true")
    parser.add_argument("--stats", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_json(MANIFEST_FILE, {})
    if not isinstance(manifest, dict):
        manifest = {}

    if args.stats:
        print(f"profile: {manifest.get('profile', 'not-synced')}")
        print(f"generated_at: {manifest.get('generated_at', 'n/a')}")
        print(f"documents: {len(manifest.get('documents', {}))}")
        print(f"pdfs: {len(manifest.get('pdfs', {}))}")
        print(f"errors: {len(manifest.get('errors', []))}")
        if not args.query:
            return 0

    if not args.query:
        raise SystemExit("query is required unless --stats is used")
    if args.limit < 1 or args.context < 0:
        raise SystemExit("--limit must be >= 1 and --context must be >= 0")

    pattern = compile_pattern(args.query, args.regex)
    files = manifest_files(manifest)
    if not files:
        files = fallback_cache_files()
    files.extend(local_skill_files())
    count = search_text_files(files, pattern, args.source, args.limit, args.context)
    if args.include_pdfs and count < args.limit:
        count = search_pdfs(manifest, pattern, args.source, args.limit, count)

    if count == 0:
        if not MANIFEST_FILE.exists():
            print("No synchronized AMD docs cache. Run sync_amd_docs.py first.", file=sys.stderr)
        else:
            print("No matches.", file=sys.stderr)
        return 1
    print(f"matches_shown: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
