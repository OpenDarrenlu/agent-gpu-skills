#!/usr/bin/env python3
"""Download and query official AMD Instinct ISA PDFs kept in a local cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
SOURCES_FILE = SKILL_DIR / "sources.json"
REFERENCES_DIR = SKILL_DIR / "references"
MANIFEST_FILE = REFERENCES_DIR / "manifest.json"
PDF_HOSTS = {"www.amd.com", "docs.amd.com"}
USER_AGENT = "Mozilla/5.0 (compatible; agent-gpu-skills-amd-instinct-isa/1.0)"
RANGE_CHUNK_BYTES = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{path.stat().st_mtime_ns if path.exists() else 'new'}.tmp")
    temp.write_bytes(data)
    temp.replace(path)


def valid_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc in PDF_HOSTS
        and parsed.path.lower().endswith(".pdf")
    )


def catalog_documents() -> list[dict[str, str]]:
    config = load_json(SOURCES_FILE, {})
    if not isinstance(config, dict) or not isinstance(config.get("documents"), list):
        raise SystemExit(f"invalid ISA catalog: {SOURCES_FILE}")
    documents: list[dict[str, str]] = []
    for item in config["documents"]:
        if not isinstance(item, dict) or not item.get("id") or not item.get("url"):
            continue
        record = {str(key): str(value) for key, value in item.items()}
        if valid_pdf_url(record["url"]):
            documents.append(record)
    return documents


def load_manifest() -> dict[str, dict[str, str]]:
    value = load_json(MANIFEST_FILE, {})
    return value if isinstance(value, dict) else {}


def all_documents() -> dict[str, dict[str, str]]:
    result = {item["id"]: item for item in catalog_documents()}
    for doc_id, item in load_manifest().items():
        if isinstance(item, dict) and item.get("url"):
            result[doc_id] = {str(key): str(value) for key, value in item.items()}
    return result


def reference_path(doc_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_id).strip("._")
    if not safe:
        raise SystemExit("document id cannot be empty")
    return REFERENCES_DIR / f"{safe}.pdf"


def print_catalog() -> int:
    manifest = load_manifest()
    for item in catalog_documents():
        path = reference_path(item["id"])
        state = "downloaded" if path.is_file() and path.stat().st_size else "missing"
        if item["id"] in manifest:
            state += ", manifest"
        print(f"{item['id']:<18} {state:<20} {item.get('generation', ''):<8} {item['title']}")
        print(f"  {item['url']}")
    for doc_id, item in manifest.items():
        if doc_id in {entry["id"] for entry in catalog_documents()}:
            continue
        path = reference_path(doc_id)
        state = "downloaded" if path.is_file() and path.stat().st_size else "missing"
        print(f"{doc_id:<18} {state:<20} custom   {item.get('title', doc_id)}")
        print(f"  {item['url']}")
    return 0


def download_by_ranges(url: str, target: Path) -> None:
    """Fallback for AMD CDN endpoints that fail a normal full GET.

    Some older AMD document paths answer small Range requests reliably while
    closing a full HTTP/2 or HTTP/1.1 stream with a gateway error. Probe the
    total size, then assemble bounded HTTPS Range requests.
    """
    probe = Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urlopen(probe, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        content_length = response.headers.get("Content-Length")
        match = re.search(r"/([0-9]+)$", content_range)
        total = int(match.group(1)) if match else int(content_length or 0)
        if total <= 0:
            raise RuntimeError("Range endpoint did not provide a total PDF size")
        response.read(1)

    with target.open("wb") as output:
        for start in range(0, total, RANGE_CHUNK_BYTES):
            end = min(total - 1, start + RANGE_CHUNK_BYTES - 1)
            expected = end - start + 1
            last_error: Exception | None = None
            for attempt in range(4):
                request = Request(
                    url,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Range": f"bytes={start}-{end}",
                    },
                )
                try:
                    with urlopen(request, timeout=90) as response:
                        chunk = response.read()
                    if len(chunk) != expected:
                        raise RuntimeError(f"expected {expected} bytes, received {len(chunk)}")
                    output.write(chunk)
                    break
                except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                    last_error = exc
                    if attempt == 3:
                        raise
                    time.sleep(min(15.0, 1.5 * (2**attempt)))
            if last_error is not None and output.tell() < end + 1:
                raise last_error


def download_document(doc: dict[str, str], force: bool) -> Path:
    url = doc["url"]
    if not valid_pdf_url(url):
        raise RuntimeError("refusing non-official or non-PDF URL")
    target = reference_path(doc["id"])
    if target.is_file() and target.stat().st_size > 1024 and not force:
        print(f"cached: {doc['id']} -> {target}")
        return target

    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{__import__('os').getpid()}.part")
    command = [
        "curl", "--fail", "--location", "--http1.1", "--retry", "3", "--retry-all-errors",
        "--retry-delay", "2", "--connect-timeout", "15", "--max-time", "90",
        "--silent", "--show-error", "--user-agent", USER_AGENT,
        "--output", str(temp), url,
    ]
    try:
        legacy_cdn_path = "/system/files/TechDocs/" in urlparse(url).path
        try:
            if legacy_cdn_path:
                raise RuntimeError("legacy AMD CDN path")
            subprocess.run(command, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"using HTTPS Range download for {doc['id']} ({exc})", file=sys.stderr)
            download_by_ranges(url, temp)
        if not temp.is_file() or temp.stat().st_size <= 1024:
            raise RuntimeError("downloaded PDF is empty or unexpectedly small")
        if temp.read_bytes()[:5] != b"%PDF-":
            raise RuntimeError("downloaded content is not a PDF")
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()

    manifest = load_manifest()
    manifest[doc["id"]] = {
        **doc,
        "path": str(target.relative_to(SKILL_DIR)),
        "downloaded_at": utc_now(),
        "bytes": str(target.stat().st_size),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    atomic_write(MANIFEST_FILE, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"downloaded: {doc['id']} -> {target} ({target.stat().st_size} bytes)")
    return target


def choose_documents(args: argparse.Namespace) -> list[dict[str, str]]:
    documents = all_documents()
    requested = list(args.documents or [])
    if args.all:
        if requested or args.url:
            raise SystemExit("use either --all, document ids, or --url, not a combination")
        return list(documents.values())
    if args.url:
        if requested:
            raise SystemExit("use either document ids or --url")
        if not valid_pdf_url(args.url):
            raise SystemExit("--url must be an HTTPS PDF hosted by www.amd.com or docs.amd.com")
        doc_id = args.id or re.sub(r"[^A-Za-z0-9._-]+", "_", Path(unquote(urlparse(args.url).path)).stem).lower()
        return [{"id": doc_id, "generation": "custom", "kind": "pdf", "title": args.title or doc_id, "url": args.url}]
    if not requested:
        raise SystemExit("provide document ids, --all, or --url")
    missing = [doc_id for doc_id in requested if doc_id not in documents]
    if missing:
        raise SystemExit("unknown document id(s): " + ", ".join(missing))
    return [documents[doc_id] for doc_id in requested]


def extract_pages(pdf: Path) -> list[str]:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        with tempfile.TemporaryDirectory(prefix="amd-isa-text-") as temp_dir:
            text_path = Path(temp_dir) / "document.txt"
            subprocess.run([pdftotext, "-layout", str(pdf), str(text_path)], check=True)
            return text_path.read_text(errors="replace").split("\f")
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("install pdftotext or Python pdfplumber to query PDFs") from exc
    with pdfplumber.open(pdf) as handle:
        return [(page.extract_text(x_tolerance=1.5, y_tolerance=3) or "") for page in handle.pages]


def query_documents(args: argparse.Namespace) -> int:
    if args.limit < 1 or args.context < 0:
        raise SystemExit("--limit must be >= 1 and --context must be >= 0")
    expression = args.query if args.regex else re.escape(args.query)
    try:
        pattern = re.compile(expression, re.IGNORECASE)
    except re.error as exc:
        raise SystemExit(f"invalid regular expression: {exc}") from exc

    documents = all_documents()
    ids = args.document or sorted(documents)
    shown = 0
    for doc_id in ids:
        if doc_id not in documents:
            print(f"unknown document id: {doc_id}", file=sys.stderr)
            continue
        doc = documents[doc_id]
        path = reference_path(doc_id)
        if not path.is_file():
            continue
        try:
            pages = extract_pages(path)
        except Exception as exc:
            print(f"query failed for {path}: {exc}", file=sys.stderr)
            continue
        for page_number, page in enumerate(pages, 1):
            lines = page.splitlines()
            for index, line in enumerate(lines):
                if not pattern.search(line):
                    continue
                shown += 1
                start = max(0, index - args.context)
                end = min(len(lines), index + args.context + 1)
                print(f"[{shown}] {doc_id} — {doc.get('title', doc_id)}")
                print(f"    source: {doc['url']}")
                print(f"    local: {path}")
                print(f"    PDF page: {page_number}")
                for line_number in range(start, end):
                    marker = ">" if line_number == index else " "
                    print(f"  {marker}{line_number + 1:6d}: {lines[line_number]}")
                print()
                if shown >= args.limit:
                    return shown
                break
    if shown == 0:
        print("No matches in downloaded ISA PDFs.", file=sys.stderr)
        return 1
    print(f"matches_shown: {shown}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list catalog entries and local download state")

    download = subparsers.add_parser("download", help="download official PDFs to ignored local references")
    download.add_argument("documents", nargs="*", help="catalog ids such as cdna5-isa")
    download.add_argument("--all", action="store_true")
    download.add_argument("--url", help="download a new official PDF not yet in the catalog")
    download.add_argument("--id", help="id for --url; defaults to the URL filename")
    download.add_argument("--title", help="title for --url")
    download.add_argument("--force", action="store_true")
    download.add_argument("--accept-document-terms", action="store_true", required=True)

    query = subparsers.add_parser("query", help="search downloaded PDFs and report page numbers")
    query.add_argument("query")
    query.add_argument("--document", action="append")
    query.add_argument("--regex", action="store_true")
    query.add_argument("--limit", type=int, default=25)
    query.add_argument("--context", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "list":
        return print_catalog()
    if args.command == "download":
        failures: list[str] = []
        for doc in choose_documents(args):
            try:
                download_document(doc, args.force)
            except Exception as exc:
                failures.append(f"{doc['id']}: {type(exc).__name__}: {exc}")
                print(f"download failed: {failures[-1]}", file=sys.stderr)
        return 1 if failures else 0
    return query_documents(args)


if __name__ == "__main__":
    raise SystemExit(main())
