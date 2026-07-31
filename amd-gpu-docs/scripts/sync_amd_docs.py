#!/usr/bin/env python3
"""Synchronize a local, searchable cache of official AMD technical docs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
CACHE_DIR = SKILL_DIR / "cache"
SOURCES_FILE = SKILL_DIR / "sources.json"
MANIFEST_FILE = CACHE_DIR / "manifest.json"
INDEX_FILE = CACHE_DIR / "INDEX.md"
USER_AGENT = "agent-gpu-skills-amd-docs/1.0 (+https://github.com/OpenDarrenlu/agent-gpu-skills)"

HTML_EXTENSIONS = {"", ".html", ".htm"}
SKIP_SUFFIXES = {
    ".7z", ".avi", ".bin", ".bz2", ".css", ".csv", ".doc", ".docx",
    ".gif", ".gz", ".ico", ".jpeg", ".jpg", ".js", ".json", ".mp3",
    ".mp4", ".png", ".ppt", ".pptx", ".svg", ".tar", ".tgz", ".webp",
    ".xls", ".xlsx", ".xml", ".xz", ".zip",
}
SKIP_PATH_PARTS = (
    "/_static/", "/_images/", "/_downloads/", "/search.html",
    "/genindex.html", "/py-modindex.html",
)
PDF_HOSTS = {"www.amd.com", "docs.amd.com"}
SITEMAP_ONLY_HOSTS = {"gpuopen.com"}
WRITE_LOCK = threading.Lock()


@dataclass
class ParsedPage:
    title: str
    links: list[str]
    source_links: list[str]
    text: str


class DocumentationHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.source_links: list[str] = []
        self.title_parts: list[str] = []
        self.all_text: list[str] = []
        self.main_text: list[str] = []
        self._skip_depth = 0
        self._main_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {"main", "article"}:
            self._main_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = attrs_dict.get("href")
            if href:
                self.links.append(href)
                if "/_sources/" in href or href.startswith("_sources/"):
                    self.source_links.append(href)
        if tag in {"p", "div", "section", "li", "tr", "h1", "h2", "h3", "h4", "pre", "br"}:
            self._append_text("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"main", "article"} and self._main_depth:
            self._main_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "li", "tr", "h1", "h2", "h3", "h4", "pre"}:
            self._append_text("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        self._append_text(data)

    def _append_text(self, text: str) -> None:
        if self._skip_depth:
            return
        self.all_text.append(text)
        if self._main_depth:
            self.main_text.append(text)

    def result(self) -> ParsedPage:
        selected = self.main_text if any(x.strip() for x in self.main_text) else self.all_text
        text = normalize_text("".join(selected))
        title = normalize_inline("".join(self.title_parts))
        return ParsedPage(title, self.links, self.source_links, text)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def load_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_bytes(data)
    temp.replace(path)


def sanitize_component(component: str) -> str:
    component = unquote(component)
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", component).strip("._")
    return component or "index"


def local_path(root: Path, url: str, suffix: str | None = None) -> Path:
    parsed = urlparse(url)
    parts = [sanitize_component(part) for part in parsed.path.split("/") if part]
    # Sphinx URLs are often directory URLs, but redirects and hand-written
    # links do not always retain the trailing slash.  Always map an
    # extensionless final component to a file so two pages cannot collide
    # with one another as a directory and a file in the cache.
    if not parts or parsed.path.endswith("/") or not Path(parts[-1]).suffix:
        parts.append("index.html")
    target = root / sanitize_component(parsed.netloc) / Path(*parts)
    if suffix is not None:
        target = target.with_suffix(suffix)
    return target


def canonicalize(url: str) -> str:
    parsed = urlparse(url.replace("\\", "/"))
    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and not path.endswith("/") and not Path(path).suffix:
        path += "/"
    return urlunparse((scheme, host, path, "", "", ""))


def is_current_docs_path(host: str, path: str) -> bool:
    if host in {"rocm.blogs.amd.com", "gpuopen.com"}:
        return True
    markers = ("/latest", "/en/latest", "/en/main", "/main/")
    return any(marker in path for marker in markers)


def page_allowed(url: str, allowed_hosts: set[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in allowed_hosts:
        return False
    path_lower = parsed.path.lower()
    if any(part in path_lower for part in SKIP_PATH_PARTS):
        return False
    suffix = Path(path_lower).suffix
    if suffix in SKIP_SUFFIXES or suffix == ".pdf":
        return False
    if suffix not in HTML_EXTENSIONS:
        return False
    return is_current_docs_path(parsed.netloc, path_lower)


def fetch_bytes(url: str, timeout: int = 45, attempts: int = 4) -> tuple[bytes, str, str]:
    """Fetch a URL with bounded retries for transient throttling/server errors."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        try:
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                return response.read(), response.geturl(), content_type
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                delay = 0.0
            # Keep a server-provided delay bounded, while still backing off
            # exponentially when the response has no Retry-After header.
            delay = max(delay, min(30.0, 1.5 * (2**attempt)))
            time.sleep(delay)
        except (URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(min(30.0, 1.5 * (2**attempt)))
    assert last_error is not None
    raise last_error


def parse_html(content: bytes) -> ParsedPage:
    parser = DocumentationHTMLParser()
    parser.feed(content.decode("utf-8", "replace"))
    return parser.result()


def source_extension(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".md", ".rst", ".txt"} else ".txt"


def process_page(url: str, refresh: bool) -> dict[str, object]:
    html_path = local_path(CACHE_DIR / "html", url)
    fetched = False
    final_url = url
    if html_path.exists() and not refresh:
        content = html_path.read_bytes()
        content_type = "text/html"
    else:
        content, final_url, content_type = fetch_bytes(url)
        if "html" not in content_type:
            raise ValueError(f"unexpected content type {content_type}")
        atomic_write(html_path, content)
        fetched = True

    parsed = parse_html(content)
    text_path = local_path(CACHE_DIR / "text", final_url, ".txt")
    header = f"SOURCE_URL: {final_url}\nTITLE: {parsed.title}\n\n"
    atomic_write(text_path, (header + parsed.text).encode("utf-8"))

    normalized_links: list[str] = []
    pdf_links: list[str] = []
    for href in parsed.links:
        joined = canonicalize(urljoin(final_url, href))
        parsed_joined = urlparse(joined)
        if parsed_joined.path.lower().endswith(".pdf") and parsed_joined.netloc in PDF_HOSTS:
            pdf_links.append(joined)
        else:
            normalized_links.append(joined)

    source_path: Path | None = None
    source_url: str | None = None
    for href in parsed.source_links:
        candidate = canonicalize(urljoin(final_url, href))
        candidate_path = local_path(CACHE_DIR / "sources", candidate, source_extension(candidate))
        try:
            if candidate_path.exists() and not refresh:
                source_content = candidate_path.read_bytes()
            else:
                source_content, resolved, source_type = fetch_bytes(candidate)
                if source_type not in {"text/markdown", "text/plain", "text/x-rst"}:
                    continue
                candidate = resolved
                candidate_path = local_path(CACHE_DIR / "sources", candidate, source_extension(candidate))
                atomic_write(candidate_path, source_content)
            if source_content.strip():
                source_path = candidate_path
                source_url = candidate
                break
        except (HTTPError, URLError, TimeoutError, ValueError):
            continue

    return {
        "url": url,
        "final_url": final_url,
        "title": parsed.title,
        "html_path": str(html_path.relative_to(SKILL_DIR)),
        "text_path": str(text_path.relative_to(SKILL_DIR)),
        "source_path": str(source_path.relative_to(SKILL_DIR)) if source_path else None,
        "source_url": source_url,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "fetched": fetched,
        "links": normalized_links,
        "pdf_links": sorted(set(pdf_links)),
    }


def sitemap_urls(url: str) -> list[str]:
    content, _, _ = fetch_bytes(url)
    root = ElementTree.fromstring(content)
    return [
        canonicalize(element.text.strip())
        for element in root.iter()
        if element.tag.endswith("loc") and element.text
    ]


def download_pdf(url: str, refresh: bool) -> dict[str, object]:
    target = local_path(CACHE_DIR / "pdfs", url, ".pdf")
    if target.exists() and target.stat().st_size > 0 and not refresh:
        return {"url": url, "path": str(target.relative_to(SKILL_DIR)), "cached": True, "bytes": target.stat().st_size}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.part")
    command = [
        "curl", "--fail", "--location", "--retry", "2", "--retry-delay", "2",
        "--connect-timeout", "15", "--max-time", "180", "--silent", "--show-error",
        "--output", str(temp), url,
    ]
    try:
        subprocess.run(command, check=True)
        if not temp.exists() or temp.stat().st_size == 0:
            raise RuntimeError("downloaded PDF is empty")
        temp.replace(target)
        return {"url": url, "path": str(target.relative_to(SKILL_DIR)), "cached": False, "bytes": target.stat().st_size}
    finally:
        if temp.exists():
            temp.unlink()


def selected_sources(config: dict[str, object], profile: str) -> tuple[list[dict], list[dict], list[dict]]:
    def pick(key: str) -> list[dict]:
        return [item for item in config.get(key, []) if profile in item.get("profiles", [])]

    return pick("web_roots"), pick("sitemaps"), pick("pdf_discovery_pages")


def write_index(profile: str, documents: dict[str, dict], pdfs: dict[str, dict], errors: list[dict], config: dict) -> None:
    by_host: dict[str, int] = {}
    source_files = 0
    for record in documents.values():
        host = urlparse(record.get("final_url", record["url"])).netloc
        by_host[host] = by_host.get(host, 0) + 1
        source_files += int(bool(record.get("source_path")))

    lines = [
        "# AMD local documentation index",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Profile: `{profile}`",
        f"- HTML pages: `{len(documents)}`",
        f"- Original Markdown/rST sources: `{source_files}`",
        f"- Local PDFs: `{len(pdfs)}`",
        f"- Fetch errors: `{len(errors)}`",
        "",
        "## Pages by host",
        "",
    ]
    for host, count in sorted(by_host.items()):
        lines.append(f"- `{host}`: {count}")
    lines += ["", "## Local sources", ""]
    for item in config.get("local_roots", []):
        lines.append(f"- `{item['id']}`: `{item['relative_path']}`")
    if errors:
        lines += ["", "## Recent errors", ""]
        for item in errors[:50]:
            lines.append(f"- `{item['url']}` — {item['error']}")
    atomic_write(INDEX_FILE, ("\n".join(lines) + "\n").encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "all"), default="core")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means no explicit page limit")
    parser.add_argument("--refresh", action="store_true", help="refetch files already in the local cache")
    parser.add_argument("--include-pdfs", action="store_true")
    parser.add_argument("--accept-document-terms", action="store_true")
    parser.add_argument("--list-sources", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > 64:
        raise SystemExit("--workers must be between 1 and 64")
    if args.max_pages < 0:
        raise SystemExit("--max-pages must be >= 0")
    if args.include_pdfs and not args.accept_document_terms:
        raise SystemExit("--include-pdfs requires --accept-document-terms")

    config = load_json(SOURCES_FILE, {})
    if not isinstance(config, dict):
        raise SystemExit(f"invalid sources file: {SOURCES_FILE}")
    roots, sitemaps, discovery_pages = selected_sources(config, args.profile)

    if args.list_sources:
        for item in roots:
            print(f"web     {item['id']:<20} {item['url']}")
        for item in sitemaps:
            print(f"sitemap {item['id']:<20} {item['url']}")
        for item in discovery_pages:
            print(f"pdf-seed {item['id']:<17} {item['url']}")
        for item in config.get("local_roots", []):
            print(f"local   {item['id']:<20} {item['relative_path']}")
        return 0

    allowed_hosts = {urlparse(item["url"]).netloc for item in roots}
    allowed_hosts.update(urlparse(item["url"]).netloc for item in sitemaps)
    seeds = [canonicalize(item["url"]) for item in roots]
    seeds.extend(canonicalize(item["url"]) for item in discovery_pages)
    sitemap_errors: list[dict] = []
    for item in sitemaps:
        try:
            seeds.extend(url for url in sitemap_urls(item["url"]) if page_allowed(url, allowed_hosts))
        except Exception as exc:  # network/XML error is recorded, not fatal to other sources
            sitemap_errors.append({"url": item["url"], "error": f"{type(exc).__name__}: {exc}"})

    pending = deque(dict.fromkeys(seeds))
    seen: set[str] = set()
    documents: dict[str, dict] = {}
    pdf_urls: set[str] = set()
    errors = list(sitemap_errors)

    print(f"AMD docs sync: profile={args.profile}, workers={args.workers}, initial_pages={len(pending)}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        active: dict[object, str] = {}
        while pending or active:
            while pending and len(active) < args.workers:
                url = pending.popleft()
                if url in seen or not page_allowed(url, allowed_hosts):
                    continue
                if args.max_pages and len(seen) >= args.max_pages:
                    pending.clear()
                    break
                seen.add(url)
                active[executor.submit(process_page, url, args.refresh)] = url

            if not active:
                break
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                url = active.pop(future)
                try:
                    record = future.result()
                    links = record.pop("links")
                    pdf_urls.update(record.pop("pdf_links"))
                    documents[url] = record
                    host = urlparse(url).netloc
                    if host not in SITEMAP_ONLY_HOSTS:
                        for link in links:
                            if link not in seen and page_allowed(link, allowed_hosts):
                                pending.append(link)
                except Exception as exc:
                    errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
                done = len(documents) + len(errors)
                if done % 100 == 0:
                    print(f"  processed={done}, queued={len(pending)}, errors={len(errors)}")

    pdf_records: dict[str, dict] = {}
    if args.include_pdfs and pdf_urls:
        print(f"Downloading {len(pdf_urls)} discovered official PDFs to the ignored local cache...")
        with ThreadPoolExecutor(max_workers=min(args.workers, 8)) as executor:
            futures = {executor.submit(download_pdf, url, args.refresh): url for url in sorted(pdf_urls)}
            for future, url in list(futures.items()):
                try:
                    pdf_records[url] = future.result()
                except Exception as exc:
                    errors.append({"url": url, "error": f"PDF {type(exc).__name__}: {exc}"})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "profile": args.profile,
        "documents": dict(sorted(documents.items())),
        "pdfs": dict(sorted(pdf_records.items())),
        "discovered_pdf_urls": sorted(pdf_urls),
        "errors": errors,
    }
    atomic_write(MANIFEST_FILE, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    write_index(args.profile, documents, pdf_records, errors, config)

    source_count = sum(bool(record.get("source_path")) for record in documents.values())
    print(f"Completed: pages={len(documents)}, sources={source_count}, pdfs={len(pdf_records)}, errors={len(errors)}")
    print(f"Index: {INDEX_FILE}")
    return 0 if documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
