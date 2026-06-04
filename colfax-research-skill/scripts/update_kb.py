#!/usr/bin/env python3
"""
Colfax Research Knowledge Base Incremental Updater

Usage:
    python3 update_kb.py [--output-dir /path/to/kb]

This script scrapes the Colfax Research website and incrementally updates
the local knowledge base with new or modified articles.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://research.colfax-intl.com"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "colfax_knowledge_base")


def get_soup(url, retries=3):
    """Get BeautifulSoup object from URL with retries."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            if i < retries - 1:
                time.sleep(2 ** i)
                continue
            print(f"Error fetching {url}: {e}")
            return None


def get_article_links_from_page(page_num):
    """Get all article links from a listing page."""
    if page_num == 1:
        url = BASE_URL
    else:
        url = f"{BASE_URL}/?query-33-page={page_num}"

    soup = get_soup(url)
    if not soup:
        return []

    articles = []
    seen = set()
    content_area = soup.find("main") or soup.find(
        "div", class_=re.compile("content|site-content")
    )

    if content_area:
        for link in content_area.find_all("a", href=True):
            href = link.get("href", "")
            if (
                href.startswith("https://research.colfax-intl.com/")
                and "/category/" not in href
                and "/page/" not in href
                and "?" not in href
                and href != BASE_URL + "/"
                and href
                not in [
                    BASE_URL + "/research/",
                    BASE_URL + "/blog/",
                    BASE_URL + "/videos/",
                    BASE_URL + "/mission-statement/",
                    BASE_URL + "/about/",
                    BASE_URL + "/contact/",
                    BASE_URL + "/frontpage-slider/",
                    BASE_URL + "/videos/feed/",
                ]
            ):
                path = urlparse(href).path.strip("/")
                if path and path not in [
                    "research",
                    "blog",
                    "videos",
                    "mission-statement",
                    "about",
                    "contact",
                    "frontpage-slider",
                    "videos/feed",
                    "archived-content",
                    "feed",
                ]:
                    if href not in seen:
                        seen.add(href)
                        articles.append(href)

    return articles


def extract_article_info(article_url):
    """Extract all information from an article page."""
    soup = get_soup(article_url)
    if not soup:
        return None

    info = {
        "url": article_url,
        "title": "",
        "date": "",
        "date_iso": "",
        "categories": [],
        "tags": [],
        "excerpt": "",
        "content": "",
        "content_html": "",
        "pdf_url": "",
        "pdf_filename": "",
        "content_type": "blog",
        "scraped_at": datetime.now().isoformat(),
    }

    title_tag = soup.find("h1", class_="entry-title") or soup.find("h1")
    if title_tag:
        info["title"] = title_tag.get_text(strip=True)

    date_tag = soup.find("time", class_="entry-date") or soup.find("time")
    if date_tag:
        info["date"] = date_tag.get_text(strip=True)
        datetime_attr = date_tag.get("datetime", "")
        if datetime_attr:
            info["date_iso"] = datetime_attr

    cat_links = soup.find("span", class_="cat-links")
    if cat_links:
        cats = [a.get_text(strip=True) for a in cat_links.find_all("a")]
        info["categories"] = cats

    tags_links = soup.find("span", class_="tags-links")
    if tags_links:
        tags = [a.get_text(strip=True) for a in tags_links.find_all("a")]
        info["tags"] = tags

    content_div = soup.find("div", class_="entry-content")

    if content_div:
        for link in content_div.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if "/download/" in href:
                if href.startswith("/"):
                    href = urljoin(BASE_URL, href)
                info["pdf_url"] = href

                url_filename = os.path.basename(urlparse(href).path.strip("/"))
                if ".pdf" in text.lower():
                    info["pdf_filename"] = text
                elif ".pdf" in url_filename.lower():
                    info["pdf_filename"] = url_filename
                else:
                    clean_name = re.sub(r"[^\w\-]", "_", url_filename).lower()
                    if clean_name:
                        info["pdf_filename"] = f"{clean_name}.pdf"
                    else:
                        title_slug = re.sub(r"[^\w\-]", "_", info["title"][:50]).lower()
                        info["pdf_filename"] = f"{title_slug}.pdf"

                info["content_type"] = "pdf"
                break

    if content_div:
        content_clone = BeautifulSoup(str(content_div), "html.parser")
        for script in content_clone.find_all(["script", "style"]):
            script.decompose()
        info["content"] = content_clone.get_text(separator="\n", strip=True)
        info["content_html"] = str(content_div)

    excerpt_meta = soup.find("meta", attrs={"name": "description"})
    if excerpt_meta:
        info["excerpt"] = excerpt_meta.get("content", "")

    if not info["excerpt"] and content_div:
        first_p = content_div.find("p")
        if first_p:
            info["excerpt"] = first_p.get_text(strip=True)[:500]

    return info


def download_pdf(pdf_url, output_path):
    """Download a PDF file."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Error downloading PDF: {e}")
        return False


def save_article_markdown(info, output_path):
    """Save article info as markdown file."""
    md_content = f"""# {info['title']}

**URL:** {info['url']}
**Date:** {info['date']}
**ISO Date:** {info.get('date_iso', '')}
**Categories:** {', '.join(info['categories'])}
**Tags:** {', '.join(info['tags'])}
**Content Type:** {info['content_type']}
**PDF URL:** {info['pdf_url']}
**PDF Filename:** {info['pdf_filename']}
**Scraped At:** {info['scraped_at']}

---

## Excerpt

{info['excerpt']}

---

## Content

{info['content']}
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    return True


def load_metadata(output_dir):
    """Load existing metadata if available."""
    metadata_file = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"articles": {}, "last_full_update": None, "total_articles": 0}


def save_metadata(metadata, output_dir):
    """Save metadata to file."""
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally update Colfax Research knowledge base"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help="Output directory for knowledge base",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    articles_dir = os.path.join(output_dir, "articles")
    pdfs_dir = os.path.join(output_dir, "pdfs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(articles_dir, exist_ok=True)
    os.makedirs(pdfs_dir, exist_ok=True)

    metadata = load_metadata(output_dir)
    articles_data = metadata.get("articles", {})

    print(f"Starting incremental update...")
    print(f"Current knowledge base: {len(articles_data)} articles")
    print("=" * 60)

    # Get all article URLs from website
    all_urls = []
    for page in range(1, 6):
        urls = get_article_links_from_page(page)
        all_urls.extend(urls)
        time.sleep(0.5)

    seen = set()
    unique_urls = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    print(f"\nFound {len(unique_urls)} articles on website")

    new_count = 0
    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, url in enumerate(unique_urls, 1):
        print(f"\n[{idx}/{len(unique_urls)}] {url}")

        existing = articles_data.get(url, {})
        if existing and existing.get("scraped_at"):
            print(f"  Already exists, skipping")
            skipped_count += 1
            continue

        info = extract_article_info(url)
        if not info:
            print(f"  Failed to extract")
            failed_count += 1
            continue

        print(f"  {info['title'][:70]}")
        print(f"  Date: {info['date']}")

        url_path = urlparse(url).path.strip("/")
        filename_base = url_path.replace("/", "_")
        md_path = os.path.join(articles_dir, f"{filename_base}.md")
        save_article_markdown(info, md_path)

        if info["pdf_url"]:
            pdf_path = os.path.join(pdfs_dir, info["pdf_filename"])
            if not os.path.exists(pdf_path):
                print(f"  Downloading PDF: {info['pdf_filename']}")
                success = download_pdf(info["pdf_url"], pdf_path)
                if success:
                    print(f"  PDF downloaded")
                    info["pdf_local_path"] = pdf_path
                else:
                    print(f"  PDF download failed")
                    info["pdf_local_path"] = ""
            else:
                print(f"  PDF already exists")
                info["pdf_local_path"] = pdf_path

        metadata_entry = {
            "url": info["url"],
            "title": info["title"],
            "date": info["date"],
            "date_iso": info.get("date_iso", ""),
            "categories": info["categories"],
            "tags": info["tags"],
            "excerpt": info["excerpt"],
            "pdf_url": info["pdf_url"],
            "pdf_filename": info["pdf_filename"],
            "pdf_local_path": info.get("pdf_local_path", ""),
            "content_type": info["content_type"],
            "scraped_at": info["scraped_at"],
            "local_markdown": md_path,
        }

        articles_data[url] = metadata_entry

        if existing:
            updated_count += 1
        else:
            new_count += 1

        time.sleep(1)

    metadata["articles"] = articles_data
    metadata["last_full_update"] = datetime.now().isoformat()
    metadata["total_articles"] = len(articles_data)
    save_metadata(metadata, output_dir)

    print("\n" + "=" * 60)
    print("UPDATE SUMMARY:")
    print(f"  New articles: {new_count}")
    print(f"  Updated articles: {updated_count}")
    print(f"  Skipped (up to date): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total in knowledge base: {len(articles_data)}")
    print(f"\nKnowledge base: {output_dir}")


if __name__ == "__main__":
    main()
