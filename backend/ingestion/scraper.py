"""Scrape the ROS2 Humble Tutorials section into data/ros2_humble/*.json

Usage (from backend/):
    python ingestion/scraper.py --limit 5     # test run
    python ingestion/scraper.py               # full tutorials section
"""
import argparse
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://docs.ros.org/en/humble/"
START = BASE + "Tutorials.html"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "ros2_humble"
HEADERS = {"User-Agent": "RoboMind-learning-project (personal RAG study)"}
DELAY_SECONDS = 1.0
MIN_TEXT_CHARS = 200   # skip near-empty index pages (we still follow their links)

CONTENT_SELECTORS = ["div[role=main]", "article", "main", "div.body", "div.document"]


def in_scope(url: str) -> bool:
    """Stay inside the Tutorials section, HTML pages only."""
    return url == START or (url.startswith(BASE + "Tutorials/") and url.endswith(".html"))


def find_main(soup: BeautifulSoup):
    for sel in CONTENT_SELECTORS:
        node = soup.select_one(sel)
        if node:
            return node
    return None


def collect_links(main, page_url: str) -> list[str]:
    links = []
    for a in main.find_all("a", href=True):
        url, _ = urldefrag(urljoin(page_url, a["href"]))
        if in_scope(url):
            links.append(url)
    return links


def extract(main) -> tuple[str, str]:
    """Return (title, text) with code blocks fenced and headings marked."""
    for tag in main.select("script, style, nav, button, .copybtn, a.headerlink"):
        tag.decompose()

    h1 = main.find("h1")
    title = re.sub(r"\s+", " ", h1.get_text()).strip() if h1 else ""
    title = re.sub(r"[\ue000-\uf8ff]", "", title).strip()

    for pre in main.find_all("pre"):
        code = pre.get_text().strip("\n")
        pre.replace_with("\n\n```\n" + code + "\n```\n\n")

    for level in range(1, 6):
        for h in main.find_all(f"h{level}"):
            h.insert_before("\n\n" + "#" * level + " ")
            h.append("\n")

    for tag in main.find_all(["p", "li", "tr", "dt", "dd"]):
        tag.append("\n")

    text = main.get_text()
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    # drop the "older version of ROS 2" banner that Sphinx puts on every page
    text = re.sub(
        r"You.re reading the documentation for an older.*?please have a look at \w+\.\s*",
        "", text, flags=re.DOTALL)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


def slug_for(url: str) -> str:
    return url.removeprefix(BASE).removesuffix(".html").replace("/", "__")


def section_for(url: str) -> str:
    parts = url.removeprefix(BASE).removesuffix(".html").split("/")
    return parts[1] if len(parts) > 2 else "Tutorials"


def scrape(limit: int | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queue, seen = deque([START]), {START}
    fetched = saved = 0

    while queue and (limit is None or fetched < limit):
        url = queue.popleft()
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[skip] {url} -> {e}")
            continue
        fetched += 1

        soup = BeautifulSoup(resp.content, "html.parser")
        main = find_main(soup)
        if main is None:
            print(f"[warn] no main content found: {url}")
            continue

        for link in collect_links(main, url):    # collect BEFORE extract() edits the tree
            if link not in seen:
                seen.add(link)
                queue.append(link)

        title, text = extract(main)
        if len(text) < MIN_TEXT_CHARS:
            print(f"[index] {url} ({len(text)} chars, not saved)")
        else:
            record = {
                "url": url,
                "title": title,
                "section": section_for(url),
                "version": "humble",
                "source": "ros2-docs",
                "text": text,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
            (OUT_DIR / f"{slug_for(url)}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            saved += 1
            print(f"[saved] {title!r} ({len(text)} chars) <- {url}")

        time.sleep(DELAY_SECONDS)

    print(f"\nDone. Fetched {fetched} pages, saved {saved}, {len(queue)} still queued.")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max pages to fetch")
    scrape(ap.parse_args().limit)
