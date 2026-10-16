"""Split scraped ROS2 pages into heading-aware, code-safe chunks.

Usage (from backend/):
    python ingestion/chunker.py          # prints stats + a sample chunk
"""
import glob
import hashlib
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ros2_humble"

# ~4 characters per token, so 2000 chars ~ 512 tokens and 200 chars ~ 50 tokens
CHUNK_CHARS = 2000
OVERLAP_CHARS = 200
MIN_CHARS = 80   # drop heading-only or near-empty fragments

BLANK = "\n\u2424\n"                      # stands in for blank lines inside code
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_HEADING = re.compile(r"^(#{1,4}) +(.*)")
_by_size = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_CHARS, chunk_overlap=OVERLAP_CHARS, strip_whitespace=False
)


def split_sections(text: str) -> list[tuple[dict, str]]:
    """Split on # headings, ignoring '#' lines inside code fences."""
    sections, path, buf, in_code = [], {}, [], False

    def flush():
        body = "\n".join(buf).strip("\n")
        if body.strip():
            sections.append((dict(path), body))

    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            flush()
            buf.clear()
            level = len(m.group(1))
            path = {k: v for k, v in path.items() if k < level}
            path[level] = m.group(2).strip()
        buf.append(line)
    flush()
    return sections


def size_split(body: str) -> list[str]:
    """Split long sections by size without cutting code blocks at blank lines."""
    protected = _FENCE.sub(lambda m: m.group(0).replace("\n\n", BLANK), body)
    pieces = [p.replace(BLANK, "\n\n").strip("\n") for p in _by_size.split_text(protected)]
    fixed, carry = [], False
    for p in pieces:                     # if a huge code block must be cut, close/reopen the fence
        if carry:
            p = "```\n" + p
        if p.count("```") % 2:
            p += "\n```"
            carry = True
        else:
            carry = False
        fixed.append(p)
    return fixed


def chunk_document(doc: dict) -> list[dict]:
    chunks, n = [], 0
    for path, body in split_sections(doc["text"]):
        crumbs = [doc["title"]] + [path[k] for k in sorted(path) if k >= 2]
        heading = " > ".join(crumbs)
        for piece in size_split(body):
            if len(piece) < MIN_CHARS:
                continue
            chunks.append({
                # stable ID: re-ingesting the same page overwrites instead of duplicating
                "id": hashlib.sha1(f'{doc["url"]}#{n}'.encode()).hexdigest()[:16],
                "text": f"{heading}\n\n{piece}",
                "metadata": {
                    "url": doc["url"],
                    "title": doc["title"],
                    "section": doc["section"],
                    "version": doc["version"],
                    "source": doc["source"],
                    "heading": heading,
                    "chunk_index": n,
                },
            })
            n += 1
    return chunks


def load_docs() -> list[dict]:
    return [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(DATA_DIR / "*.json")))]


if __name__ == "__main__":
    docs = load_docs()
    chunks = [c for d in docs for c in chunk_document(d)]
    sizes = [len(c["text"]) for c in chunks]
    broken_code = sum(1 for c in chunks if c["text"].count("```") % 2)
    print(f"pages: {len(docs)}  chunks: {len(chunks)}")
    print(f"chars per chunk: avg {sum(sizes)//len(sizes)}, max {max(sizes)}, min {min(sizes)}")
    print(f"chunks with an unbalanced code fence: {broken_code}")
    code = [c for c in chunks if "```" in c["text"]]
    c = code[len(code) // 2]
    print("\n--- sample chunk with code ---")
    print(c["text"])
    print(c["metadata"])
