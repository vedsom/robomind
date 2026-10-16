"""Chunk-level retrieval check: is the specific chunk that holds the answer retrieved?

Usage (from backend/):
    python eval/run_chunk_eval.py             # vector search only
    python eval/run_chunk_eval.py --rerank    # also: is it in the 10 chunks sent to Gemma?
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rag"))
from retriever import search, _col  # noqa: E402

POOL = 30


def matches(meta, case):
    return case["url"].lower() in meta["url"].lower() and case["heading"].lower() in meta["heading"].lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    if args.rerank:
        from pipeline import rerank

    cases = json.load(open(HERE / "chunk_cases.json", encoding="utf-8"))
    metas = _col.get(include=["metadatas"])["metadatas"]
    done = found_vec = found_rr = 0
    cache = {}

    for c in cases:
        if not any(matches(m, c) for m in metas):
            print(f"[check] no indexed chunk for page '{c['url']}' + heading '{c['heading']}' -> skipped")
            continue
        if c["q"] not in cache:
            hits = search(c["q"], k=POOL)
            cache[c["q"]] = (hits, rerank(c["q"], hits) if args.rerank else None)
        hits, top = cache[c["q"]]
        rank = next((i for i, h in enumerate(hits, 1) if matches(h["metadata"], c)), None)
        done += 1
        found_vec += bool(rank and rank <= 10)
        line = f"vector rank {str(rank or '>30'):>4}"
        if args.rerank:
            ok = any(matches(h["metadata"], c) for h in top)
            found_rr += ok
            line += f" | sent to Gemma: {'yes' if ok else 'NO '}"
        print(f"{line} | {c['heading']}  <- {c['q']}")

    print(f"\n{done} chunk cases evaluated")
    print(f"in vector top 10: {found_vec}/{done}")
    if args.rerank:
        print(f"in the 10 chunks sent to Gemma: {found_rr}/{done}")


if __name__ == "__main__":
    main()
