"""Retrieval evaluation: does the right page show up for each question?

Usage (from backend/):
    python eval/run_eval.py             # vector search only (fast)
    python eval/run_eval.py --rerank    # also test Gemma reranking (slower)

Page-level metric: a hit means a chunk from the expected page was retrieved.
It does not check that the exact right chunk (e.g. the build/run commands) came back.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rag"))
from retriever import search, _col  # noqa: E402

POOL = 30


def first_rank(hits, expected):
    for i, h in enumerate(hits, 1):
        url = h["metadata"]["url"].lower()
        if any(e.lower() in url for e in expected):
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    if args.rerank:
        from pipeline import rerank

    questions = json.load(open(HERE / "questions.json", encoding="utf-8"))
    indexed = {m["url"].lower() for m in _col.get(include=["metadatas"])["metadatas"]}

    rows = []
    for item in questions:
        q, exp = item["q"], item["expect"]
        if not any(e.lower() in u for e in exp for u in indexed):
            print(f"[check] no indexed page matches {exp} -> skipped: {q}")
            continue
        hits = search(q, k=POOL)
        r = first_rank(hits, exp)
        rr = first_rank(rerank(q, hits), exp) if args.rerank else None
        rows.append((r, rr))
        mark = "OK  " if r and r <= 5 else ("LOW " if r else "MISS")
        line = f"{mark} vector rank {str(r or '>30'):>4}"
        if args.rerank:
            line += f" | in reranked top 10: {'yes' if rr else 'NO '}"
        print(f"{line} | {q}")

    n = len(rows)
    if not n:
        print("No evaluable questions.")
        return
    h = lambda k: sum(1 for r, _ in rows if r and r <= k)
    mrr = sum(1 / r for r, _ in rows if r) / n
    print(f"\n{n} questions evaluated")
    print(f"vector hit@1 {h(1)}/{n}   hit@5 {h(5)}/{n}   hit@10 {h(10)}/{n}   hit@30 {h(30)}/{n}   MRR {mrr:.2f}")
    if args.rerank:
        kept = sum(1 for _, rr in rows if rr)
        print(f"after rerank: expected page in the 10 chunks sent to Gemma: {kept}/{n}")


if __name__ == "__main__":
    main()
