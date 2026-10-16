"""Query the ChromaDB collection.

Usage (from backend/):
    python rag/retriever.py "How do I create a ROS2 workspace?"
"""
import sys
from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings

VECTOR_DIR = Path(__file__).resolve().parents[2] / "vectordb" / "robomind"
_emb = OllamaEmbeddings(model="nomic-embed-text")
_col = chromadb.PersistentClient(path=str(VECTOR_DIR)).get_collection("ros2_docs")


def search(query: str, k: int = 5, version: str | None = None) -> list[dict]:
    vec = _emb.embed_query("search_query: " + query)
    res = _col.query(
        query_embeddings=[vec],
        n_results=k,
        where={"version": version} if version else None,
    )
    return [
        {"text": t, "metadata": m, "distance": d}
        for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def expand(hits: list[dict], after: int = 1, max_total: int = 16) -> list[dict]:
    """Add the next `after` chunk(s) of the same page for each hit, so procedures stay complete."""
    have = {(h["metadata"]["url"], h["metadata"]["chunk_index"]) for h in hits}
    want = {}
    for h in hits:                      # hits arrive best-first, so the cap drops the weakest
        m = h["metadata"]
        for d in range(1, after + 1):
            key = (m["url"], m["chunk_index"] + d)
            if key not in have:
                want[key] = True
    out = list(hits)
    for url, idx in want:
        if len(out) >= max_total:
            break
        res = _col.get(
            where={"$and": [{"url": url}, {"chunk_index": idx}]},
            include=["documents", "metadatas"],
        )
        for t, m in zip(res["documents"], res["metadatas"]):
            out.append({"text": t, "metadata": m, "distance": None})
    return out


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How do I create a ROS2 workspace?"
    for r in search(q):
        print(f"{r['distance']:.3f}  {r['metadata']['heading']}")
        print(f"       {r['metadata']['url']}\n")
