"""Embed chunks with nomic-embed-text and store them in ChromaDB.

Usage (from backend/):
    python ingestion/ingest.py
Safe to re-run: chunk IDs are stable, so existing chunks are overwritten.
"""
import sys
import time
from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunker import chunk_document, load_docs  # noqa: E402

VECTOR_DIR = Path(__file__).resolve().parents[2] / "vectordb" / "robomind"
COLLECTION = "ros2_docs"
EMBED_MODEL = "nomic-embed-text"
BATCH = 32


def main() -> None:
    docs = load_docs()
    chunks = [c for d in docs for c in chunk_document(d)]
    print(f"{len(docs)} pages -> {len(chunks)} chunks")

    emb = OllamaEmbeddings(model=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    col = client.get_or_create_collection(
        COLLECTION, configuration={"hnsw": {"space": "cosine"}}
    )

    start = time.time()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vectors = emb.embed_documents(["search_document: " + c["text"] for c in batch])
        col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors,
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  {min(i + BATCH, len(chunks))}/{len(chunks)}", end="\r", flush=True)

    print(f"\nDone in {time.time() - start:.0f}s. Collection '{COLLECTION}' now has {col.count()} chunks.")
    print(f"Stored at: {VECTOR_DIR}")


if __name__ == "__main__":
    main()
