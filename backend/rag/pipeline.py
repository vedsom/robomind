"""Question -> retrieve -> Gemma 4 answer with source citations.

Usage (from backend/):
    python rag/pipeline.py "How do I write a publisher in Python?"
"""
import sys

from langchain_ollama import ChatOllama

from retriever import search

LLM_MODEL = "gemma4:31b"

SYSTEM = """You are RoboMind, an assistant for ROS 2 engineers.
Answer the question using ONLY the numbered context snippets below.
- Cite the snippets you use like [1] or [2][3] right after the claim.
- Include exact commands and code from the context when relevant, in fenced code blocks.
- If the context does not contain the answer, say so plainly instead of guessing.
- Be concise and practical."""


def build_prompt(question: str, hits: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i}] ({h['metadata']['heading']})\n{h['text']}" for i, h in enumerate(hits, 1))
    return f"{SYSTEM}\n\n=== CONTEXT ===\n{ctx}\n\n=== QUESTION ===\n{question}"


def answer(question: str, k: int = 5, version: str | None = "humble") -> dict:
    hits = search(question, k=k, version=version)
    llm = ChatOllama(model=LLM_MODEL, num_ctx=8192, temperature=0)
    reply = llm.invoke(build_prompt(question, hits))
    return {"answer": reply.content, "sources": hits}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How do I write a publisher in Python?"
    out = answer(q)
    print(out["answer"])
    print("\nSources:")
    for i, h in enumerate(out["sources"], 1):
        print(f"  [{i}] {h['metadata']['heading']}\n      {h['metadata']['url']}")
