"""Question -> retrieve 30 -> Gemma rerank -> answer with citations.

Usage (from backend/):
    python rag/pipeline.py "How do I create a ROS2 workspace?"
"""
import re
import sys

from langchain_ollama import ChatOllama

from retriever import search

LLM_MODEL = "gemma4:31b"
POOL = 30    # candidates fetched from ChromaDB
TOP_K = 10   # chunks sent to the answer step after reranking

SYSTEM = """You are RoboMind, an assistant for ROS 2 engineers.
Answer the question using ONLY the numbered context snippets below.
- Cite the snippets you use like [1] or [2][3] right after the claim.
- Include exact commands and code from the context when relevant, in fenced code blocks.
- Cover the full procedure (setup, code, build, run) when the context contains it.
- If the context does not contain the answer, say so plainly instead of guessing.
- Be concise and practical."""

RERANK_PROMPT = """You rank documentation snippets for a question.

Question: {q}

Snippets:
{snips}

Pick the {n} snippets most useful for answering the question. Snippets with concrete \
steps, commands or code that directly answer it rank ABOVE general background or summaries. \
Reply with ONLY a JSON list of snippet numbers, best first, for example [3, 1, 7]."""


def rerank(question: str, hits: list[dict], n: int = TOP_K) -> list[dict]:
    snips = "\n\n".join(f"[{i}] {h['text'][:600]}" for i, h in enumerate(hits, 1))
    llm = ChatOllama(model=LLM_MODEL, num_ctx=16384, temperature=0)
    try:
        raw = llm.invoke(RERANK_PROMPT.format(q=question, snips=snips, n=n)).content
        m = re.search(r"\[[\d,\s]*\]", raw)
        ids = [int(x) for x in re.findall(r"\d+", m.group(0))] if m else []
    except Exception as e:                      # fall back to plain vector order
        print(f"[rerank failed: {e}]", file=sys.stderr)
        ids = []
    order = []
    for i in ids:
        if 1 <= i <= len(hits) and i not in order:
            order.append(i)
    for i in range(1, len(hits) + 1):           # pad if the model returned fewer than n
        if len(order) >= n:
            break
        if i not in order:
            order.append(i)
    return [hits[i - 1] for i in order[:n]]


def build_prompt(question: str, hits: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i}] ({h['metadata']['heading']})\n{h['text']}" for i, h in enumerate(hits, 1))
    return f"{SYSTEM}\n\n=== CONTEXT ===\n{ctx}\n\n=== QUESTION ===\n{question}"


def answer(question: str, version: str | None = "humble") -> dict:
    candidates = search(question, k=POOL, version=version)
    top = rerank(question, candidates)
    # present chunks in reading order (page, then position) so steps stay sequential
    top.sort(key=lambda h: (h['metadata']['url'], h['metadata']['chunk_index']))
    llm = ChatOllama(model=LLM_MODEL, num_ctx=12288, temperature=0)
    reply = llm.invoke(build_prompt(question, top))
    return {"answer": reply.content, "sources": top}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How do I create a ROS2 workspace?"
    out = answer(q)
    print(out["answer"])
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", out["answer"])
                    if 1 <= int(n) <= len(out["sources"])})
    print("\nSources cited:" if cited else "\nNo sources cited.")
    for i in cited:
        h = out["sources"][i - 1]
        print(f"  [{i}] {h['metadata']['heading']}\n      {h['metadata']['url']}")
