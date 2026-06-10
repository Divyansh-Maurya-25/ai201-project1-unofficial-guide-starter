"""Stage 5 — Grounded generation.

Builds a prompt from the retrieved chunks and asks an LLM to answer using ONLY
that context, citing sources by number. Grounding is enforced two ways:

  1. A strict system prompt that forbids outside knowledge and requires the
     model to say it doesn't know when the context is insufficient.
  2. Structure: each chunk is injected as a numbered [Source N] block and the
     model is told to cite the numbers it used, so every claim is traceable.

Backend: Groq if GROQ_API_KEY is set (the spec's choice), otherwise a local
Ollama model so the system runs offline.
"""
from __future__ import annotations

import os

from .config import (
    GROQ_MODEL,
    MAX_TOKENS,
    OLLAMA_MODEL,
    TEMPERATURE,
)
from .retrieve import RetrievedChunk, retrieve

SYSTEM_PROMPT = (
    "You are The Unofficial Guide to USF Computer Science. You answer questions "
    "for USF CS students using ONLY the numbered sources provided below.\n\n"
    "Rules:\n"
    "1. Use ONLY information found in the sources. Do not use outside knowledge.\n"
    "2. If the sources do not contain enough information to answer, say so plainly: "
    "\"The sources I have don't cover that.\" Do not guess or invent.\n"
    "3. These are student opinions from Reddit, Rate My Professors, and study guides. "
    "Attribute claims accordingly (e.g. \"students say...\") and note when sources disagree.\n"
    "4. Cite the sources you used by their number, like [Source 2].\n"
    "5. Be concise and concrete."
)


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[Source {i}] {c.citation}\n{c.text}")
    return "\n\n".join(blocks)


def _generate_groq(system: str, user: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def _generate_ollama(system: str, user: str) -> str:
    # Use the HTTP API directly to avoid extra dependencies.
    import json
    import urllib.request

    payload = {
        "model": OLLAMA_MODEL,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read()).get("response", "").strip()


def generate(query: str, chunks: list[RetrievedChunk] | None = None) -> dict:
    """Answer a query. Returns {answer, sources, backend, used_chunks}."""
    if chunks is None:
        chunks = retrieve(query)

    if not chunks:
        return {
            "answer": "The sources I have don't cover that.",
            "sources": [],
            "backend": "none",
            "used_chunks": [],
        }

    context = build_context(chunks)
    user_prompt = (
        f"Sources:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the sources above, and cite source numbers."
    )

    if os.environ.get("GROQ_API_KEY"):
        answer, backend = _generate_groq(SYSTEM_PROMPT, user_prompt), "groq"
    else:
        answer, backend = _generate_ollama(SYSTEM_PROMPT, user_prompt), "ollama"

    sources = [
        {"n": i, "citation": c.citation, "distance": round(c.distance, 3)}
        for i, c in enumerate(chunks, 1)
    ]
    return {"answer": answer, "sources": sources, "backend": backend, "used_chunks": chunks}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "how hard is COP 3514"
    out = generate(q)
    print(out["answer"])
    print("\nSources:")
    for s in out["sources"]:
        print(f"  [{s['n']}] {s['citation']}  (dist={s['distance']})")
