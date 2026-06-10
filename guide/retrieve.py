"""Stage 4 — Retrieval.

Embeds the user's question with the same model used at index time and pulls the
top-k nearest chunks from ChromaDB. Low-relevance chunks (cosine distance above
MAX_DISTANCE) are dropped so the generator only ever sees on-topic context.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import MAX_DISTANCE, TOP_K
from .embed_store import get_collection


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float

    @property
    def citation(self) -> str:
        m = self.metadata
        title = m.get("title", m.get("file", "source"))
        url = m.get("url", "")
        return f"{title}{f' ({url})' if url else ''}"


def retrieve(query: str, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE) -> list[RetrievedChunk]:
    collection = get_collection()
    res = collection.query(query_texts=[query], n_results=top_k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    results: list[RetrievedChunk] = []
    for text, meta, dist in zip(docs, metas, dists):
        if dist is not None and dist > max_distance:
            continue
        # Prefer the clean body stored at index time; fall back to the embedded doc.
        display = (meta or {}).get("display_text", text)
        results.append(RetrievedChunk(text=display, metadata=meta, distance=dist))
    return results


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "how hard is COP 3514"
    for i, c in enumerate(retrieve(q), 1):
        print(f"[{i}] dist={c.distance:.3f}  {c.citation}")
        print(c.text[:200], "...\n")
