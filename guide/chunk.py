"""Stage 2 — Chunking.

Splits each ingested document into overlapping ~800-character chunks. Splitting
prefers natural boundaries (paragraph breaks, then sentence/line breaks) so a
single Reddit comment or RMP review usually stays intact instead of being cut
mid-thought. Every chunk carries its parent document's metadata plus a stable id.
"""
from __future__ import annotations

import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE

# Boundaries to break on, most-preferred first.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Pull the end back to the nearest natural boundary inside the window.
            window = text[start:end]
            cut = -1
            for sep in _SEPARATORS:
                idx = window.rfind(sep)
                # Only accept a boundary in the back half so chunks don't get tiny.
                if idx != -1 and idx > size // 2:
                    cut = idx + len(sep)
                    break
            if cut != -1:
                end = start + cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _context_header(meta: dict) -> str:
    """A short header (document title + any course/professor) prepended to every
    chunk before embedding. This is "contextual chunk" retrieval: it anchors each
    chunk to its parent topic so a query like "easiest CS electives" matches a
    thread titled exactly that, even when the body only names specific courses."""
    parts = [meta.get("title", "")]
    if meta.get("courses"):
        parts.append(f"Courses: {meta['courses']}")
    header = " — ".join(p for p in parts if p)
    return header.strip()


def chunk_documents(
    documents: list[dict],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Turn documents into chunks: {id, text, metadata}.

    `text` is the raw chunk body (used for citations/display). `embed_text` is the
    body prefixed with a context header (used for embedding/retrieval).
    """
    chunks: list[dict] = []
    for doc_idx, doc in enumerate(documents):
        pieces = _split_text(doc["text"], size, overlap)
        base = doc["metadata"].get("professor") or doc["metadata"]["file"]
        slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
        slug = f"{doc_idx:02d}-{slug}"
        header = _context_header(doc["metadata"])
        for i, piece in enumerate(pieces):
            meta = dict(doc["metadata"])
            meta["chunk_index"] = i
            embed_text = f"{header}\n\n{piece}" if header else piece
            chunks.append({
                "id": f"{slug}-{i}",
                "text": piece,
                "embed_text": embed_text,
                "metadata": meta,
            })
    return chunks


if __name__ == "__main__":
    from .ingest import load_documents

    docs = load_documents()
    cs = chunk_documents(docs)
    print(f"{len(docs)} documents -> {len(cs)} chunks")
    sizes = [len(c["text"]) for c in cs]
    print(f"chunk size: min={min(sizes)} max={max(sizes)} avg={sum(sizes)//len(sizes)}")
