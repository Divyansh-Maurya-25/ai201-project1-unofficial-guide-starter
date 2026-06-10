"""Stage 3 — Embedding + vector store.

Embeds chunks with all-MiniLM-L6-v2 and persists them in a local ChromaDB
collection (cosine space).

Embedding backend: we prefer `sentence-transformers` (the model named in the
project spec). If it isn't installed, we fall back to ChromaDB's built-in
DefaultEmbeddingFunction, which is the *same* all-MiniLM-L6-v2 weights served
via ONNX — identical vectors, no PyTorch dependency. Either way the embedding
model is all-MiniLM-L6-v2.
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def get_embedding_function():
    """Return a Chroma-compatible all-MiniLM-L6-v2 embedding function."""
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    except Exception:  # sentence-transformers / torch not available
        # ONNX all-MiniLM-L6-v2 — same model, lightweight.
        return embedding_functions.DefaultEmbeddingFunction()


def get_client() -> "chromadb.api.ClientAPI":
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_store(chunks: list[dict], reset: bool = True):
    """Embed and store all chunks. Returns the populated collection."""
    client = get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    # Add in batches to keep memory flat. We embed `embed_text` (chunk body with a
    # context header) but stash the clean body in metadata as `display_text` so
    # retrieval can show the original text without the header noise.
    batch = 128
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        metadatas = []
        for c in part:
            meta = dict(c["metadata"])
            meta["display_text"] = c["text"]
            metadatas.append(meta)
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c.get("embed_text", c["text"]) for c in part],
            metadatas=metadatas,
        )
    return collection


def get_collection():
    """Open the existing collection for querying."""
    client = get_client()
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )
