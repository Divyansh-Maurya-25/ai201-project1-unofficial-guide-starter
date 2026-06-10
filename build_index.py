"""Build the vector store from documents/.

Run this once (and again whenever documents change):

    python build_index.py
"""
from guide.chunk import chunk_documents
from guide.embed_store import build_store
from guide.ingest import load_documents, summarize


def main() -> None:
    print("Stage 1/3 — Ingesting documents...")
    documents = load_documents()
    summarize(documents)

    print("\nStage 2/3 — Chunking...")
    chunks = chunk_documents(documents)
    sizes = [len(c["text"]) for c in chunks]
    print(f"  {len(documents)} documents -> {len(chunks)} chunks")
    print(f"  chunk chars: min={min(sizes)} max={max(sizes)} avg={sum(sizes)//len(sizes)}")

    print("\nStage 3/3 — Embedding + storing in ChromaDB...")
    collection = build_store(chunks)
    print(f"  Stored {collection.count()} chunks in collection '{collection.name}'.")
    print("\nDone. Query with: python -m guide.generate \"how hard is COP 3514\"")


if __name__ == "__main__":
    main()
