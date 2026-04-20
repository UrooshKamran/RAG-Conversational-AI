"""
rag_indexer.py
FreshMart RAG Pipeline - Offline Indexing Script

Run this ONCE before starting the server (or after adding new documents):
    python rag_indexer.py

It reads all .txt files from the documents/ folder, chunks them,
embeds them using all-MiniLM-L6-v2, and stores them in ChromaDB.
"""

import os
import re
import time
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR    = "documents"
CHROMA_DIR  = "chroma_db"
COLLECTION  = "freshmart_docs"
CHUNK_SIZE  = 400   # tokens (approx words)
CHUNK_OVERLAP = 50

MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def load_documents(docs_dir: str) -> list[dict]:
    """Load all .txt files from the documents directory."""
    docs = []
    if not os.path.exists(docs_dir):
        print(f"[ERROR] Documents directory '{docs_dir}' not found.")
        return docs

    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(docs_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                docs.append({"filename": fname, "content": content})
        except Exception as e:
            print(f"[WARN] Could not read {fname}: {e}")

    return docs


def build_index():
    print("=" * 50)
    print("  FreshMart RAG Indexer")
    print("=" * 50)

    # Load documents
    print(f"\n[1/4] Loading documents from '{DOCS_DIR}'...")
    docs = load_documents(DOCS_DIR)
    print(f"      Loaded {len(docs)} documents.")

    # Chunk documents
    print(f"\n[2/4] Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks   = []
    all_ids      = []
    all_metadata = []

    for doc in docs:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['filename']}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({
                "source": doc["filename"],
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

    print(f"      Created {len(all_chunks)} chunks from {len(docs)} documents.")

    # Set up ChromaDB
    print(f"\n[3/4] Setting up ChromaDB at '{CHROMA_DIR}'...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists (fresh rebuild)
    try:
        client.delete_collection(COLLECTION)
        print("      Deleted existing collection for fresh rebuild.")
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Embed and store in batches
    print(f"\n[4/4] Embedding and storing chunks (model: {MODEL_NAME})...")
    print("      This may take a few minutes on first run (model download)...")

    BATCH_SIZE = 50
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks    = all_chunks[i:i + BATCH_SIZE]
        batch_ids       = all_ids[i:i + BATCH_SIZE]
        batch_meta      = all_metadata[i:i + BATCH_SIZE]

        collection.add(
            documents=batch_chunks,
            ids=batch_ids,
            metadatas=batch_meta
        )
        print(f"      Stored batch {i // BATCH_SIZE + 1}/{(len(all_chunks) - 1) // BATCH_SIZE + 1} "
              f"({min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks)")

    print("\n" + "=" * 50)
    print(f"  Indexing complete!")
    print(f"  {len(docs)} documents -> {len(all_chunks)} chunks -> ChromaDB")
    print(f"  Database saved to: {CHROMA_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    start = time.time()
    build_index()
    print(f"\n  Total time: {time.time() - start:.1f}s")
