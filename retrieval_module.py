"""
retrieval_module.py
FreshMart RAG Retrieval Module

Loads the ChromaDB collection built by rag_indexer.py and provides
a simple retrieve() function used by conversation_manager.py.

Features:
- Caches the ChromaDB client and collection at module level (loaded once)
- Caches query results for repeated identical queries (fast)
- Returns top-k most relevant chunks with source attribution
- Gracefully returns empty list if index not built yet
"""

import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR  = "chroma_db"
COLLECTION  = "freshmart_docs"
MODEL_NAME  = "all-MiniLM-L6-v2"
TOP_K       = 3          # Number of chunks to retrieve per query
MAX_CHUNK_CHARS = 500    # Truncate very long chunks in the prompt

# ── Module-level singletons (loaded once, reused) ─────────────────────────────
_collection = None
_query_cache: dict[str, list[dict]] = {}


def _get_collection():
    """Lazy-load ChromaDB collection. Returns None if index not built."""
    global _collection
    if _collection is not None:
        return _collection

    if not os.path.exists(CHROMA_DIR):
        return None

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
        _collection = client.get_collection(
            name=COLLECTION,
            embedding_function=ef
        )
        return _collection
    except Exception as e:
        print(f"[RAG] Failed to load collection: {e}")
        return None


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve the top-k most relevant document chunks for a query.

    Returns a list of dicts:
        [{"text": "...", "source": "filename.txt", "score": 0.85}, ...]

    Returns empty list if index is not available or query fails.
    """
    if not query or not query.strip():
        return []

    # Check cache (avoid re-embedding identical queries)
    cache_key = hashlib.md5(f"{query.strip().lower()}:{top_k}".encode()).hexdigest()
    if cache_key in _query_cache:
        return _query_cache[cache_key]

    collection = _get_collection()
    if collection is None:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert cosine distance to similarity score (0-1)
            score = round(1 - dist, 3)
            # Only include reasonably relevant chunks
            if score < 0.2:
                continue
            text = doc[:MAX_CHUNK_CHARS] + "..." if len(doc) > MAX_CHUNK_CHARS else doc
            chunks.append({
                "text":   text,
                "source": meta.get("source", "unknown"),
                "score":  score
            })

        # Cache the result
        _query_cache[cache_key] = chunks
        return chunks

    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return []


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a string to inject into the LLM prompt.
    Returns empty string if no chunks.
    """
    if not chunks:
        return ""

    lines = ["[RETRIEVED KNOWLEDGE BASE CONTEXT]",
             "The following information was retrieved from FreshMart documents:",
             ""]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Source {i}: {chunk['source']}]")
        lines.append(chunk["text"])
        lines.append("")

    lines.append("[END OF RETRIEVED CONTEXT]")
    return "\n".join(lines)


def is_index_ready() -> bool:
    """Check if the ChromaDB index has been built."""
    return _get_collection() is not None


if __name__ == "__main__":
    # Quick test
    print("Testing retrieval module...")
    if not is_index_ready():
        print("Index not built. Run: python rag_indexer.py")
    else:
        test_queries = [
            "What is the delivery fee?",
            "How do I return a damaged product?",
            "What fruits are available?",
        ]
        for q in test_queries:
            print(f"\nQuery: {q}")
            chunks = retrieve(q)
            for c in chunks:
                print(f"  [{c['score']:.2f}] {c['source']}: {c['text'][:100]}...")
