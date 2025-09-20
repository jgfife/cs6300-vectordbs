#!/usr/bin/env python3
"""
Chroma demo: persistent collection, upsert (insert + update), query, and filters.

Run:
  python chroma_demo.py
"""

from __future__ import annotations
import shutil
from pathlib import Path
import pprint
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --------- Config ----------
DB_DIR = Path("./chroma_demo_db")
COLLECTION_NAME = "vector_db_demo"
EMBEDDER = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
pp = pprint.PrettyPrinter(indent=2, width=100)

# --------- Helpers ----------
def fresh_start():
    """Delete the demo DB so runs are reproducible in class."""
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

def get_client():
    """
    Use the modern PersistentClient if available (Chroma >= 0.5),
    otherwise fall back to the older Settings-based client.
    """
    try:
        # Chroma >= 0.5
        return chromadb.PersistentClient(path=str(DB_DIR))
    except AttributeError:
        # Older Chroma
        from chromadb.config import Settings
        return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(DB_DIR)))

def show(title: str, obj):
    print(f"\n=== {title} ===")
    pp.pprint(obj)

# --------- Demo Data ----------
docs_insert = [
    {
        "id": "doc-001",
        "text": "HNSW is a popular approximate nearest neighbor (ANN) graph index used in vector databases.",
        "meta": {"topic": "indexes", "source": "manual", "level": "intro"}
    },
    {
        "id": "doc-002",
        "text": "IVF-Flat partitions embeddings into coarse clusters, then scans a subset for exact distances.",
        "meta": {"topic": "indexes", "source": "manual", "level": "intro"}
    },
    {
        "id": "doc-003",
        "text": "PQ (Product Quantization) compresses vectors into codebooks to trade accuracy for memory/speed.",
        "meta": {"topic": "compression", "source": "manual", "level": "intermediate"}
    },
]

# This shows "upsert as UPDATE": same id with changed text & metadata.
doc_update = {
    "id": "doc-002",
    "text": "IVF (Inverted File) uses coarse quantization; IVF-Flat stores raw vectors in the cells it probes.",
    "meta": {"topic": "indexes", "source": "revised", "level": "intro"}
}

# --------- Main Demo ----------
def main():
    # Start clean for a predictable classroom run
    fresh_start()
    client = get_client()

    # Create or get a collection with an embedding function
    # Note: you can add HNSW params via metadata if desired (implementation-dependent)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDER,
        metadata={"hnsw:space": "cosine"}
    )

    # --- UPSERT (INSERT mode): new IDs create new records ---
    collection.upsert(
        ids=[d["id"] for d in docs_insert],
        documents=[d["text"] for d in docs_insert],
        metadatas=[d["meta"] for d in docs_insert],
    )
    show("Count after INSERT upsert", collection.count())

    # --- Query: basic semantic search ---
    res1 = collection.query(
        query_texts=["Which method compresses vectors with codebooks?"],
        n_results=2,
    )
    show("Query 1 (top 2)", res1)

    # --- Filter by metadata ('where'): only items with topic == 'indexes' ---
    res2 = collection.query(
        query_texts=["tell me about approximate nearest neighbor structures"],
        n_results=5,
        where={"topic": "indexes"},  # exact match
    )
    show("Query 2 with metadata filter where={'topic': 'indexes'}", res2)

    # --- Filter by document content ('where_document'): substring contains 'HNSW' ---
    # Supported operators include $contains / $not_contains (on the raw document text)
    res3 = collection.query(
        query_texts=["graph-based ANN"],
        n_results=5,
        where_document={"$contains": "HNSW"}
    )
    show("Query 3 with where_document={'$contains': 'HNSW'}", res3)

    # --- UPSERT (UPDATE mode): existing ID is replaced/updated ---
    collection.upsert(
        ids=[doc_update["id"]],
        documents=[doc_update["text"]],
        metadatas=[doc_update["meta"]],
    )

    # Verify the update took effect: filter by the new metadata value 'source': 'revised'
    res4 = collection.query(
        query_texts=["explain IVF-Flat"],
        n_results=3,
        where={"source": "revised"}
    )
    show("Query 4 after UPDATE upsert (where={'source': 'revised'})", res4)

    print("\nDone. DB persisted at:", DB_DIR.resolve())

if __name__ == "__main__":
    main()
