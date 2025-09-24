#!/usr/bin/env python3
"""
Chroma demo: persistent collection, upsert (insert + update), query, and filters.

Run:
  python chroma_demo.py
"""

from __future__ import annotations
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from db import load_data_to_chroma

def main():
    dbDir = Path("./db/chroma")
    client = chromadb.PersistentClient(path=str(dbDir))

    # Create or get a collection with an embedding function
    # Note: you can add HNSW params via metadata if desired (implementation-dependent)
    collection = client.get_or_create_collection(
        name="movie_plots",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        metadata={"hnsw:space": "cosine"}
    )

    count =load_data_to_chroma(collection, "dataset/wiki_movie_plots_deduped.csv")
    print(f"{count} items are contained in the ChromaDB collection 'movie_plots'")

    # --- Query: basic semantic search ---
    res1 = collection.query(
        query_texts=["What movies are based on plays?"],
        n_results=5,
    )
    print("What movies are based on plays?", res1)


    print("\nDone")

if __name__ == "__main__":
    main()
