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
from chromadb.config import Settings
import db

def main():
    dbDir = Path("./db/chroma")
    client = chromadb.PersistentClient(path=str(dbDir), settings=Settings(anonymized_telemetry=False))

    # Create or get a collection with an embedding function
    # Note: you can add HNSW params via metadata if desired (implementation-dependent)
    collection = client.get_or_create_collection(
        name="movie_plots",
        embedding_function=SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        metadata={"hnsw:space": "cosine"}
    )

    movies = db.load_movie_plots_csv("dataset/wiki_movie_plots_deduped.csv")
    if collection.count() == len(movies):
        print("Collection already contains all movie plots; skipping load.")
        return collection.count()

    count = db.load_data_to_chroma(collection, movies)
    print(f"{count} items are contained in the ChromaDB collection 'movie_plots'")

    # TODO: Generate queries and time them

if __name__ == "__main__":
    main()
