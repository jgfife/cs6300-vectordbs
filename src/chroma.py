#!/usr/bin/env python3
"""
Chroma demo: persistent collection, upsert (insert + update), query, and filters.

Run:
  python chroma_demo.py
"""

from __future__ import annotations
from pathlib import Path
import time
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings
import db
from generate_queries import generate_queries_from_dataset
from metrics import calculate_percentiles, print_metrics

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
    if collection.count() != len(movies):
        print(f"Loading {len(movies)} items into ChromaDB collection 'movie_plots'...")
        count = db.load_data_to_chroma(collection, movies)
        print(f"{count} items are contained in the ChromaDB collection 'movie_plots'")
    
    queries = generate_queries_from_dataset(movies, model="gemma3", ollama_url="http://localhost:11434")
    query_latencies = []
    query_results = []
    print(f"Processing {len(queries)} queries...")

    for i, query in enumerate(queries):
        start_time = time.time()
        res = collection.query(
            query_texts=[query],
            n_results=5,
        )
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        query_latencies.append(latency_ms)
        
        query_result = {
            "query_id": i + 1,
            "query": query,
            "latency_ms": latency_ms,
            "results": [
                {
                    "rank": j + 1,
                    "distance": distance,
                    "document": doc
                }
                for j, (doc, distance) in enumerate(zip(res['documents'][0], res['distances'][0]))
            ]
        }
        
        query_results.append(query_result)

    # Calculate and display query performance metrics
    if query_latencies:
        print(f"\nProcessed {len(query_results)} queries successfully")
        
        metrics = calculate_percentiles(query_latencies)
        print("\n" + "="*50)
        print_metrics(metrics, unit="ms")
        print("="*50)

if __name__ == "__main__":
    main()
