#!/usr/bin/env python3
"""
Database utilities for loading data into ChromaDB.
"""

from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Dict, List, Any
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def load_movie_plots_csv(csv_path: str | Path) -> List[Dict[str, Any]]:
    """
    Parse the wiki movie plots CSV file and return structured data.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of movie plot dictionaries
    """
    csv_path = Path(csv_path)
    movies = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Create a unique ID for each movie
            movie_id = f"movie-{i+1:06d}"
            
            # Clean and structure the data
            movie = {
                "id": movie_id,
                "text": row["Plot"],
                "meta": {
                    "title": row["Title"],
                    "year": row["Release Year"],
                    "director": row["Director"],
                    "cast": row["Cast"],
                    "genre": row["Genre"],
                    "origin": row["Origin/Ethnicity"],
                    "wiki_page": row["Wiki Page"]
                }
            }
            movies.append(movie)
    
    return movies


def load_data_to_chroma(
    collection: chromadb.Collection, 
    data: List[Dict[str, Any]]
) -> int:
    """
    Load structured data into a ChromaDB collection.
    
    Args:
        collection: ChromaDB collection to load data into
        data: List of data items, each with 'id', 'text', and 'meta' keys
        
    Returns:
        Number of items loaded
    """
    if collection.count() == len(data):
        print("Collection already contains all data; skipping load.")
        return collection.count()

    batch_size = 5000  # Safe batch size under ChromaDB's limit
    total_loaded = 0
    total_start_time = time.time()
    
    # Process in batches to respect ChromaDB's batch size limits
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        ids = [item["id"] for item in batch]
        documents = [item["text"] for item in batch]
        metadatas = [item["meta"] for item in batch]
        
        batch_start_time = time.time()
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        batch_time = time.time() - batch_start_time
        
        total_loaded += len(batch)
        print(f"Loaded batch {i//batch_size + 1}: {total_loaded}/{len(data)} items (batch time: {batch_time:.2f}s)")
    
    total_time = time.time() - total_start_time
    print(f"Total loading time: {total_time:.2f}s")
    
    return collection.count()