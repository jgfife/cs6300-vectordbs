#!/usr/bin/env python3
"""
Database utilities for loading data into ChromaDB.
"""

from __future__ import annotations
import csv
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
    csv_path: str | Path
) -> int:
    """
    Load movie plots from CSV into a ChromaDB collection.
    
    Args:
        collection: ChromaDB collection to load data into
        csv_path: Path to the CSV file
        
    Returns:
        Number of movies loaded
    """
    movies = load_movie_plots_csv(csv_path)
    if collection.count() == len(movies):
        print("Collection already contains all movie plots; skipping load.")
        return collection.count()

    batch_size = 5000  # Safe batch size under ChromaDB's limit
    total_loaded = 0
    
    # Process in batches to respect ChromaDB's batch size limits
    for i in range(0, len(movies), batch_size):
        batch = movies[i:i + batch_size]
        
        ids = [movie["id"] for movie in batch]
        documents = [movie["text"] for movie in batch]
        metadatas = [movie["meta"] for movie in batch]
        
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        total_loaded += len(batch)
        print(f"Loaded batch {i//batch_size + 1}: {total_loaded}/{len(movies)} movies")
    
    return collection.count()