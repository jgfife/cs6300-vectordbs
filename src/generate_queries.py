#!/usr/bin/env python3
"""Generate vector database queries from text using Ollama."""

from __future__ import annotations

import random
import requests
from pathlib import Path
from typing import Any, Dict, List


def generate_queries_from_dataset(
    dataset: List[Dict[str, Any]], 
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434"
) -> List[str]:
    """
    Generate vector database queries from randomly selected 10% of dataset using Ollama.
    
    Args:
        dataset: List of dictionaries with 'text' and 'meta' fields
        model: Ollama model to use for generation
        ollama_url: Base URL for Ollama API
        
    Returns:
        List of generated queries suitable for vector database search
    """
    # 10% of the dataset
    sample_size = max(1, len(dataset) // 10)

    # Try to load existing queries from file
    queries = []
    queries_file = Path("dataset/chroma_queries.txt")
    if queries_file.exists():
        print(f"Loading queries from {queries_file}")
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Loaded {len(queries)} queries from file:")

    # bail if we have enough
    if len(queries) >= sample_size:
        return queries
    
    print(f"Query file {queries_file} not found or wasn't complete. Generating new queries...")

    # Randomly select sample size from the dataset
    selected_items = random.sample(dataset, sample_size)
    count = len(queries)  # start from existing count if any
    with open(queries_file, 'w', encoding='utf-8') as f:
        for item in selected_items:
            text = item["text"]
            
            prompt = f"""
    Based on the following text, generate 1 search query that someone might use to find this document in a vector database. The query should be:
    - A natural language question or phrase
    - Related to the main concepts and topics in the text

    Text: {text[:500]}...  # Truncated for brevity

    Generate exactly 1 query:
    """
            
            try:
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                generated_text = result.get("response", "")
                
                # Parse query from response
                for line in generated_text.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove numbering if present (1., 2., etc.)
                        if line[0].isdigit() and '.' in line[:3]:
                            line = line.split('.', 1)[1].strip()
                        queries.append(line) # add to results
                        f.write(f"{line}\n") # persist to file
                        count += 1
                        print(f"Generated Query Progress: {count}/{sample_size}")                     
                        break  # Only take the first valid query
                
            except Exception as e:
                print(f"Error generating query for text: {e}")
    
    return queries    