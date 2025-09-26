#!/usr/bin/env python3
"""Generate vector database queries from text using Ollama."""

from __future__ import annotations

import random
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Represents a single search result from vector database query."""
    rank: int
    distance: float
    document: str
    relevancy_score: Optional[float] = None
    relevancy_explanation: Optional[str] = None
    is_relevant: Optional[int] = None
    
    def add_relevancy(self, score: float, explanation: str) -> None:
        """Add relevancy scoring in-place."""
        self.relevancy_score = score
        self.relevancy_explanation = explanation
        self.is_relevant = 1 if score >= 4.0 else 0


@dataclass 
class QueryResult:
    """Represents results for a single query."""
    query_id: int
    query: str
    latency_ms: float
    results: List[SearchResult]
    
    @classmethod
    def from_chroma_response(
        cls, 
        query_id: int, 
        query: str, 
        latency_ms: float, 
        chroma_response: Dict[str, Any]
    ) -> 'QueryResult':
        """Factory method to create QueryResult from ChromaDB response."""
        results = [
            SearchResult(
                rank=j + 1,
                distance=distance,
                document=doc
            )
            for j, (doc, distance) in enumerate(
                zip(chroma_response['documents'][0], chroma_response['distances'][0])
            )
        ]
        return cls(query_id, query, latency_ms, results)


class QueryResults:
    """Container for managing multiple query results with relevancy scoring."""
    
    def __init__(self):
        self.results: List[QueryResult] = []
    
    def add_result(self, query_result: QueryResult) -> None:
        """Add a query result to the collection."""
        self.results.append(query_result)
    
    def get_latencies(self) -> List[float]:
        """Get list of all query latencies in milliseconds."""
        return [result.latency_ms for result in self.results]
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)
    
    def score_relevancy(
        self, 
        model: str = "gpt-oss", 
        ollama_url: str = "http://localhost:11434"
    ) -> None:
        """Score the relevancy of query results using Ollama with gpt-oss."""
        print(f"Scoring relevancy for {len(self.results)} queries using {model}...")
        
        for query_result in self.results:
            query = query_result.query
            
            print(f"Scoring query {query_result.query_id}/{len(self.results)}: '{query}'")
            
            for result in query_result.results:
                document = result.document
                
                prompt = f"""Rate the relevancy of this document to the given query on a scale of 1-5:

Query: "{query}"

Document: "{document[:500]}..."

Provide your response in this exact format:
Score: [1-5]
Explanation: [brief 1-2 sentence explanation]

Where:
1 = Not relevant at all
2 = Slightly relevant and has minimal connection such as matching genre
3 = Moderately relevant and has parallels in topic or subject matter
4 = Highly relevant and has parallels in time period, location, theme, or genre
5 = Extremely relevant and directly addresses the query"""
                
                try:
                    response = requests.post(
                        f"{ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    result_json = response.json()
                    generated_text = result_json.get("response", "")
                    
                    # Parse score and explanation from response
                    score = 3.0  # Default fallback score
                    explanation = "Unable to parse response"
                    
                    for line in generated_text.strip().split('\n'):
                        line = line.strip()
                        if line.startswith("Score:"):
                            try:
                                score_str = line.replace("Score:", "").strip()
                                score = float(score_str)
                                # Clamp score to 1-5 range
                                score = max(1.0, min(5.0, score))
                            except ValueError:
                                pass
                        elif line.startswith("Explanation:"):
                            explanation = line.replace("Explanation:", "").strip()
                    
                    # Add relevancy data in-place
                    result.add_relevancy(score, explanation)

                    print(f"  Result Rank [{result.rank}], Score [{score}], IsRelevant [{result.is_relevant}], Explanation [{explanation}]")
                    
                except Exception as e:
                    print(f"Error scoring relevancy for result {result.rank}: {e}")
                    # Add default scoring on error
                    result.add_relevancy(3.0, f"Error during scoring: {str(e)}")
        
        print(f"Completed relevancy scoring for {len(self.results)} queries")


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
    - A summary of the key ideas

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