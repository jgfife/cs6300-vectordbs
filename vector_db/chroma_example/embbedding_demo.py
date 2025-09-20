#!/usr/bin/env python3

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedder = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "HNSW is a graph-based ANN index."
vec = embedder([text])   # returns a list of embedding vectors
print("Vector length:", len(vec[0]))
print("First 5 dims:", vec[0][:5])
