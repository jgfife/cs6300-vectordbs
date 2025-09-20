#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["A cat sits on the mat.", "A dog lies on the rug.", "Quantum tunneling is weird."]
embs = model.encode(sentences, normalize_embeddings=True)
print(embs.shape)  # (3, 384)
print(embs[0][:10])
print(embs[1][:10])
print(embs[2][:10])

