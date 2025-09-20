#!/usr/bin/env python3

#!/usr/bin/env python3
"""
FAISS demo: minimal vector store with upsert (insert/update), query, filters, and save/load.

Run:
  python faiss_demo.py
"""

from __future__ import annotations
import os
import shutil
import pickle
from pathlib import Path
import pprint
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

pp = pprint.PrettyPrinter(indent=2, width=100)

# --------- Config ----------
DB_DIR = Path("./faiss_demo_store")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
EMBED_DIM = 384

# --------- Tiny "Vector DB" wrapper over FAISS ----------
class MiniVectorStore:
    """
    A tiny, didactic wrapper around FAISS that:
      - stores text + metadata in Python dicts
      - stores embeddings in numpy arrays
      - builds a FAISS index (cosine via inner product on normalized vectors)
      - supports upsert by rebuilding (fine for small demos)
      - supports simple metadata/content filters (post-search)
      - supports save/load to disk
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.id_to_doc: Dict[str, str] = {}
        self.id_to_meta: Dict[str, Dict] = {}
        self.id_to_vec: Dict[str, np.ndarray] = {}
        self.index: Optional[faiss.Index] = None
        self._idmap: List[str] = []  # row -> string id

    # ---------- Embeddings ----------
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        # normalize_embeddings=True gives unit vectors → cosine == dot product
        return vecs.astype("float32")

    # ---------- Index lifecycle ----------
    def _rebuild_index(self):
        """(Re)build a simple IndexFlatIP over all vectors; maintain row->id map."""
        if not self.id_to_vec:
            self.index = None
            self._idmap = []
            return
        ids = list(self.id_to_vec.keys())
        self._idmap = ids
        mat = np.stack([self.id_to_vec[_id] for _id in ids], axis=0).astype("float32")
        self.index = faiss.IndexFlatIP(mat.shape[1])  # inner product on normalized vectors = cosine sim
        self.index.add(mat)

    # ---------- Public API ----------
    def upsert(self, ids: List[str], texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Insert new or update existing documents. Rebuild index for simplicity."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        vecs = self.embed_texts(texts)
        for _id, txt, meta, v in zip(ids, texts, metadatas, vecs):
            self.id_to_doc[_id] = txt
            self.id_to_meta[_id] = meta
            self.id_to_vec[_id] = v
        self._rebuild_index()

    def query(
        self,
        query_texts: List[str],
        n_results: int = 3,
        where: Optional[Dict] = None,                 # metadata filters: exact-match dict, e.g. {"topic":"indexes"}
        where_document_contains: Optional[str] = None # substring filter on raw text
    ) -> Dict:
        assert self.index is not None, "Index is empty—insert first."
        qvecs = self.embed_texts(query_texts)
        D, I = self.index.search(qvecs, min(n_results * 5, len(self._idmap)))  # over-fetch a bit, then filter
        results = {"ids": [], "documents": [], "metadatas": [], "scores": []}

        for qi, (ds, is_ ) in enumerate(zip(D, I)):
            # Collect candidates (id, score), then filter
            cands: List[Tuple[str, float]] = []
            for score, row in zip(ds, is_):
                if row == -1:
                    continue
                _id = self._idmap[row]
                # Apply metadata filter
                if where:
                    meta = self.id_to_meta[_id]
                    if not all(meta.get(k) == v for k, v in where.items()):
                        continue
                # Apply content substring filter
                if where_document_contains:
                    if where_document_contains not in self.id_to_doc[_id]:
                        continue
                cands.append((_id, float(score)))

            # Rerank by score and truncate to n_results
            cands.sort(key=lambda x: x[1], reverse=True)
            cands = cands[:n_results]

            results["ids"].append([c[0] for c in cands])
            results["scores"].append([c[1] for c in cands])
            results["documents"].append([self.id_to_doc[c[0]] for c in cands])
            results["metadatas"].append([self.id_to_meta[c[0]] for c in cands])

        return results

    # ---------- Persistence ----------
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        # Save Python dicts
        with open(path / "store.pkl", "wb") as f:
            pickle.dump(
                {
                    "id_to_doc": self.id_to_doc,
                    "id_to_meta": self.id_to_meta,
                    "id_to_vec": {k: v.astype("float32") for k, v in self.id_to_vec.items()},
                    "_idmap": self._idmap,
                    "model_name": self.model_name,
                },
                f,
            )
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))

    @classmethod
    def load(cls, path: Path) -> "MiniVectorStore":
        with open(path / "store.pkl", "rb") as f:
            blob = pickle.load(f)
        store = cls(model_name=blob.get("model_name", MODEL_NAME))
        store.id_to_doc = blob["id_to_doc"]
        store.id_to_meta = blob["id_to_meta"]
        store.id_to_vec = {k: np.asarray(v).astype("float32") for k, v in blob["id_to_vec"].items()}
        store._idmap = blob["_idmap"]
        # Load FAISS index (and trust it matches vectors); rebuild if missing
        faiss_path = path / "index.faiss"
        if faiss_path.exists():
            store.index = faiss.read_index(str(faiss_path))
        else:
            store._rebuild_index()
        return store


# --------- Demo data (same theme as your Chroma example) ----------
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

doc_update = {
    "id": "doc-002",
    "text": "IVF (Inverted File) uses coarse quantization; IVF-Flat stores raw vectors in the cells it probes.",
    "meta": {"topic": "indexes", "source": "revised", "level": "intro"}
}

# --------- Demo runner ----------
def fresh_start():
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

def show(title: str, obj):
    print(f"\n=== {title} ===")
    pp.pprint(obj)

def main():
    fresh_start()
    store = MiniVectorStore(model_name=MODEL_NAME)

    # --- UPSERT (INSERT) ---
    store.upsert(
        ids=[d["id"] for d in docs_insert],
        texts=[d["text"] for d in docs_insert],
        metadatas=[d["meta"] for d in docs_insert],
    )

    # --- Query: basic semantic search ---
    res1 = store.query(
        query_texts=["Which method compresses vectors with codebooks?"],
        n_results=2,
    )
    show("Query 1 (top 2)", res1)

    # --- Filter by metadata (where) ---
    res2 = store.query(
        query_texts=["tell me about approximate nearest neighbor structures"],
        n_results=5,
        where={"topic": "indexes"},
    )
    show("Query 2 with where={'topic': 'indexes'}", res2)

    # --- Filter by document content (substring) ---
    res3 = store.query(
        query_texts=["graph-based ANN"],
        n_results=5,
        where_document_contains="HNSW",
    )
    show("Query 3 with where_document_contains='HNSW'", res3)

    # --- UPSERT (UPDATE) ---
    store.upsert(
        ids=[doc_update["id"]],
        texts=[doc_update["text"]],
        metadatas=[doc_update["meta"]],
    )

    res4 = store.query(
        query_texts=["explain IVF-Flat"],
        n_results=3,
        where={"source": "revised"},
    )
    show("Query 4 after UPDATE upsert (where={'source': 'revised'})", res4)

    # --- Save / Load round-trip ---
    store.save(DB_DIR)
    print("\nSaved store to:", DB_DIR.resolve())
    reloaded = MiniVectorStore.load(DB_DIR)
    res5 = reloaded.query(["codebooks compression"], n_results=1)
    show("Query 5 after reload", res5)

if __name__ == "__main__":
    main()
