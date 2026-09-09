from __future__ import annotations

import json
import os
import pickle
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss is required for VectorStore. Install with: pip install faiss-cpu"
    ) from e


EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass
class Record:
    """A single stored item: its id, original text, and metadata."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    In-memory vector store using FAISS for similarity search.

    Uses an IndexFlatIP (inner product) index over L2-normalized vectors,
    which is equivalent to cosine similarity search. This is exact
    (brute-force) search — fine for up to ~1M vectors; swap in an
    IVF/HNSW index if you need to scale further.
    """

    def __init__(
        self,
        dim: int,
        embed_fn: Optional[EmbedFn] = None,
        normalize: bool = True,
    ):
        self.dim = dim
        self.embed_fn = embed_fn
        self.normalize = normalize

        self.index = faiss.IndexFlatIP(dim)
        # Maps FAISS internal row index -> our Record id
        self._row_to_id: List[str] = []
        # Maps our Record id -> Record
        self._records: Dict[str, Record] = {}

    
    # Internal helpers


    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of dim {self.dim}, got {vectors.shape[1]}"
            )
        if self.normalize:
            faiss.normalize_L2(vectors)
        return vectors

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.embed_fn is None:
            raise ValueError(
                "No embed_fn configured. Pass embeddings directly via "
                "add_embeddings()/search_by_vector(), or set embed_fn."
            )
        vectors = np.array(self.embed_fn(texts), dtype="float32")
        return vectors

    # Add

    def add(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Embed `texts` (via embed_fn) and add them to the store."""
        vectors = self._embed(texts)
        return self.add_embeddings(texts, vectors, metadatas=metadatas, ids=ids)

    def add_embeddings(
        self,
        texts: Sequence[str],
        vectors: np.ndarray,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Add pre-computed embeddings directly, skipping embed_fn."""
        n = len(texts)
        if metadatas is None:
            metadatas = [{} for _ in range(n)]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]

        if not (len(texts) == len(metadatas) == len(ids) == vectors.shape[0]):
            raise ValueError("texts, metadatas, ids, and vectors must be the same length")

        vectors = self._prepare_vectors(vectors)

        self.index.add(vectors)
        for _id, text, meta in zip(ids, texts, metadatas):
            self._records[_id] = Record(id=_id, text=text, metadata=meta)
            self._row_to_id.append(_id)

        return list(ids)

    # Search

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[Record, float]]:
        """Embed `query` and return the top-k most similar records."""
        vector = self._embed([query])[0]
        return self.search_by_vector(
            vector, k=k, score_threshold=score_threshold, filter_fn=filter_fn
        )

    def search_by_vector(
        self,
        vector: Sequence[float],
        k: int = 5,
        score_threshold: Optional[float] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[Record, float]]:
        """Search using a pre-computed query vector."""
        if self.index.ntotal == 0:
            return []

        query = self._prepare_vectors(np.array(vector))

        # Over-fetch when filtering, since some hits may get dropped
        fetch_k = k * 5 if filter_fn else k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, rows = self.index.search(query, fetch_k)

        results: List[Tuple[Record, float]] = []
        for score, row in zip(scores[0], rows[0]):
            if row == -1:
                continue
            if score_threshold is not None and score < score_threshold:
                continue
            record_id = self._row_to_id[row]
            record = self._records.get(record_id)
            if record is None:
                continue  # deleted record, tombstoned row
            if filter_fn and not filter_fn(record.metadata):
                continue
            results.append((record, float(score)))
            if len(results) >= k:
                break

        return results

    # Delete

    def delete(self, ids: Sequence[str]) -> None:
        """
        Delete records by id.

        Note: FAISS's IndexFlatIP doesn't support true in-place removal,
        so this rebuilds the index from remaining records. Fine for
        occasional deletes; batch deletes together if doing many at once.
        """
        ids_to_remove = set(ids)
        remaining = [
            (rec_id, self._records[rec_id])
            for rec_id in self._row_to_id
            if rec_id not in ids_to_remove
        ]
        for rec_id in ids_to_remove:
            self._records.pop(rec_id, None)

        self.index = faiss.IndexFlatIP(self.dim)
        self._row_to_id = []

        if remaining:
            texts = [rec.text for _, rec in remaining]
            vecs = self._embed(texts) if self.embed_fn else None
            if vecs is None:
                raise ValueError(
                    "Cannot rebuild index after delete without embed_fn "
                    "(no cached vectors are stored). Re-add embeddings manually."
                )
            vecs = self._prepare_vectors(vecs)
            self.index.add(vecs)
            self._row_to_id = [rec_id for rec_id, _ in remaining]

    def clear(self) -> None:
        """Remove everything from the store."""
        self.index = faiss.IndexFlatIP(self.dim)
        self._row_to_id = []
        self._records = {}

    
    # Introspection
   

    def __len__(self) -> int:
        return len(self._records)

    def get(self, id: str) -> Optional[Record]:
        return self._records.get(id)

    # Persistence
   

    def save(self, path: str) -> None:
        """Persist the FAISS index + metadata to `path` (a directory)."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        meta = {
            "dim": self.dim,
            "normalize": self.normalize,
            "row_to_id": self._row_to_id,
            "records": {
                rec_id: {"id": rec.id, "text": rec.text, "metadata": rec.metadata}
                for rec_id, rec in self._records.items()
            },
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str, embed_fn: Optional[EmbedFn] = None) -> "VectorStore":
        """Load a previously saved store from `path`."""
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        store = cls(dim=meta["dim"], embed_fn=embed_fn, normalize=meta["normalize"])
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        store._row_to_id = meta["row_to_id"]
        store._records = {
            rec_id: Record(**rec) for rec_id, rec in meta["records"].items()
        }
        return store

# Example usage


if __name__ == "__main__":
    import random

    def fake_embed(texts: Sequence[str]) -> List[List[float]]:
        """Deterministic pseudo-embedding for demo purposes only."""
        vectors = []
        for t in texts:
            rng = random.Random(t)
            vectors.append([rng.uniform(-1, 1) for _ in range(16)])
        return vectors

    store = VectorStore(dim=16, embed_fn=fake_embed)
    store.add(
        ["the cat sat on the mat", "dogs are loyal animals", "python is a programming language"],
        metadatas=[{"topic": "cats"}, {"topic": "dogs"}, {"topic": "tech"}],
    )

    print(f"Store has {len(store)} records")
    for record, score in store.search("cat", k=2):
        print(f"{score:.3f} | {record.text} | {record.metadata}")