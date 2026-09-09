# backend/embedder.py
"""
Role 3: Embedder
Wraps sentence-transformers to turn text chunks into vectors for indexing,
and to embed user questions the same way at query time.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Config ---------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good enough for MVP
EMBEDDING_DIM = 384

# --- Lazy singleton model loader -------------------------------------------

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it across calls."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# --- Public API -------------------------------------------------------------

def embed_chunks(chunks: List[str]) -> np.ndarray:
    """
    Embed a list of text chunks (from chunker.py) for indexing.

    Args:
        chunks: list of chunk strings.

    Returns:
        np.ndarray of shape (n_chunks, EMBEDDING_DIM), dtype float32,
        L2-normalized so cosine similarity == dot product.
    """
    if not chunks:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_model()
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user question at retrieval time.

    Args:
        query: the user's question string.

    Returns:
        np.ndarray of shape (EMBEDDING_DIM,), dtype float32, normalized.
    """
    model = get_model()
    embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.astype(np.float32)


# --- Standalone sanity check --------------------------------------------

if __name__ == "__main__":
    sample_chunks = [
        "The midterm exam is worth 30% of the final grade.",
        "Office hours are held Tuesdays and Thursdays from 2-4pm in Room 204.",
        "The final project is due on the last day of class and counts for 25%.",
    ]

    chunk_embeddings = embed_chunks(sample_chunks)
    print(f"Chunk embeddings shape: {chunk_embeddings.shape}")  # (3, 384)
    print(f"dtype: {chunk_embeddings.dtype}")

    query_embedding = embed_query("When is the midterm worth?")
    print(f"Query embedding shape: {query_embedding.shape}")  # (384,)

    # Quick cosine-similarity check (since vectors are normalized, dot == cosine)
    sims = chunk_embeddings @ query_embedding
    best_idx = int(np.argmax(sims))
    print(f"Most similar chunk (score={sims[best_idx]:.3f}): {sample_chunks[best_idx]}")