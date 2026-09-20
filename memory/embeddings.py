"""
Ultron semantic embedding layer.

Uses:
    sentence-transformers/all-MiniLM-L6-v2

Responsibilities:
    - Load the embedding model once
    - Generate embeddings for memory text
    - Store embeddings in the SQLite database
    - Load existing memory embeddings
    - Provide cosine-similarity utilities

This module does NOT decide what should become a memory.
That belongs to the memory manager.
"""

from __future__ import annotations

import threading
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from .database import (
    get_active_memory_embeddings,
    get_memory_embedding,
    upsert_memory_embedding,
    delete_memory_embedding,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The model is intentionally kept resident after first load.
# Your benchmark showed this is practical on your machine.
_MODEL: SentenceTransformer | None = None

# Prevent multiple threads from loading the model simultaneously.
_MODEL_LOCK = threading.Lock()


# ============================================================
# Model loading
# ============================================================

def get_model() -> SentenceTransformer:
    """
    Load MiniLM once and reuse it for the lifetime of the process.
    """

    global _MODEL

    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = SentenceTransformer(
                    MODEL_NAME,
                    device="cpu",
                )

    return _MODEL


# ============================================================
# Embedding generation
# ============================================================

def encode(
    text: str,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert one piece of text into an embedding vector.

    Returns:
        numpy float32 vector
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()

    if not text:
        raise ValueError("Cannot embed empty text")

    model = get_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )

    return np.asarray(embedding, dtype=np.float32)


def encode_many(
    texts: Iterable[str],
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode multiple texts in one model call.

    Returns:
        numpy array with shape (count, dimensions)
    """

    texts = list(texts)

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if any(not isinstance(text, str) for text in texts):
        raise TypeError("All texts must be strings")

    cleaned = [text.strip() for text in texts]

    if any(not text for text in cleaned):
        raise ValueError("Cannot embed empty text")

    model = get_model()

    embeddings = model.encode(
        cleaned,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)


# ============================================================
# Database integration
# ============================================================

def embed_memory(
    memory_id: int,
    content: str,
) -> np.ndarray:
    """
    Generate and store the embedding for one memory.
    """

    embedding = encode(content)

    upsert_memory_embedding(
        memory_id=memory_id,
        model=MODEL_NAME,
        embedding=embedding,
    )

    return embedding


def refresh_memory_embedding(memory_id: int) -> np.ndarray:
    """
    Reload a memory from the database and regenerate its embedding.
    """

    from .database import get_memory

    memory = get_memory(memory_id)

    if memory is None:
        raise ValueError(f"Memory {memory_id} does not exist")

    return embed_memory(
        memory_id,
        memory["content"],
    )


def remove_memory_embedding(memory_id: int) -> None:
    """
    Remove a stored embedding.
    """

    delete_memory_embedding(
        memory_id=memory_id,
        model=MODEL_NAME,
    )


def get_stored_embedding(memory_id: int) -> np.ndarray | None:
    """
    Retrieve a memory's stored embedding.
    """

    return get_memory_embedding(
        memory_id=memory_id,
        model=MODEL_NAME,
    )


# ============================================================
# Semantic similarity
# ============================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Calculate cosine similarity between two vectors.

    Because our stored/query embeddings are normalized,
    this is effectively their dot product.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("cosine_similarity expects 1-D vectors")

    if a.shape != b.shape:
        raise ValueError(
            f"Embedding dimensions do not match: "
            f"{a.shape} vs {b.shape}"
        )

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ============================================================
# Semantic memory search
# ============================================================

def search_memories_semantically(
    query: str,
    *,
    limit: int = 10,
) -> list[dict]:
    """
    Perform a semantic search over all active memory embeddings.

    Returns dictionaries containing:

        memory_id
        similarity
        embedding

    The actual memory text is retrieved by the retrieval layer.
    """

    if limit <= 0:
        return []

    query_embedding = encode(query)

    stored = get_active_memory_embeddings(
        model=MODEL_NAME,
    )

    results: list[dict] = []

    for row in stored:
        memory_id = row["memory_id"]
        embedding = row["embedding"]

        similarity = cosine_similarity(
            query_embedding,
            embedding,
        )

        results.append(
            {
                "memory_id": memory_id,
                "similarity": similarity,
                "embedding": embedding,
            }
        )

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:limit]


# ============================================================
# Maintenance
# ============================================================

def embed_all_active_memories() -> int:
    """
    Ensure every active memory has a current embedding.

    Returns:
        number of embeddings created/refreshed
    """

    from .database import get_active_memories

    memories = get_active_memories()

    count = 0

    for memory in memories:
        embed_memory(
            memory["id"],
            memory["content"],
        )
        count += 1

    return count


# ============================================================
# Diagnostics
# ============================================================

def self_test() -> None:
    """
    Basic embedding-layer test.

    Does not permanently create a memory.
    """

    print("Loading embedding model...")

    first = encode(
        "Ultron is a local AI assistant."
    )

    second = encode(
        "This system contains a local artificial intelligence."
    )

    third = encode(
        "Bananas grow on plants."
    )

    print(f"MODEL: {MODEL_NAME}")
    print(f"DIMENSIONS: {first.shape[0]}")

    similarity_related = cosine_similarity(
        first,
        second,
    )

    similarity_unrelated = cosine_similarity(
        first,
        third,
    )

    print(
        f"RELATED SIMILARITY: "
        f"{similarity_related:.4f}"
    )

    print(
        f"UNRELATED SIMILARITY: "
        f"{similarity_unrelated:.4f}"
    )

    if similarity_related <= similarity_unrelated:
        raise RuntimeError(
            "Semantic similarity test failed."
        )

    print("Embedding self-test passed.")


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":
    self_test()