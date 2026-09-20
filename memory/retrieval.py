"""
Ultron hybrid memory retrieval.

Combines:

    1. SQLite FTS5 lexical retrieval
    2. MiniLM semantic similarity
    3. Memory importance
    4. Recency
    5. Access frequency

Important design rule:

    Retrieval only evaluates the CURRENT query.

It must never decide what the user meant by searching old chat
messages. Historical conversation is storage, not intent.
"""

from __future__ import annotations

import math
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .database import (
    get_connection,
    get_memory,
    get_active_memory_embeddings,
    mark_memory_accessed,
)
from .embeddings import (
    MODEL_NAME,
    encode,
    cosine_similarity,
)


# ============================================================
# Configuration
# ============================================================

# Weighting for the final hybrid score.
#
# Semantic similarity is the strongest signal because it handles
# paraphrases and concepts that do not share exact words.
SEMANTIC_WEIGHT = 0.55

# Exact/keyword matching remains important for things like:
# Qwen2.5-Coder, filenames, hardware names, etc.
LEXICAL_WEIGHT = 0.30

# User explicitly marking something important should matter,
# but should not overpower relevance.
IMPORTANCE_WEIGHT = 0.10

# Small metadata signals.
RECENCY_WEIGHT = 0.03
ACCESS_WEIGHT = 0.02

# Scores below this are normally not useful enough to inject
# into the model's context unless there is a direct lexical hit.
MINIMUM_SCORE = 0.28

# Limit how many memories are returned to the model.
DEFAULT_LIMIT = 8

# Recency half-life.
#
# After 30 days, the recency component is approximately half
# of what it would be for a brand-new memory.
RECENCY_HALF_LIFE_DAYS = 30.0


# ============================================================
# Utility functions
# ============================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: str | None) -> datetime | None:
    """
    Parse an SQLite timestamp safely.

    Supports ISO timestamps with or without timezone information.
    """

    if not value:
        return None

    try:
        text = value.strip()

        # SQLite may store timestamps ending with Z.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        parsed = datetime.fromisoformat(text)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except (ValueError, TypeError):
        return None


def _recency_score(timestamp: str | None) -> float:
    """
    Convert age into a 0..1 recency score.

    Newer memory -> closer to 1
    Older memory -> closer to 0
    """

    created = _parse_timestamp(timestamp)

    if created is None:
        return 0.0

    age_seconds = max(
        0.0,
        (_utc_now() - created).total_seconds(),
    )

    age_days = age_seconds / 86400.0

    return math.pow(
        0.5,
        age_days / RECENCY_HALF_LIFE_DAYS,
    )


def _importance_score(value: Any) -> float:
    """
    Convert database importance (0..5) into 0..1.
    """

    try:
        importance = float(value)
    except (TypeError, ValueError):
        return 0.0

    importance = max(0.0, min(5.0, importance))

    return importance / 5.0


def _access_score(value: Any) -> float:
    """
    Convert access count into a bounded 0..1 signal.

    log1p prevents frequently accessed memories from completely
    dominating newer/relevant memories.
    """

    try:
        count = max(0, int(value))
    except (TypeError, ValueError):
        count = 0

    # 1 + log1p(9) ≈ 3.3, so normalize around 10 accesses.
    return min(
        1.0,
        math.log1p(count) / math.log1p(10),
    )


# ============================================================
# FTS5 query handling
# ============================================================

def _extract_search_terms(query: str) -> list[str]:
    """
    Extract safe searchable terms from a natural-language query.

    We do this instead of sending raw user text directly into
    FTS5, because FTS5 interprets characters such as:

        AND
        OR
        NOT
        *
        "

    as query syntax.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    # Unicode-aware word extraction.
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)

    # Remove duplicates while preserving order.
    seen: set[str] = set()
    result: list[str] = []

    for term in terms:
        term = term.strip()

        if not term:
            continue

        normalized = term.casefold()

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(term)

    return result


def _build_fts_query(query: str) -> str:
    """
    Convert normal user text into a safe FTS5 OR query.
    """

    terms = _extract_search_terms(query)

    if not terms:
        return ""

    # Each term is quoted so FTS5 treats it as literal text.
    return " OR ".join(
        f'"{term.replace(chr(34), chr(34) * 2)}"'
        for term in terms
    )


# ============================================================
# Lexical retrieval
# ============================================================

def lexical_search(
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """
    Search active memories with SQLite FTS5.

    Returns:

        memory_id
        bm25
        lexical_rank

    Lower BM25 values are better in SQLite FTS5.
    We convert the ordering into a simple 0..1 rank score.
    """

    if limit <= 0:
        return []

    fts_query = _build_fts_query(query)

    if not fts_query:
        return []

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT
                rowid AS memory_id,
                bm25(memories_fts) AS bm25_score
            FROM memories_fts
            JOIN memories
                ON memories.id = memories_fts.rowid
            WHERE memories_fts MATCH ?
              AND memories.active = 1
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()

    except sqlite3.OperationalError:
        # Invalid/unusable FTS query should not crash Ultron.
        return []

    finally:
        conn.close()

    results: list[dict] = []

    total = len(rows)

    for index, row in enumerate(rows):
        # Best result gets 1.0.
        #
        # Example:
        #   1 result -> 1.0
        #   2 results -> 1.0, 0.5
        #   5 results -> 1.0, .8, .6, .4, .2
        lexical_rank = (
            1.0
            if total == 1
            else 1.0 - (index / total)
        )

        results.append(
            {
                "memory_id": int(row["memory_id"]),
                "bm25_score": float(row["bm25_score"]),
                "lexical_score": lexical_rank,
            }
        )

    return results


# ============================================================
# Semantic retrieval
# ============================================================

def semantic_search(
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """
    Search all active memory embeddings using MiniLM.

    The current database is small enough that scanning active
    memory embeddings is perfectly reasonable.

    This can later be optimized with an approximate nearest
    neighbor index if the memory collection becomes very large.
    """

    if limit <= 0:
        return []

    query_embedding = encode(query)

    stored_embeddings = get_active_memory_embeddings(
        model=MODEL_NAME,
    )

    results: list[dict] = []

    for row in stored_embeddings:
        memory_id = int(row["memory_id"])
        embedding = row["embedding"]

        similarity = cosine_similarity(
            query_embedding,
            embedding,
        )

        results.append(
            {
                "memory_id": memory_id,
                "semantic_similarity": float(similarity),
            }
        )

    results.sort(
        key=lambda item: item["semantic_similarity"],
        reverse=True,
    )

    return results[:limit]


# ============================================================
# Hybrid ranking
# ============================================================

def _normalize_semantic_score(similarity: float) -> float:
    """
    Convert cosine similarity from approximately [-1, 1]
    into [0, 1].

    We also clamp because numerical/model behavior should never
    produce a score outside the expected range.
    """

    score = (similarity + 1.0) / 2.0

    return max(
        0.0,
        min(1.0, score),
    )


def _merge_candidates(
    lexical_results: list[dict],
    semantic_results: list[dict],
) -> dict[int, dict]:
    """
    Merge lexical and semantic candidate sets by memory ID.
    """

    candidates: dict[int, dict] = {}

    for result in lexical_results:
        memory_id = int(result["memory_id"])

        candidate = candidates.setdefault(
            memory_id,
            {
                "memory_id": memory_id,
                "lexical_score": 0.0,
                "semantic_similarity": 0.0,
            },
        )

        candidate["lexical_score"] = max(
            candidate["lexical_score"],
            float(result["lexical_score"]),
        )

        candidate["bm25_score"] = result.get(
            "bm25_score"
        )

    for result in semantic_results:
        memory_id = int(result["memory_id"])

        candidate = candidates.setdefault(
            memory_id,
            {
                "memory_id": memory_id,
                "lexical_score": 0.0,
                "semantic_similarity": 0.0,
            },
        )

        candidate["semantic_similarity"] = max(
            candidate["semantic_similarity"],
            float(result["semantic_similarity"]),
        )

    return candidates


def _build_hybrid_result(
    candidate: dict,
    memory: dict,
) -> dict:
    """
    Calculate the final hybrid score for one memory.
    """

    semantic_similarity = float(
        candidate.get(
            "semantic_similarity",
            0.0,
        )
    )

    semantic_score = _normalize_semantic_score(
        semantic_similarity
    )

    lexical_score = float(
        candidate.get(
            "lexical_score",
            0.0,
        )
    )

    importance_score = _importance_score(
        memory.get("importance")
    )

    recency_score = _recency_score(
        memory.get("updated_at")
        or memory.get("created_at")
    )

    access_score = _access_score(
        memory.get("access_count")
    )

    final_score = (
        semantic_score * SEMANTIC_WEIGHT
        + lexical_score * LEXICAL_WEIGHT
        + importance_score * IMPORTANCE_WEIGHT
        + recency_score * RECENCY_WEIGHT
        + access_score * ACCESS_WEIGHT
    )

    result = dict(memory)

    result.update(
        {
            "semantic_similarity": semantic_similarity,
            "semantic_score": semantic_score,
            "lexical_score": lexical_score,
            "importance_score": importance_score,
            "recency_score": recency_score,
            "access_score": access_score,
            "hybrid_score": final_score,
        }
    )

    return result


# ============================================================
# Public retrieval API
# ============================================================

def retrieve_memories(
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    candidate_limit: int | None = None,
    minimum_score: float = MINIMUM_SCORE,
) -> list[dict]:
    """
    Retrieve memories relevant to the CURRENT user query.

    This is the main function the rest of Ultron should call.

    Important:

        query should be the user's current message.

    Do NOT pass the entire conversation history here.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    query = query.strip()

    if not query:
        return []

    if limit <= 0:
        return []

    if candidate_limit is None:
        candidate_limit = max(
            limit * 3,
            12,
        )

    # --------------------------------------------------------
    # Retrieve using both systems.
    # --------------------------------------------------------

    lexical_results = lexical_search(
        query,
        limit=candidate_limit,
    )

    semantic_results = semantic_search(
        query,
        limit=candidate_limit,
    )

    candidates = _merge_candidates(
        lexical_results,
        semantic_results,
    )

    if not candidates:
        return []

    # --------------------------------------------------------
    # Load full memory records.
    # --------------------------------------------------------

    ranked: list[dict] = []

    for candidate in candidates.values():
        memory = get_memory(
            candidate["memory_id"]
        )

        if memory is None:
            continue

        if not bool(memory.get("active", 0)):
            continue

        result = _build_hybrid_result(
            candidate,
            memory,
        )

        # A direct lexical hit gets some protection from an
        # unusually weak semantic score.
        direct_lexical_hit = (
            result["lexical_score"] > 0.0
        )

        if (
            result["hybrid_score"] >= minimum_score
            or direct_lexical_hit
        ):
            ranked.append(result)

    # --------------------------------------------------------
    # Final ordering.
    # --------------------------------------------------------

    ranked.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    ranked = ranked[:limit]

    # --------------------------------------------------------
    # Access tracking.
    #
    # Only mark memories that actually survived retrieval.
    # Merely existing in the database does not count as access.
    # --------------------------------------------------------

    for result in ranked:
        try:
            mark_memory_accessed(
                result["id"]
            )
        except Exception:
            # Retrieval should never fail just because access
            # tracking encountered a database issue.
            pass

    return ranked


# ============================================================
# Context formatting
# ============================================================

def format_memories_for_context(
    memories: list[dict],
) -> str:
    """
    Convert retrieved memories into compact context for the LLM.

    The model receives the memory content and a small amount of
    metadata, but NOT the internal embedding vectors.
    """

    if not memories:
        return ""

    sections: list[str] = []

    for memory in memories:
        content = str(
            memory.get("content", "")
        ).strip()

        if not content:
            continue

        memory_type = str(
            memory.get(
                "memory_type",
                "unknown",
            )
        )

        sections.append(
            f"- [{memory_type}] {content}"
        )

    if not sections:
        return ""

    return "\n".join(sections)


# ============================================================
# Diagnostics
# ============================================================

def get_retrieval_info() -> dict:
    """
    Return configuration information useful for debugging.
    """

    return {
        "embedding_model": MODEL_NAME,
        "semantic_weight": SEMANTIC_WEIGHT,
        "lexical_weight": LEXICAL_WEIGHT,
        "importance_weight": IMPORTANCE_WEIGHT,
        "recency_weight": RECENCY_WEIGHT,
        "access_weight": ACCESS_WEIGHT,
        "minimum_score": MINIMUM_SCORE,
        "default_limit": DEFAULT_LIMIT,
        "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
    }


def self_test() -> None:
    """
    Basic retrieval-layer test.

    The test deliberately does NOT create permanent memories.
    """

    print("Retrieval self-test starting...")

    # --------------------------------------------------------
    # Test query processing.
    # --------------------------------------------------------

    query = (
        "What coding model did I use for Python?"
    )

    terms = _extract_search_terms(query)
    fts_query = _build_fts_query(query)

    if not terms:
        raise RuntimeError(
            "Search term extraction failed."
        )

    if not fts_query:
        raise RuntimeError(
            "FTS query construction failed."
        )

    print(f"SEARCH TERMS: {terms}")
    print(f"FTS QUERY: {fts_query}")

    # --------------------------------------------------------
    # Test metadata scoring.
    # --------------------------------------------------------

    if not 0.0 <= _importance_score(5) <= 1.0:
        raise RuntimeError(
            "Importance scoring failed."
        )

    if not 0.0 <= _access_score(10) <= 1.0:
        raise RuntimeError(
            "Access scoring failed."
        )

    recency = _recency_score(
        _utc_now().isoformat()
    )

    if not 0.9 <= recency <= 1.0:
        raise RuntimeError(
            "Recency scoring failed."
        )

    # --------------------------------------------------------
    # Test database retrieval against the current database.
    #
    # It is currently expected to be empty after our cleanup,
    # so this verifies that an empty memory store behaves safely.
    # --------------------------------------------------------

    results = retrieve_memories(
        "Ultron memory retrieval test"
    )

    print(
        f"ACTIVE MEMORY RESULTS: {len(results)}"
    )

    # --------------------------------------------------------
    # Test context formatting.
    # --------------------------------------------------------

    sample = [
        {
            "content": "Ultron uses semantic memory.",
            "memory_type": "fact",
        },
        {
            "content": "MiniLM provides embeddings.",
            "memory_type": "technical",
        },
    ]

    context = format_memories_for_context(
        sample
    )

    if "Ultron uses semantic memory." not in context:
        raise RuntimeError(
            "Context formatting failed."
        )

    print("Retrieval self-test passed.")


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":
    self_test()