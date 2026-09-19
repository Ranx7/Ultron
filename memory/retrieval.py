from .database import get_connection

import re
import unicodedata


# ============================================================
# CONFIGURATION
# ============================================================

MAX_QUERY_TERMS = 10

DEFAULT_MEMORY_LIMIT = 6
DEFAULT_CONVERSATION_LIMIT = 4
DEFAULT_RECENT_LIMIT = 6

# Messages within this period are considered part of the
# same active conversation session.
SESSION_GAP_MINUTES = 45


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been",
    "before", "being", "but", "by", "can", "could", "did", "do",
    "does", "doing", "for", "from", "had", "has", "have", "he",
    "her", "here", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "me", "more", "most", "my",
    "no", "not", "of", "on", "or", "our", "ours", "out", "over",
    "same", "she", "so", "some", "than", "that", "the", "their",
    "theirs", "them", "then", "there", "these", "they", "this",
    "those", "to", "too", "us", "very", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "yours",
}


LOW_INFORMATION_PHRASES = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "cool",
    "nice",
    "lol",
    "lmao",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize_text(text):

    if not text:
        return ""

    return unicodedata.normalize(
        "NFKC",
        str(text)
    ).strip()


# ============================================================
# QUERY TERMS
# ============================================================

def _extract_terms(text):

    text = _normalize_text(text).lower()

    raw_words = re.findall(
        r"[a-z0-9_]{3,}",
        text
    )

    terms = []
    seen = set()

    for word in raw_words:

        if word in STOPWORDS:
            continue

        if word in seen:
            continue

        seen.add(word)
        terms.append(word)

    return terms[:MAX_QUERY_TERMS]


def _fts_query(text):

    terms = _extract_terms(text)

    if not terms:
        return None

    return " OR ".join(
        f'"{term}"'
        for term in terms
    )


# ============================================================
# MEMORY TOPIC DETECTION
# ============================================================

def find_memory_topics(
    query,
    limit=DEFAULT_MEMORY_LIMIT
):
    """
    Determine whether the user's message actually overlaps
    with topics currently stored in long-term memory.

    This is intentionally different from simply searching
    memories and accepting whatever FTS happens to return.

    A query such as:

        "hi ultron"

    should not cause unrelated memories to be retrieved.

    A query such as:

        "what happened with my Python project?"

    can retrieve memories containing Python/project-related
    information.
    """

    terms = _extract_terms(query)

    if not terms:
        return []


    # --------------------------------------------------------
    # SEARCH FOR TOPIC OVERLAP
    # --------------------------------------------------------

    fts = _fts_query(query)

    if not fts:
        return []


    limit = max(
        1,
        min(int(limit), 50)
    )

    conn = get_connection()

    try:

        rows = conn.execute(
            """
            SELECT
                m.id,
                m.content,
                m.memory_type,
                m.importance,
                m.last_accessed,
                m.access_count,
                memories_fts.rank AS relevance

            FROM memories_fts

            JOIN memories AS m
                ON m.id = memories_fts.rowid

            WHERE memories_fts MATCH ?
              AND m.active = 1

            ORDER BY
                memories_fts.rank ASC,
                m.importance DESC,
                m.access_count DESC

            LIMIT ?
            """,
            (
                fts,
                limit
            )
        ).fetchall()


        results = [
            dict(row)
            for row in rows
        ]


        # ----------------------------------------------------
        # REQUIRE ACTUAL TOPIC OVERLAP
        # ----------------------------------------------------

        # FTS already performed the broad matching.
        # We now make sure at least one meaningful query
        # term is actually present in the memory text.

        verified = []

        for memory in results:

            memory_text = _normalize_text(
                memory["content"]
            ).lower()

            matched_terms = [
                term
                for term in terms
                if term in memory_text
            ]

            if not matched_terms:
                continue

            memory["matched_terms"] = matched_terms

            verified.append(
                memory
            )


        return verified

    finally:

        conn.close()


def has_memory_topic(query):
    """
    Return True only when the query appears to reference
    something that actually exists in long-term memory.
    """

    return bool(
        find_memory_topics(
            query,
            limit=1
        )
    )


# ============================================================
# MESSAGE IMPORTANCE FOR RETRIEVAL
# ============================================================

def is_low_information_message(text):

    """
    Detect messages where searching historical conversation
    is more likely to create noise than useful context.
    """

    normalized = _normalize_text(
        text
    ).lower()

    if not normalized:
        return True

    if normalized in LOW_INFORMATION_PHRASES:
        return True

    terms = _extract_terms(
        normalized
    )

    # Very short conversational messages.
    if len(terms) <= 1:
        return True

    return False


def should_search_old_conversation(text):

    """
    Historical conversation search should be selective.

    Greetings, acknowledgements, and tiny casual messages should
    not search the entire lifetime of the database.
    """

    if is_low_information_message(text):
        return False

    terms = _extract_terms(
        text
    )

    return len(terms) >= 2


# ============================================================
# SEARCH LONG-TERM MEMORIES
# ============================================================

def search_memories(
    query,
    limit=DEFAULT_MEMORY_LIMIT
):

    """
    Search long-term memories only when the query has
    meaningful overlap with stored memory topics.
    """

    return find_memory_topics(
        query,
        limit=limit
    )


# ============================================================
# SEARCH OLD CONVERSATIONS
# ============================================================

def search_conversation(
    query,
    limit=DEFAULT_CONVERSATION_LIMIT
):

    if not should_search_old_conversation(query):
        return []

    fts = _fts_query(query)

    if not fts:
        return []

    limit = max(
        1,
        min(int(limit), 50)
    )

    conn = get_connection()

    try:

        rows = conn.execute(
            """
            SELECT
                m.id,
                m.role,
                m.content,
                m.created_at,
                messages_fts.rank AS relevance

            FROM messages_fts

            JOIN messages AS m
                ON m.id = messages_fts.rowid

            WHERE messages_fts MATCH ?

            ORDER BY
                messages_fts.rank ASC

            LIMIT ?
            """,
            (
                fts,
                limit
            )
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        conn.close()


# ============================================================
# RECENT SESSION
# ============================================================

def get_recent_messages(
    limit=DEFAULT_RECENT_LIMIT,
    session_minutes=SESSION_GAP_MINUTES
):

    limit = max(
        1,
        min(int(limit), 50)
    )

    session_minutes = max(
        1,
        int(session_minutes)
    )

    conn = get_connection()

    try:

        rows = conn.execute(
            """
            SELECT
                role,
                content,
                created_at

            FROM messages

            WHERE created_at >= datetime(
                'now',
                ?
            )

            ORDER BY id ASC

            LIMIT ?
            """,
            (
                f"-{session_minutes} minutes",
                limit
            )
        ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    finally:

        conn.close()


# ============================================================
# LAST MESSAGE
# ============================================================

def get_last_message():

    conn = get_connection()

    try:

        row = conn.execute(
            """
            SELECT
                id,
                role,
                content,
                created_at

            FROM messages

            ORDER BY id DESC

            LIMIT 1
            """
        ).fetchone()

        if not row:
            return None

        return dict(row)

    finally:

        conn.close()


# ============================================================
# TOUCH ACCESSED MEMORIES
# ============================================================

def touch_memories(ids):

    if not ids:
        return

    unique_ids = list(
        dict.fromkeys(ids)
    )

    conn = get_connection()

    try:

        conn.executemany(
            """
            UPDATE memories

            SET
                last_accessed = CURRENT_TIMESTAMP,
                access_count = access_count + 1

            WHERE id = ?
            """,
            [
                (memory_id,)
                for memory_id in unique_ids
            ]
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# DATABASE COUNTS
# ============================================================

def get_counts():

    conn = get_connection()

    try:

        messages = conn.execute(
            """
            SELECT COUNT(*)
            FROM messages
            """
        ).fetchone()[0]

        memories = conn.execute(
            """
            SELECT COUNT(*)
            FROM memories
            WHERE active = 1
            """
        ).fetchone()[0]

        return (
            messages,
            memories
        )

    finally:

        conn.close()


# ============================================================
# COMBINED RETRIEVAL
# ============================================================

def retrieve_context(
    query,
    memory_limit=DEFAULT_MEMORY_LIMIT,
    conversation_limit=DEFAULT_CONVERSATION_LIMIT,
    recent_limit=DEFAULT_RECENT_LIMIT
):

    memories = search_memories(
        query,
        limit=memory_limit
    )

    conversation = search_conversation(
        query,
        limit=conversation_limit
    )

    recent = get_recent_messages(
        limit=recent_limit
    )

    if memories:

        touch_memories(
            [
                memory["id"]
                for memory in memories
            ]
        )

    return {
        "memories": memories,
        "conversation": conversation,
        "recent": recent,
    }