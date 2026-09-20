from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
import uuid
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DB_PATH = Path(
    os.getenv(
        "ULTRON_DB_PATH",
        str(DATA_DIR / "ultron_memory.db"),
    )
)

DEFAULT_SESSION_ID = "default"


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_VERSION = 2

ALLOWED_ROLES = {
    "system",
    "user",
    "assistant",
    "tool",
}


# ============================================================
# NORMALIZATION / HASHING
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for comparison/deduplication.

    This does NOT modify the text we store.
    It only creates a canonical representation for hashing.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text.strip()


def content_hash(text: str) -> str:
    """
    Stable SHA-256 hash used to deduplicate memory content.
    """
    normalized = normalize_text(text).casefold()
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# EMBEDDING SERIALIZATION
# ============================================================

def serialize_embedding(values: Sequence[float]) -> bytes:
    """
    Store an embedding as compact float32 binary data.

    This keeps the database much smaller than storing JSON.
    """
    if not values:
        raise ValueError("embedding cannot be empty")

    values = [float(value) for value in values]

    return struct.pack(
        "<" + ("f" * len(values)),
        *values,
    )


def deserialize_embedding(blob: bytes, dimensions: int) -> list[float]:
    """
    Convert float32 binary embedding data back into Python floats.
    """
    if not blob:
        raise ValueError("embedding blob is empty")

    expected_size = dimensions * 4

    if len(blob) != expected_size:
        raise ValueError(
            f"embedding blob size mismatch: "
            f"expected {expected_size} bytes, got {len(blob)}"
        )

    return list(
        struct.unpack(
            "<" + ("f" * dimensions),
            blob,
        )
    )


# ============================================================
# CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Open a new SQLite connection.

    Every operation gets its own connection. This works cleanly
    with Flask/background threads and avoids sharing SQLite
    connections across threads.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=5.0,
    )

    conn.row_factory = sqlite3.Row

    # Foreign keys must be enabled per connection.
    conn.execute("PRAGMA foreign_keys = ON")

    # Better behavior for concurrent reads/writes.
    conn.execute("PRAGMA journal_mode = WAL")

    # Good durability/performance balance for a local assistant.
    conn.execute("PRAGMA synchronous = NORMAL")

    # Avoid immediate "database is locked" errors.
    conn.execute("PRAGMA busy_timeout = 5000")

    # Keep SQLite's temporary structures in memory.
    conn.execute("PRAGMA temp_store = MEMORY")

    # ~8 MB SQLite page cache.
    conn.execute("PRAGMA cache_size = -8192")

    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database() -> None:
    """
    Create the complete fresh memory database.

    The active database architecture is:

        sessions
            |
        messages
            |
        memories
            |
        memory_embeddings

    FTS5 indexes:
        messages_fts
        memories_fts
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_connection()

    try:
        conn.executescript(
            """
            -- ==================================================
            -- SESSIONS
            -- ==================================================

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_active_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );


            -- ==================================================
            -- RAW CONVERSATION HISTORY
            -- ==================================================

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id TEXT NOT NULL,

                role TEXT NOT NULL,

                content TEXT NOT NULL,

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(id)
                    ON DELETE CASCADE
            );


            -- ==================================================
            -- LONG-TERM MEMORY
            -- ==================================================

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                content TEXT NOT NULL,

                memory_type TEXT NOT NULL DEFAULT 'fact',

                importance REAL NOT NULL DEFAULT 1.0,

                source_message_id INTEGER,

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                last_accessed TEXT,

                access_count INTEGER NOT NULL DEFAULT 0,

                active INTEGER NOT NULL DEFAULT 1,

                content_hash TEXT NOT NULL UNIQUE,

                FOREIGN KEY (source_message_id)
                    REFERENCES messages(id)
                    ON DELETE SET NULL,

                CHECK (importance >= 0 AND importance <= 5),

                CHECK (active IN (0, 1))
            );


            -- ==================================================
            -- SEMANTIC EMBEDDINGS
            -- ==================================================

            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id INTEGER NOT NULL,

                model TEXT NOT NULL,

                dimensions INTEGER NOT NULL,

                embedding BLOB NOT NULL,

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (memory_id, model),

                FOREIGN KEY (memory_id)
                    REFERENCES memories(id)
                    ON DELETE CASCADE,

                CHECK (dimensions > 0)
            );


            -- ==================================================
            -- NORMAL INDEXES
            -- ==================================================

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);

            CREATE INDEX IF NOT EXISTS idx_messages_created
                ON messages(created_at);

            CREATE INDEX IF NOT EXISTS idx_memories_active
                ON memories(active, importance DESC);

            CREATE INDEX IF NOT EXISTS idx_memories_updated
                ON memories(updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(memory_type);

            CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model
                ON memory_embeddings(model);


            -- ==================================================
            -- MESSAGE FTS5
            -- ==================================================

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(
                content,
                content='messages',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );


            -- ==================================================
            -- MEMORY FTS5
            -- ==================================================

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(
                content,
                content='memories',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );


            -- ==================================================
            -- MESSAGE INSERT TRIGGER
            -- ==================================================

            CREATE TRIGGER IF NOT EXISTS messages_ai
            AFTER INSERT ON messages
            BEGIN
                INSERT INTO messages_fts(
                    rowid,
                    content
                )
                VALUES (
                    new.id,
                    new.content
                );
            END;


            -- ==================================================
            -- MESSAGE UPDATE TRIGGER
            -- ==================================================

            CREATE TRIGGER IF NOT EXISTS messages_au
            AFTER UPDATE ON messages
            BEGIN

                INSERT INTO messages_fts(
                    messages_fts,
                    rowid,
                    content
                )
                VALUES (
                    'delete',
                    old.id,
                    old.content
                );

                INSERT INTO messages_fts(
                    rowid,
                    content
                )
                VALUES (
                    new.id,
                    new.content
                );

            END;


            -- ==================================================
            -- MESSAGE DELETE TRIGGER
            -- ==================================================

            CREATE TRIGGER IF NOT EXISTS messages_ad
            AFTER DELETE ON messages
            BEGIN

                INSERT INTO messages_fts(
                    messages_fts,
                    rowid,
                    content
                )
                VALUES (
                    'delete',
                    old.id,
                    old.content
                );

            END;


            -- ==================================================
            -- MEMORY INSERT TRIGGER
            -- ==================================================

            CREATE TRIGGER IF NOT EXISTS memories_ai
            AFTER INSERT ON memories
            BEGIN

                INSERT INTO memories_fts(
                    rowid,
                    content
                )
                VALUES (
                    new.id,
                    new.content
                );

            END;


            -- ==================================================
            -- MEMORY UPDATE TRIGGER
            -- ==================================================

            CREATE TRIGGER IF NOT EXISTS memories_au
            AFTER UPDATE ON memories
            BEGIN

                INSERT INTO memories_fts(
                    memories_fts,
                    rowid,
                    content
                )
                VALUES (
                    'delete',
                    old.id,
                    old.content
                );

                INSERT INTO memories_fts(
                    rowid,
                    content
                )
                VALUES (
                    new.id,
                    new.content
                );

            END;


            -- ==================================================
            -- MEMORY DELETE TRIGGER
            -- ==================================================

            CREATE TRIGGER IF NOT EXISTS memories_ad
            AFTER DELETE ON memories
            BEGIN

                INSERT INTO memories_fts(
                    memories_fts,
                    rowid,
                    content
                )
                VALUES (
                    'delete',
                    old.id,
                    old.content
                );

            END;
            """
        )

        conn.execute(
            f"PRAGMA user_version = {DATABASE_VERSION}"
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# FTS REBUILD
# ============================================================

def rebuild_fts() -> None:
    """
    Completely rebuild both FTS5 indexes from their source tables.

    Useful after restoring/moving a database or recovering from
    an old index that may have become inconsistent.
    """
    initialize_database()

    conn = get_connection()

    try:
        conn.execute(
            "INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')"
        )

        conn.execute(
            "INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')"
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# SESSION FUNCTIONS
# ============================================================

def create_session(
    session_id: str | None = None,
) -> str:
    """
    Create a new session and return its ID.
    """
    initialize_database()

    if session_id is None:
        session_id = str(uuid.uuid4())

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT INTO sessions (
                id
            )
            VALUES (?)
            """,
            (session_id,),
        )

        conn.commit()

        return session_id

    finally:
        conn.close()


def ensure_session(
    session_id: str = DEFAULT_SESSION_ID,
) -> str:
    """
    Make sure a session exists.
    """
    initialize_database()

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (
                id
            )
            VALUES (?)
            """,
            (session_id,),
        )

        conn.execute(
            """
            UPDATE sessions
            SET last_active_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (session_id,),
        )

        conn.commit()

        return session_id

    finally:
        conn.close()


def touch_session(
    session_id: str,
) -> None:
    """
    Update session activity timestamp.
    """
    conn = get_connection()

    try:
        conn.execute(
            """
            UPDATE sessions
            SET last_active_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (session_id,),
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# MESSAGE FUNCTIONS
# ============================================================

def add_message(
    role: str,
    content: str,
    session_id: str = DEFAULT_SESSION_ID,
    created_at: str | None = None,
) -> int:
    """
    Store a raw conversation message.

    Returns:
        message ID
    """
    if role not in ALLOWED_ROLES:
        raise ValueError(
            f"Invalid role '{role}'. "
            f"Allowed roles: {sorted(ALLOWED_ROLES)}"
        )

    if not isinstance(content, str):
        raise TypeError("content must be a string")

    if not content.strip():
        raise ValueError("content cannot be empty")

    ensure_session(session_id)

    conn = get_connection()

    try:
        if created_at is None:
            cursor = conn.execute(
                """
                INSERT INTO messages (
                    session_id,
                    role,
                    content
                )
                VALUES (?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                ),
            )
        else:
            cursor = conn.execute(
                """
                INSERT INTO messages (
                    session_id,
                    role,
                    content,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    created_at,
                ),
            )

        conn.execute(
            """
            UPDATE sessions
            SET last_active_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (session_id,),
        )

        conn.commit()

        return int(cursor.lastrowid)

    finally:
        conn.close()


def get_message(
    message_id: int,
) -> dict | None:
    """
    Retrieve one message.
    """
    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT
                id,
                session_id,
                role,
                content,
                created_at
            FROM messages
            WHERE id = ?
            """,
            (message_id,),
        ).fetchone()

        return dict(row) if row else None

    finally:
        conn.close()


def get_recent_messages(
    session_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Retrieve recent conversation messages.

    Session filtering is optional for backwards compatibility.
    """
    if limit <= 0:
        return []

    conn = get_connection()

    try:
        if session_id is None:
            rows = conn.execute(
                """
                SELECT
                    id,
                    session_id,
                    role,
                    content,
                    created_at
                FROM messages
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                    id,
                    session_id,
                    role,
                    content,
                    created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    session_id,
                    limit,
                ),
            ).fetchall()

        rows = list(reversed(rows))

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


def count_messages(
    session_id: str | None = None,
) -> int:
    """
    Count stored messages.
    """
    conn = get_connection()

    try:
        if session_id is None:
            row = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM messages
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        return int(row[0])

    finally:
        conn.close()


# ============================================================
# MEMORY FUNCTIONS
# ============================================================

def add_or_update_memory(
    content: str,
    memory_type: str = "fact",
    importance: float = 1.0,
    source_message_id: int | None = None,
) -> tuple[int, bool]:
    """
    Insert a new long-term memory or merge it with an exact
    existing memory.

    Returns:
        (memory_id, created)

    created=True
        New memory inserted.

    created=False
        Existing memory updated/merged.
    """
    if not isinstance(content, str):
        raise TypeError("content must be a string")

    if not content.strip():
        raise ValueError("memory content cannot be empty")

    importance = float(importance)

    if importance < 0 or importance > 5:
        raise ValueError("importance must be between 0 and 5")

    memory_hash = content_hash(content)

    conn = get_connection()

    try:
        existing = conn.execute(
            """
            SELECT id
            FROM memories
            WHERE content_hash = ?
            """,
            (memory_hash,),
        ).fetchone()

        if existing:

            memory_id = int(existing["id"])

            conn.execute(
                """
                UPDATE memories
                SET
                    memory_type = ?,
                    importance = MAX(importance, ?),
                    source_message_id =
                        COALESCE(source_message_id, ?),
                    updated_at = CURRENT_TIMESTAMP,
                    active = 1
                WHERE id = ?
                """,
                (
                    memory_type,
                    importance,
                    source_message_id,
                    memory_id,
                ),
            )

            conn.commit()

            return memory_id, False

        cursor = conn.execute(
            """
            INSERT INTO memories (
                content,
                memory_type,
                importance,
                source_message_id,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                content,
                memory_type,
                importance,
                source_message_id,
                memory_hash,
            ),
        )

        conn.commit()

        return int(cursor.lastrowid), True

    finally:
        conn.close()


def update_memory(
    memory_id: int,
    content: str | None = None,
    memory_type: str | None = None,
    importance: float | None = None,
    active: int | None = None,
) -> None:
    """
    Update an existing memory.

    Changing content also changes its deduplication hash.

    Raises:
        ValueError if the new content already belongs to another
        memory.
    """
    if (
        content is None
        and memory_type is None
        and importance is None
        and active is None
    ):
        return

    conn = get_connection()

    try:
        current = conn.execute(
            """
            SELECT *
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()

        if current is None:
            raise ValueError(
                f"Memory {memory_id} does not exist"
            )

        new_content = (
            content
            if content is not None
            else current["content"]
        )

        new_type = (
            memory_type
            if memory_type is not None
            else current["memory_type"]
        )

        new_importance = (
            float(importance)
            if importance is not None
            else float(current["importance"])
        )

        new_active = (
            int(active)
            if active is not None
            else int(current["active"])
        )

        if not 0 <= new_importance <= 5:
            raise ValueError(
                "importance must be between 0 and 5"
            )

        if new_active not in (0, 1):
            raise ValueError(
                "active must be 0 or 1"
            )

        new_hash = content_hash(new_content)

        collision = conn.execute(
            """
            SELECT id
            FROM memories
            WHERE content_hash = ?
              AND id != ?
            """,
            (
                new_hash,
                memory_id,
            ),
        ).fetchone()

        if collision:
            raise ValueError(
                "Another memory already contains the same content"
            )

        conn.execute(
            """
            UPDATE memories
            SET
                content = ?,
                memory_type = ?,
                importance = ?,
                active = ?,
                content_hash = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                new_content,
                new_type,
                new_importance,
                new_active,
                new_hash,
                memory_id,
            ),
        )

        conn.commit()

    finally:
        conn.close()


def archive_memory(
    memory_id: int,
) -> None:
    """
    Mark a memory inactive without deleting it.
    """
    conn = get_connection()

    try:
        conn.execute(
            """
            UPDATE memories
            SET
                active = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (memory_id,),
        )

        conn.commit()

    finally:
        conn.close()


def restore_memory(
    memory_id: int,
) -> None:
    """
    Restore an archived memory.
    """
    conn = get_connection()

    try:
        conn.execute(
            """
            UPDATE memories
            SET
                active = 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (memory_id,),
        )

        conn.commit()

    finally:
        conn.close()


def delete_memory(
    memory_id: int,
) -> None:
    """
    Permanently delete one memory.

    Its embedding is automatically deleted through
    ON DELETE CASCADE.
    """
    conn = get_connection()

    try:
        conn.execute(
            """
            DELETE FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )

        conn.commit()

    finally:
        conn.close()


def get_memory(
    memory_id: int,
) -> dict | None:
    """
    Retrieve one memory.
    """
    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT
                id,
                content,
                memory_type,
                importance,
                source_message_id,
                created_at,
                updated_at,
                last_accessed,
                access_count,
                active,
                content_hash
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()

        return dict(row) if row else None

    finally:
        conn.close()


def get_active_memories(
    limit: int | None = None,
) -> list[dict]:
    """
    Retrieve active long-term memories.

    This is useful for semantic retrieval where MiniLM needs
    access to memory candidates.
    """
    conn = get_connection()

    try:
        if limit is None:
            rows = conn.execute(
                """
                SELECT
                    id,
                    content,
                    memory_type,
                    importance,
                    source_message_id,
                    created_at,
                    updated_at,
                    last_accessed,
                    access_count,
                    active,
                    content_hash
                FROM memories
                WHERE active = 1
                ORDER BY
                    importance DESC,
                    updated_at DESC
                """
            ).fetchall()

        else:
            rows = conn.execute(
                """
                SELECT
                    id,
                    content,
                    memory_type,
                    importance,
                    source_message_id,
                    created_at,
                    updated_at,
                    last_accessed,
                    access_count,
                    active,
                    content_hash
                FROM memories
                WHERE active = 1
                ORDER BY
                    importance DESC,
                    updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


def count_memories(
    active_only: bool = True,
) -> int:
    """
    Count long-term memories.
    """
    conn = get_connection()

    try:
        if active_only:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM memories
                WHERE active = 1
                """
            ).fetchone()

        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()

        return int(row[0])

    finally:
        conn.close()


# ============================================================
# MEMORY ACCESS TRACKING
# ============================================================


def mark_memory_accessed(memory_id: int) -> None:
    """
    Record that a memory was actually retrieved and used.
    """

    conn = get_connection()

    try:
        conn.execute(
            """
            UPDATE memories
            SET
                last_accessed = CURRENT_TIMESTAMP,
                access_count = access_count + 1
            WHERE id = ?
              AND active = 1
            """,
            (memory_id,),
        )

        conn.commit()

    finally:
        conn.close()



def touch_memories(
    memory_ids: Iterable[int],
) -> None:
    """
    Mark memories as accessed.

    This is used later for relevance/retention logic.
    """
    ids = list(memory_ids)

    if not ids:
        return

    conn = get_connection()

    try:
        conn.executemany(
            """
            UPDATE memories
            SET
                last_accessed = CURRENT_TIMESTAMP,
                access_count = access_count + 1
            WHERE id = ?
              AND active = 1
            """,
            [
                (int(memory_id),)
                for memory_id in ids
            ],
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

def upsert_memory_embedding(
    memory_id: int,
    embedding: Sequence[float],
    model: str,
) -> None:
    """
    Insert or replace a memory embedding.

    Example model:
        sentence-transformers/all-MiniLM-L6-v2
    """
    if not model or not model.strip():
        raise ValueError("model cannot be empty")

    values = [float(value) for value in embedding]

    if not values:
        raise ValueError("embedding cannot be empty")

    blob = serialize_embedding(values)
    dimensions = len(values)

    conn = get_connection()

    try:
        memory_exists = conn.execute(
            """
            SELECT 1
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()

        if memory_exists is None:
            raise ValueError(
                f"Memory {memory_id} does not exist"
            )

        conn.execute(
            """
            INSERT INTO memory_embeddings (
                memory_id,
                model,
                dimensions,
                embedding
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(memory_id, model)
            DO UPDATE SET
                dimensions = excluded.dimensions,
                embedding = excluded.embedding,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                memory_id,
                model,
                dimensions,
                blob,
            ),
        )

        conn.commit()

    finally:
        conn.close()


def get_memory_embedding(
    memory_id: int,
    model: str,
) -> list[float] | None:
    """
    Retrieve one memory's embedding.
    """
    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT
                embedding,
                dimensions
            FROM memory_embeddings
            WHERE memory_id = ?
              AND model = ?
            """,
            (
                memory_id,
                model,
            ),
        ).fetchone()

        if row is None:
            return None

        return deserialize_embedding(
            row["embedding"],
            int(row["dimensions"]),
        )

    finally:
        conn.close()


def get_active_memory_embeddings(
    model: str,
) -> list[dict]:
    """
    Retrieve embeddings for all active memories.

    Returned format:

        {
            "memory_id": int,
            "dimensions": int,
            "embedding": bytes
        }

    Keeping the blob here lets the retrieval layer decide whether
    to decode using NumPy or another optimized method.
    """
    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT
                e.memory_id,
                e.dimensions,
                e.embedding
            FROM memory_embeddings AS e
            INNER JOIN memories AS m
                ON m.id = e.memory_id
            WHERE m.active = 1
              AND e.model = ?
            """,
            (model,),
        ).fetchall()

        return [
            {
                "memory_id": int(row["memory_id"]),
                "dimensions": int(row["dimensions"]),
                "embedding": bytes(row["embedding"]),
            }
            for row in rows
        ]

    finally:
        conn.close()


def delete_memory_embedding(
    memory_id: int,
    model: str | None = None,
) -> None:
    """
    Delete one embedding or all embeddings belonging to a memory.
    """
    conn = get_connection()

    try:
        if model is None:
            conn.execute(
                """
                DELETE FROM memory_embeddings
                WHERE memory_id = ?
                """,
                (memory_id,),
            )

        else:
            conn.execute(
                """
                DELETE FROM memory_embeddings
                WHERE memory_id = ?
                  AND model = ?
                """,
                (
                    memory_id,
                    model,
                ),
            )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# GLOBAL COUNTS / STATUS
# ============================================================

def get_counts() -> tuple[int, int]:
    """
    Backwards-compatible helper.

    Returns:
        (message_count, active_memory_count)
    """
    conn = get_connection()

    try:
        messages = conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        memories = conn.execute(
            """
            SELECT COUNT(*)
            FROM memories
            WHERE active = 1
            """
        ).fetchone()[0]

        return (
            int(messages),
            int(memories),
        )

    finally:
        conn.close()


def get_database_info() -> dict:
    """
    Small diagnostic snapshot useful during development.
    """
    initialize_database()

    conn = get_connection()

    try:
        user_version = conn.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        page_count = conn.execute(
            "PRAGMA page_count"
        ).fetchone()[0]

        page_size = conn.execute(
            "PRAGMA page_size"
        ).fetchone()[0]

        messages = conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

        memories = conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]

        active_memories = conn.execute(
            """
            SELECT COUNT(*)
            FROM memories
            WHERE active = 1
            """
        ).fetchone()[0]

        embeddings = conn.execute(
            """
            SELECT COUNT(*)
            FROM memory_embeddings
            """
        ).fetchone()[0]

        return {
            "database_path": str(DB_PATH),
            "database_version": int(user_version),
            "messages": int(messages),
            "memories": int(memories),
            "active_memories": int(active_memories),
            "embeddings": int(embeddings),
            "database_size_bytes": int(
                page_count * page_size
            ),
        }

    finally:
        conn.close()


# ============================================================
# SELF-TEST
# ============================================================

def self_test() -> None:
    """
    Basic database-layer sanity test.

    This does not modify an existing database's contents except
    for creating test rows, so use only on the fresh database.
    """
    initialize_database()

    session_id = f"self-test-{uuid.uuid4()}"

    ensure_session(session_id)

    message_id = add_message(
        "user",
        "Database self test.",
        session_id=session_id,
    )

    memory_id, created = add_or_update_memory(
        "This is a database self test memory.",
        memory_type="test",
        importance=1,
        source_message_id=message_id,
    )

    # Test deduplication.
    memory_id_2, created_2 = add_or_update_memory(
        "This is a database self test memory.",
        memory_type="test",
        importance=2,
        source_message_id=message_id,
    )

    assert memory_id == memory_id_2
    assert created is True
    assert created_2 is False

    # Test embedding storage.
    test_embedding = [0.1, 0.2, 0.3, 0.4]

    upsert_memory_embedding(
        memory_id,
        test_embedding,
        "self-test-model",
    )

    loaded = get_memory_embedding(
        memory_id,
        "self-test-model",
    )

    assert loaded is not None
    assert len(loaded) == 4

    for a, b in zip(test_embedding, loaded):
        assert abs(a - b) < 0.00001

    # Test access tracking.
    touch_memories([memory_id])

    # Test archive / restore.
    archive_memory(memory_id)

    archived = get_memory(memory_id)

    assert archived is not None
    assert archived["active"] == 0

    restore_memory(memory_id)

    restored = get_memory(memory_id)

    assert restored is not None
    assert restored["active"] == 1

    # Test FTS rebuild.
    rebuild_fts()

    print("Database self-test passed.")
    print(get_database_info())


if __name__ == "__main__":
    self_test()