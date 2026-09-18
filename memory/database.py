import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "ultron_memory.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    # Good defaults for a local, single-user SQLite database.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    return conn


def initialize_database():
    conn = get_connection()

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        memory_type TEXT NOT NULL DEFAULT 'fact',
        importance INTEGER NOT NULL DEFAULT 1,
        source_message_id INTEGER,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_accessed TEXT,
        access_count INTEGER NOT NULL DEFAULT 0,
        active INTEGER NOT NULL DEFAULT 1,
        content_hash TEXT NOT NULL UNIQUE,
        FOREIGN KEY(source_message_id) REFERENCES messages(id)
    );

    CREATE INDEX IF NOT EXISTS idx_messages_created
        ON messages(created_at);

    CREATE INDEX IF NOT EXISTS idx_memories_active_importance
        ON memories(active, importance DESC);

    CREATE INDEX IF NOT EXISTS idx_memories_type
        ON memories(memory_type);

    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(content, content='messages', content_rowid='id');

    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(content, content='memories', content_rowid='id');

    CREATE TRIGGER IF NOT EXISTS messages_ai
    AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content)
        VALUES (new.id, new.content);
    END;

    CREATE TRIGGER IF NOT EXISTS messages_ad
    AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_ai
    AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, content)
        VALUES (new.id, new.content);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_ad
    AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES ('delete', old.id, old.content);
    END;
    """)

    conn.commit()
    conn.close()


def add_message(role, content):
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO messages(role, content) VALUES (?, ?)",
            (role, content),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def add_or_update_memory(
    content,
    memory_type="fact",
    importance=1,
    source_message_id=None,
):
    import hashlib

    normalized = " ".join(content.strip().split())
    content_hash = hashlib.sha256(
        normalized.casefold().encode("utf-8")
    ).hexdigest()

    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM memories WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()

        if existing:
            conn.execute(
                """
                UPDATE memories
                SET updated_at=CURRENT_TIMESTAMP,
                    importance=MAX(importance, ?),
                    active=1
                WHERE id=?
                """,
                (importance, existing["id"]),
            )
            conn.commit()
            return existing["id"], False

        cur = conn.execute(
            """
            INSERT INTO memories(
                content, memory_type, importance, source_message_id, content_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                normalized,
                memory_type,
                importance,
                source_message_id,
                content_hash,
            ),
        )
        conn.commit()
        return cur.lastrowid, True
    finally:
        conn.close()


def touch_memories(ids):
    if not ids:
        return

    conn = get_connection()
    try:
        conn.executemany(
            """
            UPDATE memories
            SET last_accessed=CURRENT_TIMESTAMP,
                access_count=access_count + 1
            WHERE id=?
            """,
            [(memory_id,) for memory_id in ids],
        )
        conn.commit()
    finally:
        conn.close()
