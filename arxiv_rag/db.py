"""SQLite schema helpers for arxiv-rag."""

from __future__ import annotations

import sqlite3
from array import array
from math import sqrt
from pathlib import Path
from typing import Sequence

from arxiv_rag.chunk_ids import compute_chunk_uid

_PAPERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    doc_id TEXT,
    title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    categories TEXT,
    published_date TEXT,
    pdf_path TEXT,
    total_pages INTEGER,
    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    source_type TEXT DEFAULT 'arxiv',
    UNIQUE(doc_id)
);
"""

_CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_uid TEXT NOT NULL UNIQUE,
    paper_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    token_count INTEGER,
    embedding BLOB,
    UNIQUE(doc_id, page_number, chunk_index),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);
"""

_CHUNKS_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='chunk_id'
);
"""

_CHUNKS_AI_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
"""

_CHUNKS_AD_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.chunk_id, old.text);
END;
"""

_CHUNKS_AU_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.chunk_id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
"""


def ensure_papers_schema(
    conn: sqlite3.Connection,
    *,
    create_if_missing: bool = True,
) -> None:
    """Ensure the papers table exists and includes a doc_id column.

    Args:
        conn: Open SQLite connection.
        create_if_missing: When False, raise if the papers table is missing.
    Raises:
        ValueError: If the papers table is missing and create_if_missing is False.
    """

    if create_if_missing:
        conn.execute(_PAPERS_TABLE_SQL)
    elif not _table_exists(conn, "papers"):
        raise ValueError("papers table not found; run download.py with --db first.")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(papers)")}
    if "doc_id" not in columns:
        conn.execute("ALTER TABLE papers ADD COLUMN doc_id TEXT")


def ensure_chunks_schema(conn: sqlite3.Connection) -> None:
    """Ensure the chunks and FTS tables exist."""

    conn.execute(_CHUNKS_TABLE_SQL)
    _ensure_chunk_uid_column(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_paper_id_idx ON chunks(paper_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS chunks_chunk_uid_uidx ON chunks(chunk_uid)"
    )
    conn.execute(_CHUNKS_FTS_SQL)
    conn.execute(_CHUNKS_AI_TRIGGER_SQL)
    conn.execute(_CHUNKS_AD_TRIGGER_SQL)
    conn.execute(_CHUNKS_AU_TRIGGER_SQL)


def _ensure_chunk_uid_column(conn: sqlite3.Connection) -> None:
    """Ensure the chunks table has a populated chunk_uid column."""

    columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    if "chunk_uid" not in columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN chunk_uid TEXT")

    _backfill_chunk_uids(conn)


def _backfill_chunk_uids(conn: sqlite3.Connection) -> None:
    """Backfill chunk_uids for rows missing a value."""

    rows = conn.execute(
        """
        SELECT chunk_id, doc_id, page_number, chunk_index, char_start, char_end
        FROM chunks
        WHERE chunk_uid IS NULL OR chunk_uid = ''
        """
    ).fetchall()
    if not rows:
        return

    updates: list[tuple[str, int]] = []
    for chunk_id, doc_id, page_number, chunk_index, char_start, char_end in rows:
        chunk_uid = compute_chunk_uid(
            doc_id=doc_id,
            page_number=page_number,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
        )
        updates.append((chunk_uid, chunk_id))

    conn.executemany(
        "UPDATE chunks SET chunk_uid = ? WHERE chunk_id = ?",
        updates,
    )


def ensure_papers_db(db_path: Path) -> None:
    """Create the SQLite database and papers table if missing.

    Args:
        db_path: SQLite database path.
    Raises:
        sqlite3.Error: If the database cannot be initialized.
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        ensure_papers_schema(conn, create_if_missing=True)
        conn.commit()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def normalize_embedding(vector: Sequence[float]) -> list[float]:
    """Normalize an embedding vector to unit length.

    Args:
        vector: Embedding values.
    Returns:
        Normalized vector as a list of floats.
    Raises:
        ValueError: If the vector is empty.
    """

    if not vector:
        raise ValueError("embedding vector must be non-empty")

    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def serialize_embedding(vector: Sequence[float]) -> bytes:
    """Serialize an embedding vector as float32 bytes.

    Args:
        vector: Embedding values (ideally normalized already).
    Returns:
        Bytes representation of the float32 array.
    Raises:
        ValueError: If the vector is empty.
    """

    if not vector:
        raise ValueError("embedding vector must be non-empty")
    return array("f", vector).tobytes()


def deserialize_embedding(blob: bytes) -> list[float]:
    """Deserialize float32 bytes into an embedding vector.

    Args:
        blob: Raw bytes from SQLite.
    Returns:
        Embedding vector as a list of floats.
    """

    values = array("f")
    values.frombytes(blob)
    return list(values)


def deserialize_embedding_array(blob: bytes) -> array:
    """Deserialize float32 bytes into an array('f') embedding.

    Args:
        blob: Raw bytes from SQLite.
    Returns:
        Embedding array for efficient numeric operations.
    """

    values = array("f")
    values.frombytes(blob)
    return values
