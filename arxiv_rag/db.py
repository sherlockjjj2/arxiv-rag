"""SQLite schema helpers for arxiv-rag."""

from __future__ import annotations

import sqlite3
from pathlib import Path

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
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_paper_id_idx ON chunks(paper_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id)")
    conn.execute(_CHUNKS_FTS_SQL)
    conn.execute(_CHUNKS_AI_TRIGGER_SQL)
    conn.execute(_CHUNKS_AD_TRIGGER_SQL)
    conn.execute(_CHUNKS_AU_TRIGGER_SQL)


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
