"""SQLite schema helpers for arxiv-rag."""

from __future__ import annotations

import json
import sqlite3
from array import array
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Sequence

from arxiv_rag.arxiv_ids import normalize_base_id_for_lookup
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

_QUERY_LOG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_log (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    retrieved_chunks TEXT,
    answer TEXT,
    latency_ms INTEGER,
    model TEXT,
    feedback INTEGER
);
"""


@dataclass(frozen=True)
class QueryLogEntry:
    """Query log payload for SQLite storage.

    Args:
        query_text: User query text.
        retrieved_chunks: Retrieved chunk UID list.
        answer: Generated answer text (if any).
        latency_ms: Total latency in milliseconds.
        model: Model identifier string.
    """

    query_text: str
    retrieved_chunks: Sequence[str] | None = None
    answer: str | None = None
    latency_ms: int | None = None
    model: str | None = None


def ensure_query_log_schema(conn: sqlite3.Connection) -> None:
    """Ensure the query_log table exists.

    Args:
        conn: Open SQLite connection.
    """

    conn.execute(_QUERY_LOG_TABLE_SQL)


def insert_query_log(db_path: Path, entry: QueryLogEntry) -> int:
    """Insert a query log entry into SQLite.

    Args:
        db_path: SQLite database path.
        entry: QueryLogEntry payload.
    Returns:
        The inserted query_id.
    Raises:
        FileNotFoundError: If the database path does not exist.
        ValueError: If query_text is empty.
        sqlite3.Error: If the insert fails.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not entry.query_text.strip():
        raise ValueError("query_text must be non-empty")

    retrieved_chunks = (
        json.dumps(list(entry.retrieved_chunks))
        if entry.retrieved_chunks is not None
        else None
    )
    with sqlite3.connect(db_path) as conn:
        ensure_query_log_schema(conn)
        cursor = conn.execute(
            """
            INSERT INTO query_log (query_text, retrieved_chunks, answer, latency_ms, model)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entry.query_text,
                retrieved_chunks,
                entry.answer,
                entry.latency_ms,
                entry.model,
            ),
        )
        conn.commit()
        last_row_id = cursor.lastrowid
        if last_row_id is None:
            raise sqlite3.Error("Insert into query_log did not return a rowid.")
        return int(last_row_id)


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
        ensure_query_log_schema(conn)
        conn.commit()


def load_paper_ids(db_path: Path) -> set[str]:
    """Load known arXiv paper IDs from the papers table.

    Args:
        db_path: SQLite database path.
    Returns:
        Set of paper IDs present in the database.
    Raises:
        FileNotFoundError: If the database path does not exist.
        sqlite3.Error: If the query fails.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT paper_id FROM papers").fetchall()
    return {row[0] for row in rows}


def load_page_numbers_by_paper(
    db_path: Path,
    paper_ids: Sequence[str],
) -> dict[str, set[int]]:
    """Load page numbers that exist in the chunks table for each paper ID.

    Args:
        db_path: SQLite database path.
        paper_ids: Paper IDs to filter the lookup.
    Returns:
        Mapping from paper_id to a set of page numbers present in chunks.
    Raises:
        FileNotFoundError: If the database path does not exist.
        sqlite3.Error: If the query fails.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    unique_ids = sorted({paper_id for paper_id in paper_ids if paper_id})
    if not unique_ids:
        return {}

    placeholders = ", ".join("?" for _ in unique_ids)
    sql = f"SELECT paper_id, page_number FROM chunks WHERE paper_id IN ({placeholders})"

    pages_by_paper: dict[str, set[int]] = {paper_id: set() for paper_id in unique_ids}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, unique_ids).fetchall()

    for paper_id, page_number in rows:
        pages_by_paper.setdefault(paper_id, set()).add(page_number)

    return pages_by_paper


def load_chunk_uids_by_page(
    db_path: Path,
    pages: Sequence[tuple[str, int]],
) -> dict[tuple[str, int], list[str]]:
    """Load chunk UIDs for specific (paper_id, page_number) pairs.

    Args:
        db_path: SQLite database path.
        pages: Sequence of (paper_id, page_number) pairs.
    Returns:
        Mapping of (paper_id, page_number) to chunk UID list.
    Raises:
        FileNotFoundError: If the database path does not exist.
        sqlite3.Error: If the query fails.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    unique_pairs = list({(paper_id, page) for paper_id, page in pages if paper_id})
    if not unique_pairs:
        return {}

    clauses = " OR ".join("(paper_id = ? AND page_number = ?)" for _ in unique_pairs)
    params: list[object] = []
    for paper_id, page_number in unique_pairs:
        params.extend([paper_id, page_number])

    sql = f"""
        SELECT paper_id, page_number, chunk_uid
        FROM chunks
        WHERE {clauses}
    """

    mapping: dict[tuple[str, int], list[str]] = {
        (paper_id, page_number): [] for paper_id, page_number in unique_pairs
    }
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    for paper_id, page_number, chunk_uid in rows:
        mapping.setdefault((paper_id, page_number), []).append(chunk_uid)

    return mapping


def load_paper_pdf_paths(
    db_path: Path,
    paper_ids: Sequence[str],
) -> dict[str, Path]:
    """Load pdf_path values for the provided paper IDs.

    Args:
        db_path: SQLite database path.
        paper_ids: Paper IDs to filter the lookup.
    Returns:
        Mapping from normalized paper_id to PDF path.
    Raises:
        FileNotFoundError: If the database path does not exist.
        sqlite3.Error: If the query fails.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    unique_ids = sorted({paper_id for paper_id in paper_ids if paper_id})
    if not unique_ids:
        return {}

    placeholders = ", ".join("?" for _ in unique_ids)
    normalized_ids = [normalize_base_id_for_lookup(paper_id) for paper_id in unique_ids]
    sql = (
        "SELECT paper_id, pdf_path "
        "FROM papers "
        f"WHERE lower(paper_id) IN ({placeholders})"
    )

    paths_by_id: dict[str, Path] = {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, normalized_ids).fetchall()

    for paper_id, pdf_path in rows:
        if not pdf_path:
            continue
        paths_by_id[normalize_base_id_for_lookup(paper_id)] = Path(pdf_path)

    return paths_by_id


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
