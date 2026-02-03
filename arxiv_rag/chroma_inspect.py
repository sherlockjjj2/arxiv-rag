"""Helpers for inspecting Chroma collections."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from arxiv_rag.chroma_client import ChromaConfig, ChromaStore


@dataclass(frozen=True)
class ChromaDocCount:
    """Count of vectors in Chroma for a document."""

    doc_id: str
    count: int


def load_doc_ids(
    conn: sqlite3.Connection,
    *,
    doc_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Load distinct doc_ids from SQLite.

    Args:
        conn: SQLite connection.
        doc_ids: Optional doc_ids to filter by.
        limit: Optional limit on number of doc_ids.
    Returns:
        List of distinct doc_id values.
    Raises:
        ValueError: When limit is non-positive.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")

    params: list[object] = []
    where_clause = ""
    if doc_ids:
        placeholders = ", ".join("?" for _ in doc_ids)
        where_clause = f"WHERE doc_id IN ({placeholders})"
        params.extend(doc_ids)

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(limit)

    sql = f"""
        SELECT DISTINCT doc_id
        FROM chunks
        {where_clause}
        ORDER BY doc_id
        {limit_clause}
    """
    rows = conn.execute(sql, params).fetchall()
    return [row[0] for row in rows if row[0]]


def inspect_chroma_counts(
    *,
    db_path: Path,
    chroma_config: ChromaConfig,
    doc_ids: Sequence[str] | None = None,
    limit: int | None = None,
    chroma_store: ChromaStore | None = None,
) -> list[ChromaDocCount]:
    """Inspect Chroma counts per doc_id.

    Args:
        db_path: SQLite database path.
        chroma_config: Chroma configuration.
        doc_ids: Optional doc_ids to inspect.
        limit: Optional limit for number of doc_ids.
        chroma_store: Optional Chroma store override for testing.
    Returns:
        List of ChromaDocCount entries.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When limit is invalid.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")

    with sqlite3.connect(db_path) as conn:
        doc_ids_to_check = load_doc_ids(conn, doc_ids=doc_ids, limit=limit)

    store = chroma_store or ChromaStore(chroma_config)
    return [
        ChromaDocCount(doc_id=doc_id, count=store.count_by_doc_id(doc_id))
        for doc_id in doc_ids_to_check
    ]
