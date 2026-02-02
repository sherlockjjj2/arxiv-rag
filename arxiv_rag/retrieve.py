"""Retrieve chunks from SQLite FTS indexes."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChunkResult:
    """Chunk search result from the FTS index."""

    chunk_id: int
    paper_id: str
    page_number: int
    text: str
    score: float | None = None


def build_fts_query(question: str) -> str:
    """Build an AND-joined FTS query from the question text.

    Args:
        question: Raw user query string.
    Returns:
        FTS query string with terms joined by AND.
    Edge cases:
        Returns an empty string when no tokens are present.
    """

    tokens = [token for token in re.split(r"\s+", question.strip()) if token]
    return " AND ".join(tokens)


def format_snippet(text: str, max_chars: int) -> str:
    """Normalize whitespace and truncate to a maximum character count.

    Args:
        text: Raw chunk text.
        max_chars: Maximum length of the returned snippet.
    Returns:
        Snippet with collapsed whitespace and optional truncation.
    Raises:
        ValueError: When max_chars is not positive.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return f"{normalized[: max_chars - 3].rstrip()}..."


def search_fts(
    question: str,
    *,
    top_k: int,
    db_path: Path,
) -> list[ChunkResult]:
    """Search SQLite FTS5 for relevant chunks using BM25.

    Args:
        question: Query text.
        top_k: Number of results to return.
        db_path: Path to the SQLite database.
    Returns:
        List of ChunkResult rows ordered by BM25 score.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When top_k is not positive.
        sqlite3.Error: When SQLite operations fail.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    fts_query = build_fts_query(question)
    if not fts_query:
        return []

    sql = """
        SELECT
            chunks.chunk_id,
            chunks.paper_id,
            chunks.page_number,
            chunks.text,
            bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks ON chunks_fts.rowid = chunks.chunk_id
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, (fts_query, top_k)).fetchall()

    return [
        ChunkResult(
            chunk_id=row[0],
            paper_id=row[1],
            page_number=row[2],
            text=row[3],
            score=row[4],
        )
        for row in rows
    ]
