"""Retrieve chunks from SQLite FTS indexes."""

from __future__ import annotations

import heapq
import re
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from arxiv_rag.db import deserialize_embedding_array, normalize_embedding
from arxiv_rag.embeddings_client import EmbeddingsClient

@dataclass(frozen=True)
class ChunkResult:
    """Chunk search result from the FTS index."""

    chunk_uid: str
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
            chunks.chunk_uid,
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
            chunk_uid=row[0],
            chunk_id=row[1],
            paper_id=row[2],
            page_number=row[3],
            text=row[4],
            score=row[5],
        )
        for row in rows
    ]


def search_vector(
    question: str,
    *,
    top_k: int,
    db_path: Path,
    embeddings_client: EmbeddingsClient,
) -> list[ChunkResult]:
    """Search chunks by cosine similarity using stored embeddings.

    Args:
        question: Query text.
        top_k: Number of results to return.
        db_path: Path to the SQLite database.
        embeddings_client: Client for query embeddings.
    Returns:
        List of ChunkResult rows ordered by cosine similarity.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When top_k is not positive.
        sqlite3.Error: When SQLite operations fail.
    """

    query_embedding = embeddings_client.embed([question]).embeddings[0]
    return search_vector_with_embedding(
        query_embedding,
        top_k=top_k,
        db_path=db_path,
    )


def search_vector_with_embedding(
    query_embedding: Sequence[float],
    *,
    top_k: int,
    db_path: Path,
) -> list[ChunkResult]:
    """Search chunks by cosine similarity using a provided embedding.

    Args:
        query_embedding: Pre-computed query embedding.
        top_k: Number of results to return.
        db_path: Path to the SQLite database.
    Returns:
        List of ChunkResult rows ordered by cosine similarity.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When top_k is not positive or embeddings are missing.
        sqlite3.Error: When SQLite operations fail.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    normalized_query = normalize_embedding(query_embedding)
    rows = _load_vector_rows(db_path)
    if not rows:
        return []

    query_array = array("f", normalized_query)
    scored: list[tuple[float, ChunkResult]] = []
    for row in rows:
        embedding = deserialize_embedding_array(row.embedding)
        if len(embedding) != len(query_array):
            continue
        score = _dot_product(query_array, embedding)
        scored.append(
            (
                score,
                ChunkResult(
                    chunk_uid=row.chunk_uid,
                    chunk_id=row.chunk_id,
                    paper_id=row.paper_id,
                    page_number=row.page_number,
                    text=row.text,
                    score=score,
                ),
            )
        )

    if not scored:
        return []

    top_results = heapq.nlargest(top_k, scored, key=lambda pair: pair[0])
    return [result for _, result in top_results]


@dataclass(frozen=True)
class _VectorRow:
    chunk_uid: str
    chunk_id: int
    paper_id: str
    page_number: int
    text: str
    embedding: bytes


def _load_vector_rows(db_path: Path) -> list[_VectorRow]:
    """Load chunk rows with embeddings from SQLite."""

    sql = """
        SELECT chunk_uid, chunk_id, paper_id, page_number, text, embedding
        FROM chunks
        WHERE embedding IS NOT NULL
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [
        _VectorRow(
            chunk_uid=row[0],
            chunk_id=row[1],
            paper_id=row[2],
            page_number=row[3],
            text=row[4],
            embedding=row[5],
        )
        for row in rows
    ]


def _dot_product(left: Iterable[float], right: Iterable[float]) -> float:
    """Compute dot product between two equal-length vectors."""

    return sum(left_value * right_value for left_value, right_value in zip(left, right))
