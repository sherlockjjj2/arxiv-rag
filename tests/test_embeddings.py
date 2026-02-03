from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path


def _load_modules():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from arxiv_rag.chunk_ids import compute_chunk_uid
    from arxiv_rag.db import (
        deserialize_embedding,
        ensure_chunks_schema,
        ensure_papers_schema,
        normalize_embedding,
        serialize_embedding,
    )
    from arxiv_rag.retrieve import search_vector_with_embedding

    return (
        compute_chunk_uid,
        deserialize_embedding,
        ensure_chunks_schema,
        ensure_papers_schema,
        normalize_embedding,
        serialize_embedding,
        search_vector_with_embedding,
    )


def test_embedding_roundtrip() -> None:
    (
        _,
        deserialize_embedding,
        _,
        _,
        normalize_embedding,
        serialize_embedding,
        _,
    ) = _load_modules()
    vector = [3.0, 4.0]
    normalized = normalize_embedding(vector)

    assert math.isclose(normalized[0], 0.6, rel_tol=1e-6)
    assert math.isclose(normalized[1], 0.8, rel_tol=1e-6)

    blob = serialize_embedding(normalized)
    restored = deserialize_embedding(blob)

    assert len(restored) == 2
    assert math.isclose(restored[0], normalized[0], rel_tol=1e-6)
    assert math.isclose(restored[1], normalized[1], rel_tol=1e-6)


def test_search_vector_with_embedding(tmp_path) -> None:
    (
        compute_chunk_uid,
        _,
        ensure_chunks_schema,
        ensure_papers_schema,
        normalize_embedding,
        serialize_embedding,
        search_vector_with_embedding,
    ) = _load_modules()
    db_path = tmp_path / "vectors.db"
    with sqlite3.connect(db_path) as conn:
        ensure_papers_schema(conn, create_if_missing=True)
        ensure_chunks_schema(conn)

        conn.execute(
            """
            INSERT INTO papers (paper_id, doc_id, title)
            VALUES (?, ?, ?)
            """,
            ("p1", "doc1", "Paper 1"),
        )

        chunk_uid_1 = compute_chunk_uid(
            doc_id="doc1",
            page_number=1,
            chunk_index=0,
            char_start=0,
            char_end=10,
        )
        chunk_uid_2 = compute_chunk_uid(
            doc_id="doc1",
            page_number=1,
            chunk_index=1,
            char_start=11,
            char_end=20,
        )

        embedding_a = normalize_embedding([1.0, 0.0, 0.0])
        embedding_b = normalize_embedding([0.0, 1.0, 0.0])

        conn.executemany(
            """
            INSERT INTO chunks (
                chunk_uid,
                paper_id,
                doc_id,
                page_number,
                chunk_index,
                text,
                char_start,
                char_end,
                token_count,
                embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk_uid_1,
                    "p1",
                    "doc1",
                    1,
                    0,
                    "first chunk",
                    0,
                    10,
                    3,
                    serialize_embedding(embedding_a),
                ),
                (
                    chunk_uid_2,
                    "p1",
                    "doc1",
                    1,
                    1,
                    "second chunk",
                    11,
                    20,
                    3,
                    serialize_embedding(embedding_b),
                ),
            ],
        )
        conn.commit()

    results = search_vector_with_embedding(
        [1.0, 0.0, 0.0],
        top_k=1,
        db_path=db_path,
    )

    assert len(results) == 1
    assert results[0].chunk_uid == chunk_uid_1
