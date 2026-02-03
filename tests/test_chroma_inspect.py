from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def _load_modules():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from arxiv_rag.chroma_client import ChromaConfig
    from arxiv_rag.chroma_inspect import inspect_chroma_counts
    from arxiv_rag.db import ensure_chunks_schema, ensure_papers_schema

    return (
        ChromaConfig,
        inspect_chroma_counts,
        ensure_chunks_schema,
        ensure_papers_schema,
    )


class _FakeChromaStore:
    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def count_by_doc_id(self, doc_id: str) -> int:
        return self._counts.get(doc_id, 0)


def _seed_db(db_path: Path) -> None:
    (
        _,
        _,
        ensure_chunks_schema,
        ensure_papers_schema,
    ) = _load_modules()
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
        conn.execute(
            """
            INSERT INTO papers (paper_id, doc_id, title)
            VALUES (?, ?, ?)
            """,
            ("p2", "doc2", "Paper 2"),
        )
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
                token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("uid-1", "p1", "doc1", 1, 0, "first chunk", 0, 10, 3),
                ("uid-2", "p2", "doc2", 1, 0, "second chunk", 0, 10, 3),
            ],
        )
        conn.commit()


def test_inspect_chroma_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.db"
    _seed_db(db_path)
    (ChromaConfig, inspect_chroma_counts, _, _) = _load_modules()
    chroma_config = ChromaConfig(
        persist_dir=tmp_path / "chroma",
        collection_name="test",
        distance="cosine",
    )
    fake_store = _FakeChromaStore({"doc1": 5, "doc2": 3})

    counts = inspect_chroma_counts(
        db_path=db_path,
        chroma_config=chroma_config,
        chroma_store=fake_store,
    )

    assert [(entry.doc_id, entry.count) for entry in counts] == [
        ("doc1", 5),
        ("doc2", 3),
    ]
