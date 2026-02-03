from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path


def _load_modules():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from arxiv_rag.chroma_client import ChromaConfig
    from arxiv_rag.chunk_ids import compute_chunk_uid
    from arxiv_rag.db import ensure_chunks_schema, ensure_papers_schema
    from arxiv_rag.retrieve import search_vector_chroma

    return (
        ChromaConfig,
        compute_chunk_uid,
        ensure_chunks_schema,
        ensure_papers_schema,
        search_vector_chroma,
    )


@dataclass(frozen=True)
class _EmbeddingResult:
    embeddings: list[list[float]]
    total_tokens: int | None


class _FakeEmbeddingsClient:
    def embed(self, inputs):
        embeddings = [[1.0, 0.0, 0.0] for _ in inputs]
        return _EmbeddingResult(embeddings=embeddings, total_tokens=None)


class _FakeChromaStore:
    def __init__(self, ids: list[str], distances: list[float]):
        self._ids = ids
        self._distances = distances

    def query(self, *, query_embeddings, top_k, where=None):
        return self._ids[:top_k], self._distances[:top_k]


def _seed_db(db_path: Path) -> list[str]:
    (
        _,
        compute_chunk_uid,
        ensure_chunks_schema,
        ensure_papers_schema,
        _,
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
                ),
            ],
        )
        conn.commit()

    return [chunk_uid_1, chunk_uid_2]


def test_search_vector_chroma_preserves_order(tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.db"
    chunk_uids = _seed_db(db_path)
    (ChromaConfig, _, _, _, search_vector_chroma) = _load_modules()
    chroma_config = ChromaConfig(
        persist_dir=tmp_path / "chroma",
        collection_name="test",
        distance="cosine",
    )

    fake_store = _FakeChromaStore(
        ids=[chunk_uids[1], chunk_uids[0]],
        distances=[0.1, 0.2],
    )
    fake_embeddings = _FakeEmbeddingsClient()

    results = search_vector_chroma(
        "dense retrieval",
        top_k=2,
        db_path=db_path,
        embeddings_client=fake_embeddings,
        chroma_config=chroma_config,
        chroma_store=fake_store,
    )

    assert [result.chunk_uid for result in results] == [chunk_uids[1], chunk_uids[0]]


def test_search_vector_chroma_skips_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.db"
    chunk_uids = _seed_db(db_path)
    (ChromaConfig, _, _, _, search_vector_chroma) = _load_modules()
    chroma_config = ChromaConfig(
        persist_dir=tmp_path / "chroma",
        collection_name="test",
        distance="cosine",
    )

    fake_store = _FakeChromaStore(
        ids=[chunk_uids[0], "missing", chunk_uids[1]],
        distances=[0.1, 0.2, 0.3],
    )
    fake_embeddings = _FakeEmbeddingsClient()

    results = search_vector_chroma(
        "dense retrieval",
        top_k=3,
        db_path=db_path,
        embeddings_client=fake_embeddings,
        chroma_config=chroma_config,
        chroma_store=fake_store,
    )

    assert [result.chunk_uid for result in results] == [chunk_uids[0], chunk_uids[1]]
