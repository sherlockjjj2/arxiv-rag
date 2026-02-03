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
    from arxiv_rag.embeddings_client import EmbeddingsConfig
    from arxiv_rag.indexer import ChunkRow, build_chroma_metadata, index_chunks

    return (
        ChromaConfig,
        compute_chunk_uid,
        ensure_chunks_schema,
        ensure_papers_schema,
        EmbeddingsConfig,
        ChunkRow,
        build_chroma_metadata,
        index_chunks,
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
    def __init__(self):
        self.deleted: list[str] = []
        self.upserts: list[tuple[list[str], list[list[float]], list[dict[str, object]]]] = []

    def delete_by_doc_id(self, doc_id: str) -> int:
        self.deleted.append(doc_id)
        return 0

    def upsert_embeddings(self, *, ids, embeddings, metadatas) -> None:
        self.upserts.append((list(ids), list(embeddings), list(metadatas)))


def _seed_db(db_path: Path) -> list[str]:
    (
        _,
        compute_chunk_uid,
        ensure_chunks_schema,
        ensure_papers_schema,
        _,
        _,
        _,
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
        conn.execute(
            """
            INSERT INTO papers (paper_id, doc_id, title)
            VALUES (?, ?, ?)
            """,
            ("p2", "doc2", "Paper 2"),
        )

        chunk_uid_1 = compute_chunk_uid(
            doc_id="doc1",
            page_number=1,
            chunk_index=0,
            char_start=0,
            char_end=10,
        )
        chunk_uid_2 = compute_chunk_uid(
            doc_id="doc2",
            page_number=1,
            chunk_index=0,
            char_start=0,
            char_end=10,
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
                    "p2",
                    "doc2",
                    1,
                    0,
                    "second chunk",
                    0,
                    10,
                    3,
                ),
            ],
        )
        conn.commit()

    return [chunk_uid_1, chunk_uid_2]


def test_build_chroma_metadata() -> None:
    (
        _,
        _,
        _,
        _,
        _,
        ChunkRow,
        build_chroma_metadata,
        _,
    ) = _load_modules()
    row = ChunkRow(
        chunk_uid="uid",
        doc_id="doc1",
        paper_id="p1",
        page_number=2,
        chunk_index=1,
        token_count=42,
        text="hello",
    )
    metadata = build_chroma_metadata(row)
    assert metadata["doc_id"] == "doc1"
    assert metadata["paper_id"] == "p1"
    assert metadata["page_number"] == 2
    assert metadata["chunk_index"] == 1
    assert metadata["token_count"] == 42


def test_indexer_deletes_and_upserts(tmp_path: Path) -> None:
    db_path = tmp_path / "vectors.db"
    chunk_uids = _seed_db(db_path)

    (
        ChromaConfig,
        _,
        _,
        _,
        EmbeddingsConfig,
        _,
        _,
        index_chunks,
    ) = _load_modules()
    fake_store = _FakeChromaStore()
    fake_embeddings = _FakeEmbeddingsClient()
    chroma_config = ChromaConfig(
        persist_dir=tmp_path / "chroma",
        collection_name="test",
        distance="cosine",
    )
    embeddings_config = EmbeddingsConfig(model="text-embedding-3-small")

    total = index_chunks(
        db_path=db_path,
        chroma_config=chroma_config,
        embeddings_config=embeddings_config,
        chroma_store=fake_store,
        embeddings_client=fake_embeddings,
    )

    assert total == 2
    assert set(fake_store.deleted) == {"doc1", "doc2"}
    assert fake_store.upserts
    upsert_ids = fake_store.upserts[0][0]
    assert upsert_ids == chunk_uids
