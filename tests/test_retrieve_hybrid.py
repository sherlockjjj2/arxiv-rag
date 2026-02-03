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
    from arxiv_rag.db import ensure_chunks_schema, normalize_embedding, serialize_embedding
    from arxiv_rag.retrieve import ChunkResult, _fuse_rrf, search_hybrid

    return (
        ChromaConfig,
        ensure_chunks_schema,
        normalize_embedding,
        serialize_embedding,
        ChunkResult,
        _fuse_rrf,
        search_hybrid,
    )


def test_fuse_rrf_dedupes_and_preserves_first_seen() -> None:
    (
        _,
        _,
        _,
        _,
        ChunkResult,
        _fuse_rrf,
        _,
    ) = _load_modules()
    fts_results = [
        ChunkResult(
            chunk_uid="uid-1",
            chunk_id=1,
            paper_id="p1",
            page_number=1,
            text="fts-one",
            score=1.0,
        ),
        ChunkResult(
            chunk_uid="uid-2",
            chunk_id=2,
            paper_id="p1",
            page_number=2,
            text="fts-two",
            score=2.0,
        ),
    ]
    vector_results = [
        ChunkResult(
            chunk_uid="uid-2",
            chunk_id=99,
            paper_id="p2",
            page_number=9,
            text="vector-two",
            score=0.1,
        ),
        ChunkResult(
            chunk_uid="uid-3",
            chunk_id=3,
            paper_id="p3",
            page_number=3,
            text="vector-three",
            score=0.2,
        ),
    ]

    results = _fuse_rrf(
        fts_results,
        vector_results,
        top_k=3,
        rrf_k=60,
        fts_weight=1.0,
        vector_weight=1.0,
    )

    assert [result.chunk_uid for result in results] == ["uid-2", "uid-1", "uid-3"]
    assert results[0].text == "fts-two"
    assert results[0].chunk_id == 2
    assert len(results[0].provenance) == 2


@dataclass(frozen=True)
class _EmbeddingResult:
    embeddings: list[list[float]]
    total_tokens: int | None


class _FakeEmbeddingsClient:
    def embed(self, inputs):
        embeddings = [[1.0, 0.0] for _ in inputs]
        return _EmbeddingResult(embeddings=embeddings, total_tokens=None)


class _FailingChromaStore:
    def query(self, *, query_embeddings, top_k, where=None):
        raise ImportError("chroma not available")


def _seed_hybrid_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    (
        _,
        ensure_chunks_schema,
        normalize_embedding,
        serialize_embedding,
        _,
        _,
        _,
    ) = _load_modules()
    with sqlite3.connect(db_path) as conn:
        ensure_chunks_schema(conn)
        embedding = serialize_embedding(normalize_embedding([1.0, 0.0]))
        conn.execute(
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
            (
                "uid-1",
                "p1",
                "doc1",
                1,
                0,
                "dense retrieval uses embeddings",
                0,
                10,
                4,
                embedding,
            ),
        )
        conn.commit()


def test_search_hybrid_falls_back_to_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "hybrid.db"
    _seed_hybrid_db(db_path)
    (ChromaConfig, _, _, _, _, _, search_hybrid) = _load_modules()
    chroma_config = ChromaConfig(
        persist_dir=tmp_path / "chroma",
        collection_name="test",
        distance="cosine",
    )

    output = search_hybrid(
        "dense retrieval",
        top_k=1,
        db_path=db_path,
        embeddings_client=_FakeEmbeddingsClient(),
        chroma_config=chroma_config,
        chroma_store=_FailingChromaStore(),
    )

    assert output.results
    assert any("falling back" in warning for warning in output.warnings)
    assert any(
        prov.backend == "vector"
        for prov in output.results[0].provenance
    )
