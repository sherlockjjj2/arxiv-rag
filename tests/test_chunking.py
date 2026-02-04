import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest


def _load_chunk_module():
    module_path = Path(__file__).resolve().parents[1] / "arxiv_rag" / "chunk.py"
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("chunk", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["chunk"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _create_papers_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT,
            categories TEXT,
            published_date TEXT,
            pdf_path TEXT,
            total_pages INTEGER,
            indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            source_type TEXT DEFAULT 'arxiv'
        );
        """
    )
    conn.commit()


def test_chunk_page_overlap_and_counts():
    chunk = _load_chunk_module()
    encoder = chunk.get_encoding("cl100k_base")
    text = ("hello world " * 200).strip()

    config = chunk.ChunkConfig(target_tokens=40, overlap_tokens=10)
    chunks = chunk.chunk_page(
        page_text=text,
        page_num=1,
        paper_id="2501.12345",
        doc_id="doc1",
        config=config,
        encoder=encoder,
    )

    assert chunks
    for idx, record in enumerate(chunks):
        assert record.chunk_index == idx
        assert record.page_number == 1
        assert record.token_count == len(encoder.encode(record.text))

    for left, right in zip(chunks, chunks[1:]):
        left_tokens = encoder.encode(left.text)
        right_tokens = encoder.encode(right.text)
        assert (
            left_tokens[-config.overlap_tokens :]
            == right_tokens[: config.overlap_tokens]
        )


def test_chunk_page_skips_empty_text():
    chunk = _load_chunk_module()
    encoder = chunk.get_encoding("cl100k_base")
    config = chunk.ChunkConfig(target_tokens=10, overlap_tokens=2)
    chunks = chunk.chunk_page(
        page_text="   \n\t",
        page_num=1,
        paper_id="2501.12345",
        doc_id="doc1",
        config=config,
        encoder=encoder,
    )
    assert chunks == []


def test_chunk_page_preserves_multibyte_symbols():
    chunk = _load_chunk_module()
    encoder = chunk.get_encoding("cl100k_base")
    text = "texts CF ⊂C, where |CF| = k ≪|C|. For a ﬁxed k"

    config = chunk.ChunkConfig(target_tokens=5, overlap_tokens=1)
    chunks = chunk.chunk_page(
        page_text=text,
        page_num=1,
        paper_id="2501.12345",
        doc_id="doc1",
        config=config,
        encoder=encoder,
    )

    assert chunks
    assert all("\ufffd" not in record.text for record in chunks)
    combined = " ".join(record.text for record in chunks)
    assert "⊂" in combined
    assert "≪" in combined
    assert "ﬁ" in combined


def test_chunk_page_rejects_invalid_config():
    chunk = _load_chunk_module()
    encoder = chunk.get_encoding("cl100k_base")
    with pytest.raises(ValueError):
        chunk.chunk_page(
            page_text="hello world",
            page_num=1,
            paper_id="2501.12345",
            doc_id="doc1",
            config=chunk.ChunkConfig(target_tokens=0, overlap_tokens=0),
            encoder=encoder,
        )
    with pytest.raises(ValueError):
        chunk.chunk_page(
            page_text="hello world",
            page_num=1,
            paper_id="2501.12345",
            doc_id="doc1",
            config=chunk.ChunkConfig(target_tokens=5, overlap_tokens=5),
            encoder=encoder,
        )


def test_load_parsed_document_requires_doc_id(tmp_path: Path):
    chunk = _load_chunk_module()
    parsed_path = tmp_path / "parsed.json"
    parsed_path.write_text(json.dumps({"pages": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        chunk.load_parsed_document(parsed_path)


def test_load_parsed_document_rejects_non_json_input(tmp_path: Path):
    chunk = _load_chunk_module()
    parsed_path = tmp_path / "2505.09388v1.pdf"
    parsed_path.write_bytes(b"%PDF-\x8f\x00")

    with pytest.raises(ValueError, match="--parsed expects JSON files"):
        chunk.load_parsed_document(parsed_path)


def test_ingest_replaces_chunks_on_new_doc_id(tmp_path: Path):
    chunk = _load_chunk_module()
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        _create_papers_table(conn)
        conn.execute(
            """
            INSERT INTO papers (
                paper_id, title, authors, abstract, categories, published_date,
                pdf_path, total_pages, source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2501.12345",
                "Title",
                "[]",
                "Abstract",
                "[]",
                "2025-01-01",
                "data/arxiv-papers/2501.12345v1.pdf",
                1,
                "arxiv",
            ),
        )
        conn.commit()

        config = chunk.ChunkConfig(target_tokens=20, overlap_tokens=5)
        doc_v1 = chunk.ParsedDocument(
            doc_id="doc_v1",
            pdf_path=Path("data/arxiv-papers/2501.12345v1.pdf"),
            pages=[chunk.ParsedPage(page_number=1, text="hello world " * 50)],
        )
        chunks_v1 = chunk.chunk_document(doc_v1, "2501.12345", config)
        chunk._ingest_chunks(conn, doc_v1, "2501.12345", chunks_v1)

        count_v1 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count_v1 == len(chunks_v1)

        doc_v2 = chunk.ParsedDocument(
            doc_id="doc_v2",
            pdf_path=Path("data/arxiv-papers/2501.12345v2.pdf"),
            pages=[chunk.ParsedPage(page_number=1, text="new text " * 60)],
        )
        chunks_v2 = chunk.chunk_document(doc_v2, "2501.12345", config)
        chunk._ingest_chunks(conn, doc_v2, "2501.12345", chunks_v2)

        count_v2 = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count_v2 == len(chunks_v2)
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ?",
                ("doc_v1",),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT doc_id FROM papers WHERE paper_id = ?",
                ("2501.12345",),
            ).fetchone()[0]
            == "doc_v2"
        )
