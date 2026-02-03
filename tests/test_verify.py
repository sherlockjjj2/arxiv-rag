from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from arxiv_rag.chunk_ids import compute_chunk_uid
from arxiv_rag.db import ensure_chunks_schema, ensure_papers_schema
from arxiv_rag.verify import parse_and_validate_citations, parse_citations


def _seed_db(
    db_path: Path,
    *,
    paper_id: str = "2301.01234",
    page_number: int = 3,
) -> None:
    with sqlite3.connect(db_path) as conn:
        ensure_papers_schema(conn, create_if_missing=True)
        ensure_chunks_schema(conn)

        doc_id = f"doc-{paper_id}"
        conn.execute(
            "INSERT INTO papers (paper_id, doc_id, title) VALUES (?, ?, ?)",
            (paper_id, doc_id, "Test Paper"),
        )

        chunk_uid = compute_chunk_uid(
            doc_id=doc_id,
            page_number=page_number,
            chunk_index=0,
            char_start=0,
            char_end=4,
        )
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
                token_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_uid, paper_id, doc_id, page_number, 0, "text", 0, 4, 1),
        )
        conn.commit()


def test_parse_citations_normalizes_versioned_ids() -> None:
    answer = (
        'Fact [arXiv:2301.01234v2 p.7] *"quote"* '
        'More [arXiv:cs.DS/0101001v3 p.2] *"quote"*'
    )
    citations = parse_citations(answer)
    assert [citation.paper_id for citation in citations] == [
        "2301.01234",
        "cs.DS/0101001",
    ]
    assert [citation.page_number for citation in citations] == [7, 2]


def test_parse_citations_rejects_malformed() -> None:
    answer = 'Bad [[arXiv:2301.01234 p.1]] *"quote"*'
    with pytest.raises(ValueError):
        parse_citations(answer)


def test_parse_citations_rejects_page_zero() -> None:
    answer = 'Bad [arXiv:2301.01234 p.0] *"quote"*'
    with pytest.raises(ValueError):
        parse_citations(answer)


def test_parse_and_validate_citations_success(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    answer = 'OK [arXiv:2301.01234 p.3] *"quote"*'
    citations = parse_and_validate_citations(answer, db_path=db_path)
    assert len(citations) == 1
    assert citations[0].paper_id == "2301.01234"
    assert citations[0].page_number == 3


def test_parse_and_validate_citations_unknown_paper(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    answer = 'Bad [arXiv:9999.99999 p.1] *"quote"*'
    with pytest.raises(ValueError):
        parse_and_validate_citations(answer, db_path=db_path)


def test_parse_and_validate_citations_missing_page(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    answer = 'Bad [arXiv:2301.01234 p.10] *"quote"*'
    with pytest.raises(ValueError):
        parse_and_validate_citations(answer, db_path=db_path)


def test_parse_and_validate_citations_case_insensitive_old_id(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "test.db"
    _seed_db(db_path, paper_id="cs.DS/0101001", page_number=1)
    answer = 'OK [arXiv:cs.ds/0101001 p.1] *"quote"*'
    citations = parse_and_validate_citations(answer, db_path=db_path)
    assert citations[0].paper_id == "cs.ds/0101001"
