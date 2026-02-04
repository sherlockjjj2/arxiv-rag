import importlib.util
import sqlite3
import sys
from pathlib import Path

from typer.testing import CliRunner


def _load_module(module_name: str, filename: str):
    module_path = Path(__file__).resolve().parents[1] / "arxiv_rag" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_cli_module():
    return _load_module("arxiv_rag.cli", "cli.py")


def _load_retrieve_module():
    return _load_module("arxiv_rag.retrieve", "retrieve.py")


def _create_fts_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_uid TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                token_count INTEGER,
                embedding BLOB
            );
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='chunk_id'
            );
            """
        )
        rows = [
            (
                1,
                "uid-1",
                "2312.10997",
                "doc1",
                3,
                0,
                "Dense retrieval uses embeddings for search.",
                0,
                0,
                7,
            ),
            (
                2,
                "uid-2",
                "2312.10997",
                "doc1",
                5,
                0,
                "Sparse retrieval uses BM25 ranking.",
                0,
                0,
                6,
            ),
        ]
        conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id,
                chunk_uid,
                paper_id,
                doc_id,
                page_number,
                chunk_index,
                text,
                char_start,
                char_end,
                token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.executemany(
            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
            [(1, rows[0][6]), (2, rows[1][6])],
        )
        conn.commit()


def test_search_fts_returns_results(tmp_path: Path) -> None:
    db_path = tmp_path / "arxiv_rag.db"
    _create_fts_db(db_path)
    retrieve = _load_retrieve_module()

    results = retrieve.search_fts(
        "dense retrieval",
        top_k=5,
        db_path=db_path,
    )

    assert results
    assert results[0].paper_id == "2312.10997"
    assert results[0].page_number == 3


def test_format_snippet_truncates() -> None:
    retrieve = _load_retrieve_module()
    snippet = retrieve.format_snippet("hello   world\nagain", 10)
    assert snippet == "hello w..."


def test_build_fts_query_relaxed() -> None:
    retrieve = _load_retrieve_module()
    query = retrieve.build_fts_query("What are the typical Agent patterns")
    assert query == "(typical AND patterns) AND (Agent)"


def test_build_fts_query_preserves_unicode_terms() -> None:
    retrieve = _load_retrieve_module()
    query = retrieve.build_fts_query("rényi entropy")
    assert query == "(entropy) AND (rényi)"


def test_build_fts_query_handles_greek_letters() -> None:
    retrieve = _load_retrieve_module()
    query = retrieve.build_fts_query("α-β divergence")
    assert query == "(divergence) AND (α OR β)"


def test_build_fts_query_non_ascii_only() -> None:
    retrieve = _load_retrieve_module()
    query = retrieve.build_fts_query("模型")
    assert query == "模型"


def test_query_cli_outputs_snippet(tmp_path: Path) -> None:
    db_path = tmp_path / "arxiv_rag.db"
    _create_fts_db(db_path)
    cli = _load_cli_module()

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "query",
            "dense retrieval",
            "--db",
            str(db_path),
            "--top-k",
            "1",
            "--snippet-chars",
            "20",
        ],
    )

    assert result.exit_code == 0
    assert "[arXiv:2312.10997 p.3]" in result.output
    assert "Dense retrieval" in result.output


def test_query_cli_handles_missing_db_without_traceback(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing.db"
    cli = _load_cli_module()
    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "query",
            "dense retrieval",
            "--mode",
            "vector",
            "--db",
            str(missing_db),
        ],
    )

    assert result.exit_code == 1
    assert "Database not found" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)
