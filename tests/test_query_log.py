import json
import sqlite3
from pathlib import Path

from arxiv_rag.db import QueryLogEntry, insert_query_log


def test_insert_query_log(tmp_path: Path) -> None:
    db_path = tmp_path / "arxiv_rag.db"
    db_path.touch()

    entry = QueryLogEntry(
        query_text="what is dense retrieval",
        retrieved_chunks=["uid1", "uid2"],
        answer="dense retrieval uses embeddings",
        latency_ms=120,
        model="retrieval=hybrid:text-embedding-3-small",
    )
    query_id = insert_query_log(db_path, entry)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT query_text, retrieved_chunks, answer, latency_ms, model "
            "FROM query_log WHERE query_id = ?",
            (query_id,),
        ).fetchone()

    assert row is not None
    query_text, retrieved_chunks, answer, latency_ms, model = row
    assert query_text == entry.query_text
    assert json.loads(retrieved_chunks) == ["uid1", "uid2"]
    assert answer == entry.answer
    assert latency_ms == entry.latency_ms
    assert model == entry.model
