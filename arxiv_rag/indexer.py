"""Index embeddings into Chroma from the SQLite chunk store."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tiktoken import Encoding, get_encoding

from arxiv_rag.chroma_client import ChromaConfig, ChromaStore
from arxiv_rag.db import normalize_embedding
from arxiv_rag.embeddings_client import (
    EmbeddingsClient,
    EmbeddingsClientLike,
    EmbeddingsConfig,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRow:
    """Chunk row for embedding + indexing."""

    chunk_uid: str
    doc_id: str
    paper_id: str
    page_number: int
    chunk_index: int
    token_count: int | None
    text: str


def build_chroma_metadata(row: ChunkRow) -> dict[str, object]:
    """Build Chroma metadata for a chunk.

    Args:
        row: Chunk row data.
    Returns:
        Metadata dict for Chroma storage.
    """

    metadata = {
        "doc_id": row.doc_id,
        "paper_id": row.paper_id,
        "page_number": row.page_number,
        "chunk_index": row.chunk_index,
        "token_count": row.token_count,
    }
    if row.token_count is None:
        metadata.pop("token_count")
    return metadata


def load_chunks(
    conn: sqlite3.Connection,
    *,
    doc_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[ChunkRow]:
    """Load chunks from SQLite for indexing.

    Args:
        conn: SQLite connection.
        doc_ids: Optional list of doc_ids to filter by.
        limit: Optional maximum number of chunks to load.
    Returns:
        List of ChunkRow entries ordered by chunk_id.
    Raises:
        ValueError: When limit is non-positive.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")

    params: list[object] = []
    where_clause = ""
    if doc_ids:
        placeholders = ", ".join("?" for _ in doc_ids)
        where_clause = f"WHERE doc_id IN ({placeholders})"
        params.extend(doc_ids)

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(limit)

    sql = f"""
        SELECT chunk_uid, doc_id, paper_id, page_number, chunk_index, token_count, text
        FROM chunks
        {where_clause}
        ORDER BY chunk_id
        {limit_clause}
    """
    rows = conn.execute(sql, params).fetchall()
    return [
        ChunkRow(
            chunk_uid=row[0],
            doc_id=row[1],
            paper_id=row[2],
            page_number=row[3],
            chunk_index=row[4],
            token_count=row[5],
            text=row[6],
        )
        for row in rows
    ]


def batch_chunks_by_tokens(
    chunks: Sequence[ChunkRow],
    *,
    encoding: Encoding,
    max_request_tokens: int,
    max_input_tokens: int,
) -> list[list[ChunkRow]]:
    """Batch chunks to honor per-request and per-input token caps.

    Args:
        chunks: Chunk rows to batch.
        encoding: Token encoding for counts.
        max_request_tokens: Max tokens per embeddings request.
        max_input_tokens: Max tokens per single chunk.
    Returns:
        List of chunk batches.
    Raises:
        ValueError: When token limits are exceeded.
    """

    batches: list[list[ChunkRow]] = []
    current: list[ChunkRow] = []
    current_tokens = 0

    for chunk in chunks:
        token_count = _token_count(chunk, encoding)
        if token_count > max_input_tokens:
            raise ValueError(
                f"Chunk {chunk.chunk_uid} exceeds max_input_tokens={max_input_tokens}"
            )
        if token_count > max_request_tokens:
            raise ValueError(
                f"Chunk {chunk.chunk_uid} exceeds max_request_tokens={max_request_tokens}"
            )

        if current and current_tokens + token_count > max_request_tokens:
            batches.append(current)
            current = []
            current_tokens = 0

        current.append(chunk)
        current_tokens += token_count

    if current:
        batches.append(current)

    return batches


def index_chunks(
    *,
    db_path: Path,
    chroma_config: ChromaConfig,
    embeddings_config: EmbeddingsConfig,
    doc_ids: Sequence[str] | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    force_delete: bool = False,
    chroma_store: ChromaStore | None = None,
    embeddings_client: EmbeddingsClientLike | None = None,
) -> int:
    """Index chunk embeddings into Chroma.

    Args:
        db_path: SQLite database path.
        chroma_config: Chroma configuration.
        embeddings_config: Embeddings configuration.
        doc_ids: Optional doc_ids to re-index.
        limit: Optional limit of chunks to index.
        batch_size: Optional fixed batch size; overrides token-based batching.
        force_delete: When True, allow deletes even if limit is set.
        chroma_store: Optional Chroma store override for testing.
        embeddings_client: Optional embeddings client override for testing.
    Returns:
        Number of chunks indexed.
    Raises:
        FileNotFoundError: When the database is missing.
        ValueError: When limits are invalid or embeddings mismatch.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    embeddings_client = embeddings_client or EmbeddingsClient(embeddings_config)
    chroma_store = chroma_store or ChromaStore(chroma_config)
    encoding = get_encoding("cl100k_base")

    with sqlite3.connect(db_path) as conn:
        chunks = load_chunks(conn, doc_ids=doc_ids, limit=limit)

    requested_doc_ids = list(
        dict.fromkeys(doc_ids or [chunk.doc_id for chunk in chunks])
    )
    allow_delete = limit is None or force_delete
    if limit is not None and not force_delete and requested_doc_ids:
        LOGGER.warning(
            "limit=%s set; skipping delete for %s doc_ids. "
            "Use force_delete=True to allow deletes.",
            limit,
            len(requested_doc_ids),
        )

    if not chunks:
        if allow_delete and doc_ids:
            for doc_id in requested_doc_ids:
                deleted = chroma_store.delete_by_doc_id(doc_id)
                LOGGER.info("Deleted %s vectors for doc_id=%s", deleted, doc_id)
        return 0

    chunks_by_doc_id: dict[str, list[ChunkRow]] = {}
    for chunk in chunks:
        chunks_by_doc_id.setdefault(chunk.doc_id, []).append(chunk)

    total = len(chunks)
    processed = 0
    total_tokens = 0
    batch_index = 0

    doc_ids_to_process = requested_doc_ids or list(chunks_by_doc_id.keys())
    for doc_id in doc_ids_to_process:
        doc_chunks = chunks_by_doc_id.get(doc_id, [])
        if not doc_chunks:
            if allow_delete and doc_ids:
                deleted = chroma_store.delete_by_doc_id(doc_id)
                LOGGER.info("Deleted %s vectors for doc_id=%s", deleted, doc_id)
            continue

        existing_ids = (
            set(chroma_store.list_ids_by_doc_id(doc_id)) if allow_delete else set()
        )
        batches = _build_batches(
            doc_chunks,
            encoding=encoding,
            max_request_tokens=embeddings_config.max_request_tokens,
            max_input_tokens=embeddings_config.max_input_tokens,
            batch_size=batch_size,
        )

        for batch in batches:
            batch_index += 1
            batch_texts = [chunk.text for chunk in batch]
            result = embeddings_client.embed(batch_texts)
            if len(result.embeddings) != len(batch):
                raise ValueError("Embedding response size mismatch.")

            ids = [chunk.chunk_uid for chunk in batch]
            embeddings = [
                normalize_embedding(embedding) for embedding in result.embeddings
            ]
            metadatas = [build_chroma_metadata(chunk) for chunk in batch]
            chroma_store.upsert_embeddings(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            processed += len(batch)
            if result.total_tokens is not None:
                total_tokens += result.total_tokens
            LOGGER.info(
                "batch=%s indexed=%s/%s batch_size=%s tokens=%s",
                batch_index,
                processed,
                total,
                len(batch),
                result.total_tokens,
            )

        if allow_delete:
            new_ids = {chunk.chunk_uid for chunk in doc_chunks}
            stale_ids = sorted(existing_ids - new_ids)
            if stale_ids:
                deleted = chroma_store.delete_by_ids(stale_ids)
                LOGGER.info(
                    "Deleted %s stale vectors for doc_id=%s",
                    deleted,
                    doc_id,
                )

    return processed


def _token_count(chunk: ChunkRow, encoding: Encoding) -> int:
    if chunk.token_count is not None:
        return chunk.token_count
    return len(encoding.encode(chunk.text, disallowed_special=()))


def _build_batches(
    chunks: Sequence[ChunkRow],
    *,
    encoding: Encoding,
    max_request_tokens: int,
    max_input_tokens: int,
    batch_size: int | None,
) -> list[list[ChunkRow]]:
    if batch_size is None:
        return batch_chunks_by_tokens(
            chunks,
            encoding=encoding,
            max_request_tokens=max_request_tokens,
            max_input_tokens=max_input_tokens,
        )

    batches: list[list[ChunkRow]] = []
    current: list[ChunkRow] = []
    current_tokens = 0

    for chunk in chunks:
        token_count = _token_count(chunk, encoding)
        if token_count > max_input_tokens:
            raise ValueError(
                f"Chunk {chunk.chunk_uid} exceeds max_input_tokens={max_input_tokens}"
            )
        current.append(chunk)
        current_tokens += token_count
        if len(current) >= batch_size:
            if current_tokens > max_request_tokens:
                raise ValueError(
                    "Batch exceeds max_request_tokens; reduce batch_size or max_request_tokens."
                )
            batches.append(current)
            current = []
            current_tokens = 0

    if current:
        if current_tokens > max_request_tokens:
            raise ValueError(
                "Batch exceeds max_request_tokens; reduce batch_size or max_request_tokens."
            )
        batches.append(current)

    return batches
