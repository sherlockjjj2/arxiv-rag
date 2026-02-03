"""Typer-based CLI for arxiv-rag."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import typer
from dotenv import load_dotenv
from tiktoken import Encoding, get_encoding

from arxiv_rag.chroma_client import ChromaConfig
from arxiv_rag.chroma_inspect import inspect_chroma_counts
from arxiv_rag.db import normalize_embedding, serialize_embedding
from arxiv_rag.embeddings_client import EmbeddingsClient, EmbeddingsConfig
from arxiv_rag.indexer import index_chunks
from arxiv_rag.retrieve import format_snippet, search_fts, search_vector_chroma

app = typer.Typer(help="CLI for arXiv RAG utilities.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Show help when no subcommand is provided."""

    load_dotenv()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command()
def query(
    question: str,
    top_k: int = typer.Option(5, help="Number of chunks to return."),
    mode: Literal["fts", "vector"] = typer.Option(
        "fts",
        help="Retrieval mode.",
    ),
    model: str = typer.Option(
        "text-embedding-3-small",
        help="Embedding model for vector search.",
    ),
    db: Path = typer.Option(
        Path("data/arxiv_rag.db"),
        help="SQLite database path.",
        exists=False,
        dir_okay=False,
    ),
    chroma_dir: Path = typer.Option(
        Path("data/chroma"),
        envvar="CHROMA_DIR",
        help="Chroma persistence directory.",
    ),
    collection: str = typer.Option(
        "arxiv_chunks_te3s_v1",
        envvar="CHROMA_COLLECTION",
        help="Chroma collection name.",
    ),
    distance: Literal["cosine", "l2", "ip"] = typer.Option(
        "cosine",
        envvar="CHROMA_DISTANCE",
        help="Chroma distance metric.",
    ),
    snippet_chars: int = typer.Option(
        240,
        help="Maximum characters to display per snippet.",
    ),
    show_score: bool = typer.Option(
        False,
        help="Include BM25 score in the output.",
    ),
) -> None:
    """Query SQLite for matching chunks via FTS or embeddings."""

    if snippet_chars <= 0:
        typer.echo("snippet_chars must be > 0", err=True)
        raise typer.Exit(code=1)

    try:
        if mode == "fts":
            results = search_fts(question, top_k=top_k, db_path=db)
        else:
            embeddings_client = EmbeddingsClient(EmbeddingsConfig(model=model))
            chroma_config = ChromaConfig(
                persist_dir=chroma_dir,
                collection_name=collection,
                distance=distance,
            )
            results = search_vector_chroma(
                question,
                top_k=top_k,
                db_path=db,
                embeddings_client=embeddings_client,
                chroma_config=chroma_config,
            )
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except sqlite3.Error as exc:
        typer.echo(f"SQLite error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if not results:
        typer.echo("No results found.")
        raise typer.Exit(code=0)

    for index, result in enumerate(results):
        snippet = format_snippet(result.text, snippet_chars)
        line = f"[arXiv:{result.paper_id} p.{result.page_number}] {snippet}"
        if show_score and result.score is not None:
            line = f"{line} (score={result.score:.3f})"
        typer.echo(line)
        if index < len(results) - 1:
            typer.echo()


@dataclass(frozen=True)
class _ChunkToEmbed:
    chunk_id: int
    text: str
    token_count: int | None


@app.command()
def embed(
    db: Path = typer.Option(
        Path("data/arxiv_rag.db"),
        help="SQLite database path.",
        exists=False,
        dir_okay=False,
    ),
    model: str = typer.Option(
        "text-embedding-3-small",
        help="Embedding model name.",
    ),
    max_request_tokens: int = typer.Option(
        32000,
        help="Maximum total tokens per embeddings request.",
    ),
    max_input_tokens: int = typer.Option(
        8192,
        help="Maximum tokens allowed for a single input.",
    ),
    limit: int | None = typer.Option(
        None,
        help="Limit number of chunks to embed.",
    ),
    force: bool = typer.Option(
        False,
        help="Recompute embeddings even when already present.",
    ),
) -> None:
    """Generate and store embeddings for chunks."""

    logging.basicConfig(level=logging.INFO)
    run_id = uuid4().hex[:8]
    logger = logging.getLogger("arxiv_rag.embed")

    if max_request_tokens <= 0:
        typer.echo("max_request_tokens must be > 0", err=True)
        raise typer.Exit(code=1)
    if max_input_tokens <= 0:
        typer.echo("max_input_tokens must be > 0", err=True)
        raise typer.Exit(code=1)
    if limit is not None and limit <= 0:
        typer.echo("limit must be > 0", err=True)
        raise typer.Exit(code=1)

    config = EmbeddingsConfig(
        model=model,
        max_request_tokens=max_request_tokens,
        max_input_tokens=max_input_tokens,
    )
    embeddings_client = EmbeddingsClient(config)
    encoding = get_encoding("cl100k_base")

    try:
        with sqlite3.connect(db) as conn:
            chunks = _load_chunks_to_embed(conn, force=force, limit=limit)
            if not chunks:
                typer.echo("No chunks to embed.")
                raise typer.Exit(code=0)

            batches = _batch_chunks_by_tokens(
                chunks,
                encoding=encoding,
                max_request_tokens=config.max_request_tokens,
                max_input_tokens=config.max_input_tokens,
            )

            total = len(chunks)
            processed = 0
            total_tokens = 0
            for index, batch in enumerate(batches, start=1):
                batch_texts = [chunk.text for chunk in batch]
                result = embeddings_client.embed(batch_texts)
                if len(result.embeddings) != len(batch):
                    raise ValueError("Embedding response size mismatch.")

                updates: list[tuple[bytes, int]] = []
                for chunk, embedding in zip(batch, result.embeddings):
                    normalized = normalize_embedding(embedding)
                    updates.append((serialize_embedding(normalized), chunk.chunk_id))

                conn.executemany(
                    "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
                    updates,
                )
                conn.commit()

                processed += len(batch)
                if result.total_tokens is not None:
                    total_tokens += result.total_tokens
                logger.info(
                    "run=%s batch=%s embedded=%s/%s tokens=%s",
                    run_id,
                    index,
                    processed,
                    total,
                    result.total_tokens,
                )

            typer.echo(
                f"Embedded {processed} chunks. total_tokens={total_tokens}",
            )
    except sqlite3.Error as exc:
        typer.echo(f"SQLite error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def index(
    db: Path = typer.Option(
        Path("data/arxiv_rag.db"),
        help="SQLite database path.",
        exists=False,
        dir_okay=False,
    ),
    model: str = typer.Option(
        "text-embedding-3-small",
        help="Embedding model name.",
    ),
    chroma_dir: Path = typer.Option(
        Path("data/chroma"),
        envvar="CHROMA_DIR",
        help="Chroma persistence directory.",
    ),
    collection: str = typer.Option(
        "arxiv_chunks_te3s_v1",
        envvar="CHROMA_COLLECTION",
        help="Chroma collection name.",
    ),
    distance: Literal["cosine", "l2", "ip"] = typer.Option(
        "cosine",
        envvar="CHROMA_DISTANCE",
        help="Chroma distance metric.",
    ),
    doc_id: list[str] | None = typer.Option(
        None,
        help="Limit indexing to specific doc_id values (repeatable).",
    ),
    limit: int | None = typer.Option(
        None,
        help="Limit number of chunks to index.",
    ),
    max_request_tokens: int = typer.Option(
        32000,
        help="Maximum total tokens per embeddings request.",
    ),
    max_input_tokens: int = typer.Option(
        8192,
        help="Maximum tokens allowed for a single input.",
    ),
    batch_size: int | None = typer.Option(
        None,
        help="Fixed batch size (overrides token-based batching).",
    ),
) -> None:
    """Generate embeddings and index into Chroma."""

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    chroma_config = ChromaConfig(
        persist_dir=chroma_dir,
        collection_name=collection,
        distance=distance,
    )
    embeddings_config = EmbeddingsConfig(
        model=model,
        max_request_tokens=max_request_tokens,
        max_input_tokens=max_input_tokens,
    )

    try:
        total = index_chunks(
            db_path=db,
            chroma_config=chroma_config,
            embeddings_config=embeddings_config,
            doc_ids=doc_id,
            limit=limit,
            batch_size=batch_size,
        )
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except sqlite3.Error as exc:
        typer.echo(f"SQLite error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if total == 0:
        typer.echo("No chunks to index.")
        raise typer.Exit(code=0)

    typer.echo(f"Indexed {total} chunks into Chroma.")


@app.command()
def inspect(
    db: Path = typer.Option(
        Path("data/arxiv_rag.db"),
        help="SQLite database path.",
        exists=False,
        dir_okay=False,
    ),
    chroma_dir: Path = typer.Option(
        Path("data/chroma"),
        envvar="CHROMA_DIR",
        help="Chroma persistence directory.",
    ),
    collection: str = typer.Option(
        "arxiv_chunks_te3s_v1",
        envvar="CHROMA_COLLECTION",
        help="Chroma collection name.",
    ),
    distance: Literal["cosine", "l2", "ip"] = typer.Option(
        "cosine",
        envvar="CHROMA_DISTANCE",
        help="Chroma distance metric.",
    ),
    doc_id: list[str] | None = typer.Option(
        None,
        help="Limit to specific doc_id values (repeatable).",
    ),
    limit: int | None = typer.Option(
        None,
        help="Limit number of doc_ids to inspect.",
    ),
) -> None:
    """Inspect Chroma counts per doc_id."""

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    chroma_config = ChromaConfig(
        persist_dir=chroma_dir,
        collection_name=collection,
        distance=distance,
    )

    try:
        counts = inspect_chroma_counts(
            db_path=db,
            chroma_config=chroma_config,
            doc_ids=doc_id,
            limit=limit,
        )
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except sqlite3.Error as exc:
        typer.echo(f"SQLite error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if not counts:
        typer.echo("No doc_ids found.")
        raise typer.Exit(code=0)

    total = 0
    for entry in counts:
        typer.echo(f"{entry.doc_id}\t{entry.count}")
        total += entry.count
    typer.echo(f"TOTAL\t{total}")


def _load_chunks_to_embed(
    conn: sqlite3.Connection,
    *,
    force: bool,
    limit: int | None,
) -> list[_ChunkToEmbed]:
    """Load chunk rows that require embeddings."""

    where_clause = "" if force else "WHERE embedding IS NULL"
    limit_clause = "" if limit is None else "LIMIT ?"
    sql = f"""
        SELECT chunk_id, text, token_count
        FROM chunks
        {where_clause}
        ORDER BY chunk_id
        {limit_clause}
    """
    if limit is None:
        rows = conn.execute(sql).fetchall()
    else:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [
        _ChunkToEmbed(
            chunk_id=row[0],
            text=row[1],
            token_count=row[2],
        )
        for row in rows
    ]


def _batch_chunks_by_tokens(
    chunks: list[_ChunkToEmbed],
    *,
    encoding: Encoding,
    max_request_tokens: int,
    max_input_tokens: int,
) -> list[list[_ChunkToEmbed]]:
    """Batch chunks to honor per-request and per-input token caps."""

    batches: list[list[_ChunkToEmbed]] = []
    current: list[_ChunkToEmbed] = []
    current_tokens = 0

    for chunk in chunks:
        token_count = chunk.token_count
        if token_count is None:
            token_count = len(encoding.encode(chunk.text, disallowed_special=()))
        if token_count > max_input_tokens:
            raise ValueError(
                f"Chunk {chunk.chunk_id} exceeds max_input_tokens={max_input_tokens}"
            )
        if token_count > max_request_tokens:
            raise ValueError(
                f"Chunk {chunk.chunk_id} exceeds max_request_tokens={max_request_tokens}"
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


if __name__ == "__main__":
    app()
