"""Typer-based CLI for arxiv-rag."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import typer
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, RateLimitError
from tiktoken import Encoding, get_encoding

from arxiv_rag.chroma_client import ChromaConfig
from arxiv_rag.chroma_inspect import inspect_chroma_counts
from arxiv_rag.db import (
    QueryLogEntry,
    insert_query_log,
    normalize_embedding,
    serialize_embedding,
)
from arxiv_rag.embeddings_client import EmbeddingsClient, EmbeddingsConfig
from arxiv_rag.evaluate import (
    RetrievalConfig,
    check_eval_set_coverage,
    generate_eval_set,
    load_eval_set,
    run_eval,
    save_eval_report,
)
from arxiv_rag.generate import Chunk as GenerationChunk, generate_answer
from arxiv_rag.indexer import index_chunks
from arxiv_rag.retrieve import (
    ChunkResult,
    format_snippet,
    HybridChunkResult,
    search_fts,
    search_hybrid,
    search_vector_chroma,
)
from arxiv_rag.verify import verify_answer

app = typer.Typer(help="CLI for arXiv RAG utilities.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Show help when no subcommand is provided."""

    load_dotenv()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _run_query(
    *,
    question: str,
    top_k: int,
    mode: Literal["fts", "vector", "hybrid"],
    model: str,
    db: Path,
    chroma_dir: Path,
    collection: str,
    distance: Literal["cosine", "l2", "ip"],
    snippet_chars: int,
    show_score: bool,
    verbose: bool,
    rrf_k: int,
    rrf_weight_fts: float,
    rrf_weight_vector: float,
) -> tuple[list[ChunkResult | HybridChunkResult], list[str]]:
    """Run a retrieval query and print results to stdout.

    Args:
        question: Query string to retrieve chunks for.
        top_k: Number of chunks to return.
        mode: Retrieval backend to use.
        model: Embedding model name.
        db: SQLite database path.
        chroma_dir: Chroma persistence directory.
        collection: Chroma collection name.
        distance: Chroma distance metric.
        snippet_chars: Maximum characters to display per snippet.
        show_score: Whether to include backend scores in the output.
        verbose: Whether to print retrieval provenance details.
        rrf_k: RRF constant for hybrid fusion.
        rrf_weight_fts: RRF weight for FTS results.
        rrf_weight_vector: RRF weight for vector results.
    Returns:
        Tuple of retrieved results and warning messages.
    Raises:
        typer.Exit: If validation fails or retrieval errors occur.
    """

    if snippet_chars <= 0:
        typer.echo("snippet_chars must be > 0", err=True)
        raise typer.Exit(code=1)

    results, warnings = _run_retrieval(
        question=question,
        top_k=top_k,
        mode=mode,
        model=model,
        db=db,
        chroma_dir=chroma_dir,
        collection=collection,
        distance=distance,
        rrf_k=rrf_k,
        rrf_weight_fts=rrf_weight_fts,
        rrf_weight_vector=rrf_weight_vector,
    )
    for warning in warnings:
        typer.echo(f"Warning: {warning}", err=True)

    if not results:
        typer.echo("No results found.")
        raise typer.Exit(code=0)

    if verbose and mode == "hybrid":
        typer.echo("Raw scores: lower is better for BM25 and Chroma distance.")

    for index, result in enumerate(results):
        snippet = format_snippet(result.text, snippet_chars)
        line = f"[arXiv:{result.paper_id} p.{result.page_number}] {snippet}"
        if show_score:
            if mode == "hybrid":
                line = f"{line} (rrf={result.rrf_score:.6f})"
            elif result.score is not None:
                line = f"{line} (score={result.score:.3f})"
        typer.echo(line)
        if verbose:
            if mode == "hybrid":
                backends = "+".join(
                    sorted({prov.backend for prov in result.provenance})
                )
                typer.echo(f"  sources={backends}")
                typer.echo(f"  rrf_total={result.rrf_score:.6f}")
                for prov in result.provenance:
                    raw_value = (
                        "n/a" if prov.raw_score is None else f"{prov.raw_score:.6f}"
                    )
                    typer.echo(
                        "  "
                        f"{prov.backend} "
                        f"rank={prov.rank} "
                        f"raw={raw_value} "
                        f"norm={prov.normalized_score:.6f} "
                        f"rrf={prov.rrf_contribution:.6f}"
                    )
            elif result.score is not None:
                typer.echo(f"  score={result.score:.6f}")
        if index < len(results) - 1:
            typer.echo()
    return results, warnings


def _retrieve_results(
    *,
    question: str,
    top_k: int,
    mode: Literal["fts", "vector", "hybrid"],
    model: str,
    db: Path,
    chroma_dir: Path,
    collection: str,
    distance: Literal["cosine", "l2", "ip"],
    rrf_k: int,
    rrf_weight_fts: float,
    rrf_weight_vector: float,
) -> tuple[list[ChunkResult | HybridChunkResult], list[str]]:
    """Retrieve chunks for a query without printing results."""

    warnings: list[str] = []
    if mode == "fts":
        results = search_fts(question, top_k=top_k, db_path=db)
    elif mode == "vector":
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
    else:
        embeddings_client = EmbeddingsClient(EmbeddingsConfig(model=model))
        chroma_config = ChromaConfig(
            persist_dir=chroma_dir,
            collection_name=collection,
            distance=distance,
        )
        output = search_hybrid(
            question,
            top_k=top_k,
            db_path=db,
            embeddings_client=embeddings_client,
            chroma_config=chroma_config,
            rrf_k=rrf_k,
            fts_weight=rrf_weight_fts,
            vector_weight=rrf_weight_vector,
        )
        warnings.extend(output.warnings)
        results = output.results

    return results, warnings


def _run_retrieval(
    *,
    question: str,
    top_k: int,
    mode: Literal["fts", "vector", "hybrid"],
    model: str,
    db: Path,
    chroma_dir: Path,
    collection: str,
    distance: Literal["cosine", "l2", "ip"],
    rrf_k: int,
    rrf_weight_fts: float,
    rrf_weight_vector: float,
) -> tuple[list[ChunkResult | HybridChunkResult], list[str]]:
    """Run retrieval with CLI-oriented error handling."""

    try:
        return _retrieve_results(
            question=question,
            top_k=top_k,
            mode=mode,
            model=model,
            db=db,
            chroma_dir=chroma_dir,
            collection=collection,
            distance=distance,
            rrf_k=rrf_k,
            rrf_weight_fts=rrf_weight_fts,
            rrf_weight_vector=rrf_weight_vector,
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


def _safe_log_query(db: Path, entry: QueryLogEntry) -> None:
    """Log query entries to SQLite without aborting the CLI."""

    try:
        insert_query_log(db, entry)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Query log error: {exc}", err=True)


@app.command()
def query(
    question: str,
    top_k: int = typer.Option(5, help="Number of chunks to return."),
    mode: Literal["fts", "vector", "hybrid"] = typer.Option(
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
    generate: bool = typer.Option(
        False,
        help="Generate a cited answer using retrieved chunks.",
    ),
    generate_model: str = typer.Option(
        "gpt-4o-mini",
        help="Model for answer generation.",
    ),
    verify: bool = typer.Option(
        False,
        help="Verify citations and quotes in the generated answer.",
    ),
    parsed_dir: Path = typer.Option(
        Path("data/parsed"),
        help="Directory containing parsed JSON files for quote validation.",
    ),
    show_score: bool = typer.Option(
        False,
        help="Include score in the output.",
    ),
    verbose: bool = typer.Option(
        False,
        help="Show retrieval details.",
    ),
    rrf_k: int = typer.Option(
        60,
        help="RRF constant for hybrid fusion.",
    ),
    rrf_weight_fts: float = typer.Option(
        1.0,
        help="RRF weight for FTS results.",
    ),
    rrf_weight_vector: float = typer.Option(
        1.0,
        help="RRF weight for vector results.",
    ),
    log_queries: bool = typer.Option(
        True,
        "--log/--no-log",
        help="Log queries to SQLite.",
    ),
) -> None:
    """Query SQLite for matching chunks via FTS, vector, or hybrid retrieval."""

    if verify and not generate:
        typer.echo("--verify requires --generate.", err=True)
        raise typer.Exit(code=1)

    start_time = time.perf_counter()

    if generate:
        results, warnings = _run_retrieval(
            question=question,
            top_k=top_k,
            mode=mode,
            model=model,
            db=db,
            chroma_dir=chroma_dir,
            collection=collection,
            distance=distance,
            rrf_k=rrf_k,
            rrf_weight_fts=rrf_weight_fts,
            rrf_weight_vector=rrf_weight_vector,
        )
        for warning in warnings:
            typer.echo(f"Warning: {warning}", err=True)

        if not results:
            typer.echo("No results found.")
            if log_queries:
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                _safe_log_query(
                    db,
                    QueryLogEntry(
                        query_text=question,
                        retrieved_chunks=[],
                        answer=None,
                        latency_ms=latency_ms,
                        model=f"retrieval={mode}:{model};generation={generate_model}",
                    ),
                )
            raise typer.Exit(code=0)

        chunks = [
            GenerationChunk(
                chunk_uid=result.chunk_uid,
                paper_id=result.paper_id,
                page_number=result.page_number,
                text=result.text,
            )
            for result in results
        ]

        try:
            answer = generate_answer(
                question,
                chunks,
                model=generate_model,
            )
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        except FileNotFoundError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
        except (APIError, APITimeoutError, RateLimitError) as exc:
            typer.echo(f"OpenAI error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc

        typer.echo(answer)

        if log_queries:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            _safe_log_query(
                db,
                QueryLogEntry(
                    query_text=question,
                    retrieved_chunks=[result.chunk_uid for result in results],
                    answer=answer,
                    latency_ms=latency_ms,
                    model=f"retrieval={mode}:{model};generation={generate_model}",
                ),
            )

        if verify:
            try:
                report = verify_answer(
                    answer,
                    db_path=db,
                    parsed_dir=parsed_dir,
                )
            except FileNotFoundError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=2) from exc
            except sqlite3.Error as exc:
                typer.echo(f"SQLite error: {exc}", err=True)
                raise typer.Exit(code=2) from exc
            except OSError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(code=2) from exc

            typer.echo("\nVerification report:")
            if not report.errors:
                typer.echo("  pass")
            else:
                for error in report.errors:
                    suffix = []
                    if error.citation_id:
                        suffix.append(f"citation={error.citation_id}")
                    if error.sentence_index is not None:
                        suffix.append(f"sentence={error.sentence_index}")
                    if error.paragraph_index is not None:
                        suffix.append(f"paragraph={error.paragraph_index}")
                    details = f" ({', '.join(suffix)})" if suffix else ""
                    typer.echo(f"  {error.code}: {error.message}{details}")

            if report.status != "pass":
                raise typer.Exit(code=1)
        return

    try:
        results, _warnings = _run_query(
            question=question,
            top_k=top_k,
            mode=mode,
            model=model,
            db=db,
            chroma_dir=chroma_dir,
            collection=collection,
            distance=distance,
            snippet_chars=snippet_chars,
            show_score=show_score,
            verbose=verbose,
            rrf_k=rrf_k,
            rrf_weight_fts=rrf_weight_fts,
            rrf_weight_vector=rrf_weight_vector,
        )
    except typer.Exit as exc:
        if log_queries and (exc.code == 0 or exc.code is None):
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            _safe_log_query(
                db,
                QueryLogEntry(
                    query_text=question,
                    retrieved_chunks=[],
                    answer=None,
                    latency_ms=latency_ms,
                    model=f"retrieval={mode}:{model}",
                ),
            )
        raise

    if log_queries:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        _safe_log_query(
            db,
            QueryLogEntry(
                query_text=question,
                retrieved_chunks=[result.chunk_uid for result in results],
                answer=None,
                latency_ms=latency_ms,
                model=f"retrieval={mode}:{model}",
            ),
        )


@app.command()
def sources(
    question: str,
    top_k: int = typer.Option(5, help="Number of chunks to return."),
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
        help="Include score in the output.",
    ),
    rrf_k: int = typer.Option(
        60,
        help="RRF constant for hybrid fusion.",
    ),
    rrf_weight_fts: float = typer.Option(
        1.0,
        help="RRF weight for FTS results.",
    ),
    rrf_weight_vector: float = typer.Option(
        1.0,
        help="RRF weight for vector results.",
    ),
) -> None:
    """Alias for hybrid retrieval with verbose provenance output."""

    _run_query(
        question=question,
        top_k=top_k,
        mode="hybrid",
        model=model,
        db=db,
        chroma_dir=chroma_dir,
        collection=collection,
        distance=distance,
        snippet_chars=snippet_chars,
        show_score=show_score,
        verbose=True,
        rrf_k=rrf_k,
        rrf_weight_fts=rrf_weight_fts,
        rrf_weight_vector=rrf_weight_vector,
    )


@app.command("eval-generate")
def eval_generate(
    db: Path = typer.Option(
        Path("data/arxiv_rag.db"),
        help="SQLite database path.",
        exists=False,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        Path("eval/eval_set.json"),
        help="Output path for the eval set JSON.",
    ),
    n_questions: int = typer.Option(
        50,
        help="Total number of questions to generate.",
    ),
    questions_per_chunk: int = typer.Option(
        1,
        help="Questions to request per chunk.",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed for chunk sampling.",
    ),
    min_chars: int = typer.Option(
        200,
        help="Minimum chunk text length.",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        help="Model for QA generation.",
    ),
    temperature: float = typer.Option(
        0.2,
        help="Sampling temperature.",
    ),
    max_output_tokens: int = typer.Option(
        800,
        help="Maximum output tokens per request.",
    ),
    request_timeout_s: float = typer.Option(
        60.0,
        help="Timeout in seconds per request.",
    ),
    max_retries: int = typer.Option(
        5,
        help="Maximum retry attempts per request.",
    ),
    prompt_path: Path | None = typer.Option(
        None,
        help="Optional override for the QA prompt template.",
    ),
    corpus_version: str = typer.Option(
        "v1",
        help="Corpus version label to store in metadata.",
    ),
) -> None:
    """Generate a synthetic eval set from existing chunks."""

    logging.basicConfig(level=logging.INFO)
    try:
        eval_set = generate_eval_set(
            db_path=db,
            output_path=output,
            n_questions=n_questions,
            questions_per_chunk=questions_per_chunk,
            seed=seed,
            min_chars=min_chars,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            request_timeout_s=request_timeout_s,
            max_retries=max_retries,
            prompt_path=prompt_path,
            corpus_version=corpus_version,
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except (APIError, APITimeoutError, RateLimitError) as exc:
        typer.echo(f"OpenAI error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except sqlite3.Error as exc:
        typer.echo(f"SQLite error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"Wrote {len(eval_set.eval_set)} questions to {output}")


@app.command("eval-run")
def eval_run(
    eval_set_path: Path = typer.Option(
        Path("eval/eval_set.json"),
        help="Path to the eval set JSON.",
        exists=False,
        dir_okay=False,
    ),
    db: Path = typer.Option(
        Path("data/arxiv_rag.db"),
        help="SQLite database path.",
        exists=False,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        Path("eval/results"),
        help="Directory to store eval reports.",
    ),
    mode: Literal["fts", "vector", "hybrid"] = typer.Option(
        "hybrid",
        help="Retrieval mode.",
    ),
    top_k: int = typer.Option(
        10,
        help="Number of chunks to retrieve per query.",
    ),
    model: str = typer.Option(
        "text-embedding-3-small",
        help="Embedding model for vector search.",
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
    rrf_k: int = typer.Option(
        60,
        help="RRF constant for hybrid fusion.",
    ),
    rrf_weight_fts: float = typer.Option(
        1.0,
        help="RRF weight for FTS results.",
    ),
    rrf_weight_vector: float = typer.Option(
        1.0,
        help="RRF weight for vector results.",
    ),
    generate: bool = typer.Option(
        False,
        help="Generate answers and compute citation accuracy.",
    ),
    generate_model: str = typer.Option(
        "gpt-4o-mini",
        help="Model for answer generation.",
    ),
    generation_top_k: int = typer.Option(
        5,
        help="Number of chunks to pass into generation.",
    ),
) -> None:
    """Run evaluation and write a report to disk."""

    logging.basicConfig(level=logging.INFO)
    try:
        eval_set = load_eval_set(eval_set_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    try:
        coverage = check_eval_set_coverage(eval_set, db_path=db)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if coverage.missing_papers:
        typer.echo(
            f"Warning: {len(coverage.missing_papers)} papers missing from DB.",
            err=True,
        )

    retrieval_config = RetrievalConfig(
        mode=mode,
        top_k=top_k,
        model=model,
        chroma_dir=chroma_dir,
        collection=collection,
        distance=distance,
        rrf_k=rrf_k,
        rrf_weight_fts=rrf_weight_fts,
        rrf_weight_vector=rrf_weight_vector,
    )
    effective_generation_top_k = min(generation_top_k, top_k)

    try:
        report = run_eval(
            eval_set=eval_set,
            db_path=db,
            retrieval_config=retrieval_config,
            generate=generate,
            generate_model=generate_model,
            generation_top_k=effective_generation_top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except (APIError, APITimeoutError, RateLimitError) as exc:
        typer.echo(f"OpenAI error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except sqlite3.Error as exc:
        typer.echo(f"SQLite error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = output_dir / f"eval_report_{timestamp}"
    save_eval_report(report, output_path=output_base)

    citation_display = (
        f"{report.summary.citation_accuracy:.3f}"
        if report.summary.citation_accuracy is not None
        else "N/A"
    )
    typer.echo(
        f"Recall@5={report.summary.recall_at_5:.3f} "
        f"MRR={report.summary.mrr:.3f} "
        f"CitationAccuracy={citation_display}"
    )
    typer.echo(f"Wrote reports to {output_base}.json and {output_base}.md")

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
    force_delete: bool = typer.Option(
        False,
        help="Allow deleting existing vectors even when --limit is set.",
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
            force_delete=force_delete,
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
