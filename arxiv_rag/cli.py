"""Typer-based CLI for arxiv-rag."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import typer
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, RateLimitError
from rich.console import Console
from rich.progress import track
from rich.table import Table
from tiktoken import Encoding, get_encoding

from arxiv_rag import chunk as chunk_module
from arxiv_rag import download as download_module
from arxiv_rag import parse as parse_module
from arxiv_rag.arxiv_ids import base_id_from_versioned, is_valid_base_id
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
    build_openings_plan,
    check_eval_set_coverage,
    generate_eval_set,
    load_eval_set,
    load_openings_from_path,
    run_eval,
    save_eval_report,
)
from arxiv_rag.generate import Chunk as GenerationChunk, generate_answer
from arxiv_rag.indexer import index_chunks
from arxiv_rag.retrieve import (
    ChunkResult,
    format_snippet,
    HybridChunkResult,
    rerank_results_for_generation,
    search_fts,
    search_hybrid,
    search_vector_chroma,
)
from arxiv_rag.verify import verify_answer

app = typer.Typer(help="CLI for arXiv RAG utilities.")
console = Console()


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
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except (ImportError, OSError, sqlite3.Error) as exc:
        if isinstance(exc, sqlite3.Error):
            typer.echo(f"SQLite error: {exc}", err=True)
        else:
            typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


def _exit_code_from_typer_exit(exc: typer.Exit) -> int | None:
    """Return a normalized exit code from a Typer/Click Exit exception."""

    exit_code = getattr(exc, "exit_code", None)
    if isinstance(exit_code, int):
        return exit_code
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    return None


def _safe_log_query(db: Path, entry: QueryLogEntry) -> None:
    """Log query entries to SQLite without aborting the CLI."""

    try:
        insert_query_log(db, entry)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Query log error: {exc}", err=True)


@dataclass(frozen=True)
class IngestFailure:
    """Failure metadata for staged corpus ingestion."""

    paper_id: str
    stage: Literal["input", "download", "metadata", "parse", "chunk"]
    message: str


@dataclass(frozen=True)
class IngestSummary:
    """Summary report for add-ids/rebuild ingestion runs."""

    requested_count: int
    parsed_count: int
    chunked_count: int
    indexed_chunks: int
    failures: list[IngestFailure]
    warnings: list[str]


def _load_base_ids_from_file(path: Path) -> list[str]:
    """Load raw arXiv IDs from a .txt or .json file."""

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"IDs file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        if isinstance(payload, list):
            if all(isinstance(item, str) for item in payload):
                return [item.strip() for item in payload if item.strip()]
            raise ValueError("JSON list payload must contain only string IDs.")
        if isinstance(payload, dict) and "ids" in payload:
            ids_value = payload["ids"]
            if isinstance(ids_value, list) and all(
                isinstance(item, str) for item in ids_value
            ):
                return [item.strip() for item in ids_value if item.strip()]
            raise ValueError("JSON object payload requires string list under 'ids'.")
        raise ValueError("JSON payload must be a string list or {'ids': [...]} object.")
    raise ValueError("IDs file must use .txt or .json extension.")


def _normalize_base_ids(raw_ids: list[str]) -> tuple[list[str], list[str]]:
    """Normalize, dedupe, and validate arXiv IDs."""

    normalized_ids: list[str] = []
    invalid_ids: list[str] = []
    seen: set[str] = set()

    for raw_id in raw_ids:
        base_id = base_id_from_versioned(raw_id.strip())
        if not base_id:
            continue
        if base_id in seen:
            continue
        seen.add(base_id)
        if is_valid_base_id(base_id):
            normalized_ids.append(base_id)
        else:
            invalid_ids.append(base_id)

    return normalized_ids, invalid_ids


def _build_download_config(
    *,
    pdf_dir: Path,
    db: Path,
) -> download_module.DownloadConfig:
    """Build a download config with CLI-overridden paths."""

    base_config = download_module.default_config()
    return replace(
        base_config,
        pdf_dir=pdf_dir,
        id_index=pdf_dir / "arxiv_ids.txt",
        default_db_path=db,
    )


def _parse_and_chunk_targets(
    *,
    targets: list[tuple[str, Path]],
    db: Path,
    parsed_dir: Path,
    remove_headers_footers: bool,
    target_tokens: int,
    overlap_tokens: int,
    chunk_encoding: str,
    show_progress: bool,
) -> tuple[int, int, list[str], list[IngestFailure], list[str]]:
    """Parse PDFs and ingest chunk rows for target papers."""

    parsed_dir.mkdir(parents=True, exist_ok=True)
    chunk_config = chunk_module.ChunkConfig(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        encoding_name=chunk_encoding,
    )
    parsed_count = 0
    chunked_count = 0
    doc_ids_for_reindex: list[str] = []
    failures: list[IngestFailure] = []
    warnings: list[str] = []

    iterator = (
        track(targets, description="Parsing + chunking papers")
        if show_progress
        else targets
    )
    with sqlite3.connect(db) as conn:
        for paper_id, pdf_path in iterator:
            try:
                parsed_payload, parse_warnings = parse_module.parse_pdf_with_warnings(
                    pdf_path,
                    remove_headers_footers=remove_headers_footers,
                )
                parsed_count += 1
            except (FileNotFoundError, ValueError) as exc:
                failures.append(
                    IngestFailure(paper_id=paper_id, stage="parse", message=str(exc))
                )
                continue

            warnings.extend(f"{paper_id}: {warning}" for warning in parse_warnings)
            parsed_path = parsed_dir / f"{pdf_path.stem}.json"
            try:
                parsed_path.write_text(
                    json.dumps(parsed_payload, indent=2),
                    encoding="utf-8",
                )
                parsed_doc = chunk_module.load_parsed_document(parsed_path)
                chunks = chunk_module.chunk_document(parsed_doc, paper_id, chunk_config)
                previous_doc_id = chunk_module.ingest_chunks(
                    conn,
                    parsed_doc,
                    paper_id,
                    chunks,
                )
                conn.commit()
            except (OSError, ValueError, sqlite3.Error) as exc:
                conn.rollback()
                failures.append(
                    IngestFailure(paper_id=paper_id, stage="chunk", message=str(exc))
                )
                continue

            chunked_count += 1
            doc_ids_for_reindex.append(parsed_doc.doc_id)
            if previous_doc_id and previous_doc_id != parsed_doc.doc_id:
                doc_ids_for_reindex.append(previous_doc_id)

    unique_doc_ids = list(dict.fromkeys(doc_ids_for_reindex))
    return parsed_count, chunked_count, unique_doc_ids, failures, warnings


def _print_ingest_summary(summary: IngestSummary) -> None:
    """Print a human-readable ingestion summary."""

    table = Table(title="Ingest Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Requested papers", str(summary.requested_count))
    table.add_row("Parsed papers", str(summary.parsed_count))
    table.add_row("Chunked papers", str(summary.chunked_count))
    table.add_row("Indexed chunks", str(summary.indexed_chunks))
    table.add_row("Warnings", str(len(summary.warnings)))
    table.add_row("Failures", str(len(summary.failures)))
    console.print(table)

    for warning in summary.warnings:
        typer.echo(f"Warning: {warning}", err=True)

    if summary.failures:
        typer.echo("Failures:", err=True)
        for failure in summary.failures:
            typer.echo(
                f"- {failure.paper_id} [{failure.stage}] {failure.message}",
                err=True,
            )


def _run_ingest_and_index(
    *,
    add_ids: Path | None,
    rebuild: bool,
    db: Path,
    pdf_dir: Path,
    parsed_dir: Path,
    remove_headers_footers: bool,
    target_tokens: int,
    overlap_tokens: int,
    chunk_encoding: str,
    download_retries: int,
    download_timeout: int,
    show_progress: bool,
    chroma_config: ChromaConfig,
    embeddings_config: EmbeddingsConfig,
    batch_size: int | None,
    force_delete: bool,
) -> IngestSummary:
    """Run add-ids/rebuild ingestion and index updated doc_ids."""

    if add_ids is None and not rebuild:
        raise ValueError("Expected --add-ids or --rebuild for ingest workflow.")

    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be < target_tokens")
    if download_retries <= 0:
        raise ValueError("download_retries must be > 0")
    if download_timeout <= 0:
        raise ValueError("download_timeout must be > 0")

    config = _build_download_config(pdf_dir=pdf_dir, db=db)
    failures: list[IngestFailure] = []
    warnings: list[str] = []

    if add_ids is not None:
        raw_ids = _load_base_ids_from_file(add_ids)
        normalized_ids, invalid_ids = _normalize_base_ids(raw_ids)
        if invalid_ids:
            for base_id in invalid_ids:
                failures.append(
                    IngestFailure(
                        paper_id=base_id,
                        stage="input",
                        message="Invalid arXiv ID format.",
                    )
                )
        if not normalized_ids:
            return IngestSummary(
                requested_count=0,
                parsed_count=0,
                chunked_count=0,
                indexed_chunks=0,
                failures=failures,
                warnings=warnings,
            )

        if show_progress:
            typer.echo(f"Downloading {len(normalized_ids)} requested papers...")
        download_module.download_by_ids(
            normalized_ids,
            retries=download_retries,
            timeout=download_timeout,
            db_path=db,
            config=config,
        )

        missing_metadata_ids = set(
            download_module.upsert_metadata_for_ids(
                normalized_ids,
                db_path=db,
                config=config,
            )
        )
        latest_pdf_paths = download_module.collect_latest_pdf_paths(config)
        targets: list[tuple[str, Path]] = []
        for paper_id in normalized_ids:
            if paper_id in missing_metadata_ids:
                failures.append(
                    IngestFailure(
                        paper_id=paper_id,
                        stage="metadata",
                        message="Missing metadata or downloaded PDF.",
                    )
                )
                continue
            if (pdf_path := latest_pdf_paths.get(paper_id)) is None:
                failures.append(
                    IngestFailure(
                        paper_id=paper_id,
                        stage="download",
                        message="Downloaded PDF not found on disk.",
                    )
                )
                continue
            targets.append((paper_id, pdf_path))
        requested_count = len(normalized_ids)
    else:
        latest_pdf_paths = download_module.collect_latest_pdf_paths(config)
        if not latest_pdf_paths:
            return IngestSummary(
                requested_count=0,
                parsed_count=0,
                chunked_count=0,
                indexed_chunks=0,
                failures=[],
                warnings=[],
            )
        all_ids = sorted(latest_pdf_paths)
        missing_metadata_ids = set(
            download_module.upsert_metadata_for_ids(
                all_ids,
                db_path=db,
                config=config,
            )
        )
        targets = [
            (paper_id, latest_pdf_paths[paper_id])
            for paper_id in all_ids
            if paper_id not in missing_metadata_ids
        ]
        for paper_id in sorted(missing_metadata_ids):
            failures.append(
                IngestFailure(
                    paper_id=paper_id,
                    stage="metadata",
                    message="Missing metadata for local PDF.",
                )
            )
        requested_count = len(all_ids)

    parsed_count, chunked_count, doc_ids, stage_failures, stage_warnings = (
        _parse_and_chunk_targets(
            targets=targets,
            db=db,
            parsed_dir=parsed_dir,
            remove_headers_footers=remove_headers_footers,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            chunk_encoding=chunk_encoding,
            show_progress=show_progress,
        )
    )
    failures.extend(stage_failures)
    warnings.extend(stage_warnings)

    indexed_chunks = 0
    if doc_ids:
        indexed_chunks = index_chunks(
            db_path=db,
            chroma_config=chroma_config,
            embeddings_config=embeddings_config,
            doc_ids=doc_ids,
            limit=None,
            batch_size=batch_size,
            force_delete=force_delete,
        )

    return IngestSummary(
        requested_count=requested_count,
        parsed_count=parsed_count,
        chunked_count=chunked_count,
        indexed_chunks=indexed_chunks,
        failures=failures,
        warnings=warnings,
    )


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
    generation_rerank: Literal["none", "lexical"] = typer.Option(
        "none",
        help="Rerank retrieved chunks before generation.",
    ),
    generation_select_evidence: bool = typer.Option(
        False,
        help="Select a smaller evidence set before generation.",
    ),
    generation_select_k: int = typer.Option(
        3,
        help="Max chunks to keep when evidence selection is enabled.",
    ),
    generation_quote_first: bool = typer.Option(
        False,
        help="Force quote-first generation anchored to one chunk.",
    ),
    generation_cite_chunk_index: bool = typer.Option(
        False,
        help="Use [chunk:N] citations and map to pages after generation.",
    ),
    generation_repair_citations: bool = typer.Option(
        False,
        help="Repair missing or invalid citations after generation.",
    ),
    generation_repair_max_attempts: int = typer.Option(
        1,
        help="Max attempts for citation repair when enabled.",
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
    if generate and generation_select_k <= 0:
        typer.echo("--generation-select-k must be > 0.", err=True)
        raise typer.Exit(code=1)
    if generate and generation_repair_max_attempts <= 0:
        typer.echo("--generation-repair-max-attempts must be > 0.", err=True)
        raise typer.Exit(code=1)
    if generate and generation_cite_chunk_index and generation_quote_first:
        typer.echo(
            "--generation-cite-chunk-index cannot be combined with "
            "--generation-quote-first.",
            err=True,
        )
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

        if generation_rerank == "lexical":
            results = rerank_results_for_generation(question, results)

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
                select_evidence=generation_select_evidence,
                selection_max_chunks=generation_select_k,
                quote_first=generation_quote_first,
                cite_chunk_index=generation_cite_chunk_index,
                repair_citations=generation_repair_citations,
                repair_max_attempts=generation_repair_max_attempts,
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
        exit_code = _exit_code_from_typer_exit(exc)
        if log_queries and (exit_code == 0 or exit_code is None):
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
        0.8,
        help="Sampling temperature.",
    ),
    top_p: float = typer.Option(
        0.9,
        help="Top-p nucleus sampling for QA generation.",
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
    openings_path: Path | None = typer.Option(
        None,
        help="Optional path to a JSON list or newline-delimited openings list.",
    ),
    corpus_version: str = typer.Option(
        "v1",
        help="Corpus version label to store in metadata.",
    ),
) -> None:
    """Generate a synthetic eval set from existing chunks."""

    logging.basicConfig(level=logging.INFO)
    try:
        openings_plan = (
            load_openings_from_path(openings_path)
            if openings_path is not None
            else build_openings_plan(n_questions, seed=seed)
        )
        eval_set = generate_eval_set(
            db_path=db,
            output_path=output,
            n_questions=n_questions,
            questions_per_chunk=questions_per_chunk,
            seed=seed,
            min_chars=min_chars,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            request_timeout_s=request_timeout_s,
            max_retries=max_retries,
            prompt_path=prompt_path,
            corpus_version=corpus_version,
            openings_plan=openings_plan,
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
    generation_rerank: Literal["none", "lexical"] = typer.Option(
        "none",
        help="Rerank retrieved chunks before generation.",
    ),
    generation_select_evidence: bool = typer.Option(
        False,
        help="Select a smaller evidence set before generation.",
    ),
    generation_select_k: int = typer.Option(
        3,
        help="Max chunks to keep when evidence selection is enabled.",
    ),
    generation_quote_first: bool = typer.Option(
        False,
        help="Force quote-first generation anchored to one chunk.",
    ),
    generation_cite_chunk_index: bool = typer.Option(
        False,
        help="Use [chunk:N] citations and map to pages after generation.",
    ),
    generation_repair_citations: bool = typer.Option(
        False,
        help="Repair missing or invalid citations after generation.",
    ),
    generation_repair_max_attempts: int = typer.Option(
        1,
        help="Max attempts for citation repair when enabled.",
    ),
    generation_concurrency: int = typer.Option(
        4,
        help="Maximum concurrent generation calls when --generate is enabled.",
    ),
    cache_db: Path = typer.Option(
        Path("eval/cache/eval_cache.db"),
        help="SQLite cache path for eval embeddings and generated answers.",
    ),
    disable_cache: bool = typer.Option(
        False,
        "--disable-cache",
        help="Disable persistent eval caching.",
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
            generation_rerank=generation_rerank,
            generation_select_evidence=generation_select_evidence,
            generation_select_k=generation_select_k,
            generation_quote_first=generation_quote_first,
            generation_cite_chunk_index=generation_cite_chunk_index,
            generation_repair_citations=generation_repair_citations,
            generation_repair_max_attempts=generation_repair_max_attempts,
            generation_concurrency=generation_concurrency,
            cache_db_path=None if disable_cache else cache_db,
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
    add_ids: Path | None = typer.Option(
        None,
        help="Path to a .txt/.json file of arXiv IDs to download and ingest.",
    ),
    rebuild: bool = typer.Option(
        False,
        help="Rebuild parse/chunks/vector index from local PDFs.",
    ),
    pdf_dir: Path = typer.Option(
        Path("data/arxiv-papers"),
        help="Directory containing downloaded arXiv PDFs.",
    ),
    parsed_dir: Path = typer.Option(
        Path("data/parsed"),
        help="Directory to write parsed JSON files.",
    ),
    remove_headers_footers: bool = typer.Option(
        False,
        help="Remove repeated page headers/footers during parsing.",
    ),
    target_tokens: int = typer.Option(
        512,
        help="Target token size for chunking.",
    ),
    overlap_tokens: int = typer.Option(
        100,
        help="Token overlap between adjacent chunks.",
    ),
    chunk_encoding: str = typer.Option(
        "cl100k_base",
        help="Tokenizer encoding used for chunking.",
    ),
    download_retries: int = typer.Option(
        3,
        help="Download retries per paper when using --add-ids.",
    ),
    download_timeout: int = typer.Option(
        30,
        help="Download timeout in seconds when using --add-ids.",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show progress bars for parsing/chunking.",
    ),
) -> None:
    """Generate embeddings or run full ingest + index workflows."""

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("arxiv").setLevel(logging.WARNING)
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

    if add_ids and rebuild:
        typer.echo("--add-ids and --rebuild cannot be used together.", err=True)
        raise typer.Exit(code=1)

    if add_ids is not None or rebuild:
        if doc_id:
            typer.echo("--doc-id cannot be used with --add-ids/--rebuild.", err=True)
            raise typer.Exit(code=1)
        if limit is not None:
            typer.echo("--limit cannot be used with --add-ids/--rebuild.", err=True)
            raise typer.Exit(code=1)
        try:
            summary = _run_ingest_and_index(
                add_ids=add_ids,
                rebuild=rebuild,
                db=db,
                pdf_dir=pdf_dir,
                parsed_dir=parsed_dir,
                remove_headers_footers=remove_headers_footers,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
                chunk_encoding=chunk_encoding,
                download_retries=download_retries,
                download_timeout=download_timeout,
                show_progress=progress,
                chroma_config=chroma_config,
                embeddings_config=embeddings_config,
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

        if summary.requested_count == 0 and not summary.failures:
            typer.echo("No papers found to ingest.")
            raise typer.Exit(code=0)

        _print_ingest_summary(summary)
        if summary.failures:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

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
