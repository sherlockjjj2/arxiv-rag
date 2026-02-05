"""Retrieve chunks from SQLite FTS indexes."""

from __future__ import annotations

import heapq
import re
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from arxiv_rag.chroma_client import ChromaConfig, ChromaStore
from arxiv_rag.db import deserialize_embedding_array, normalize_embedding
from arxiv_rag.embeddings_client import EmbeddingsClient

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "which",
    "with",
}
_MAX_REQUIRED_IMPORTANT_TERMS = 4
_GENERATION_RERANK_LEXICAL_WEIGHT = 0.7
_GENERATION_RERANK_RANK_WEIGHT = 0.3


def _extract_fts_tokens(question: str) -> list[str]:
    """Extract query tokens suitable for FTS matching.

    Args:
        question: Raw user query string.
    Returns:
        List of token strings in appearance order.
    Edge cases:
        Falls back to whitespace splitting when regex extraction fails.
    """

    tokens = re.findall(r"\w+", question, flags=re.UNICODE)
    if tokens:
        return tokens

    raw_tokens = [token for token in re.split(r"\s+", question.strip()) if token]
    cleaned = [
        re.sub(r"^[^\w]+|[^\w]+$", "", token, flags=re.UNICODE) for token in raw_tokens
    ]
    return [token for token in cleaned if token]


def _select_required_terms(terms: Sequence[str], max_terms: int) -> list[str]:
    """Select a limited number of required terms by length.

    Args:
        terms: Candidate required terms in appearance order.
        max_terms: Maximum number of terms to require.
    Returns:
        Selected terms in their original order.
    """

    if max_terms <= 0 or len(terms) <= max_terms:
        return list(terms)

    ranked = sorted(
        enumerate(terms),
        key=lambda item: (-len(item[1]), item[0]),
    )
    chosen_indices = sorted(index for index, _ in ranked[:max_terms])
    return [terms[index] for index in chosen_indices]


def _tokenize_for_rerank(text: str) -> list[str]:
    """Tokenize text for lexical reranking.

    Args:
        text: Input text to tokenize.
    Returns:
        Lowercased alphanumeric tokens.
    """

    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class ChunkResult:
    """Chunk search result from the FTS index."""

    chunk_uid: str
    chunk_id: int
    paper_id: str
    page_number: int
    text: str
    score: float | None = None


@dataclass(frozen=True)
class BackendProvenance:
    """Per-backend contribution to a hybrid retrieval result.

    Args:
        backend: Backend identifier.
        rank: 1-based rank within the backend results.
        raw_score: Raw backend score or distance (lower is better).
        normalized_score: Rank-normalized score for display (higher is better).
        rrf_contribution: Contribution to the RRF score.
    """

    backend: Literal["fts", "vector"]
    rank: int
    raw_score: float | None
    normalized_score: float
    rrf_contribution: float


@dataclass(frozen=True)
class HybridChunkResult:
    """Hybrid retrieval result with RRF score and provenance."""

    chunk_uid: str
    chunk_id: int
    paper_id: str
    page_number: int
    text: str
    rrf_score: float
    provenance: Sequence[BackendProvenance]


@dataclass(frozen=True)
class HybridSearchOutput:
    """Hybrid search output with warnings."""

    results: list[HybridChunkResult]
    warnings: list[str]


def build_fts_query(question: str) -> str:
    """Build a relaxed FTS query from the question text.

    Args:
        question: Raw user query string.
    Returns:
        FTS query string mixing AND/OR based on term importance.
    Edge cases:
        Returns an empty string when no tokens are present.
    """

    tokens = _extract_fts_tokens(question)
    if not tokens:
        return ""

    filtered = [token for token in tokens if token.lower() not in _STOPWORDS]
    if not filtered:
        return ""

    important = [token for token in filtered if len(token) >= 6]
    common = [token for token in filtered if token not in important]

    if important:
        required_terms = _select_required_terms(
            important,
            _MAX_REQUIRED_IMPORTANT_TERMS,
        )
        required = " AND ".join(required_terms)
        optional_terms = [token for token in filtered if token not in required_terms]
        if optional_terms:
            optional = " OR ".join(optional_terms)
            return f"({required}) AND ({optional})"
        return required

    return " OR ".join(common)


def _build_relaxed_fts_query(question: str) -> str:
    """Build a fallback FTS query that ORs all filtered terms.

    Args:
        question: Raw user query string.
    Returns:
        OR-only FTS query string, or empty string when no tokens remain.
    """

    tokens = _extract_fts_tokens(question)
    if not tokens:
        return ""
    filtered = [token for token in tokens if token.lower() not in _STOPWORDS]
    if not filtered:
        return ""
    return " OR ".join(filtered)


def format_snippet(text: str, max_chars: int) -> str:
    """Normalize whitespace and truncate to a maximum character count.

    Args:
        text: Raw chunk text.
        max_chars: Maximum length of the returned snippet.
    Returns:
        Snippet with collapsed whitespace and optional truncation.
    Raises:
        ValueError: When max_chars is not positive.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return f"{normalized[: max_chars - 3].rstrip()}..."


def search_fts(
    question: str,
    *,
    top_k: int,
    db_path: Path,
) -> list[ChunkResult]:
    """Search SQLite FTS5 for relevant chunks using BM25.

    Args:
        question: Query text.
        top_k: Number of results to return.
        db_path: Path to the SQLite database.
    Returns:
        List of ChunkResult rows ordered by BM25 score.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When top_k is not positive.
        sqlite3.Error: When SQLite operations fail.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    fts_query = build_fts_query(question)
    if not fts_query:
        return []
    relaxed_query = _build_relaxed_fts_query(question)

    sql = """
        SELECT
            chunks.chunk_uid,
            chunks.chunk_id,
            chunks.paper_id,
            chunks.page_number,
            chunks.text,
            bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks ON chunks_fts.rowid = chunks.chunk_id
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, (fts_query, top_k)).fetchall()
        if not rows and relaxed_query and relaxed_query != fts_query:
            rows = conn.execute(sql, (relaxed_query, top_k)).fetchall()

    return [
        ChunkResult(
            chunk_uid=row[0],
            chunk_id=row[1],
            paper_id=row[2],
            page_number=row[3],
            text=row[4],
            score=row[5],
        )
        for row in rows
    ]


def rerank_results_for_generation(
    query: str,
    results: Sequence[ChunkResult | HybridChunkResult],
    *,
    lexical_weight: float = _GENERATION_RERANK_LEXICAL_WEIGHT,
    rank_weight: float = _GENERATION_RERANK_RANK_WEIGHT,
) -> list[ChunkResult | HybridChunkResult]:
    """Rerank retrieval results for generation using lexical overlap + rank.

    Args:
        query: User query text.
        results: Retrieved chunks in original rank order.
        lexical_weight: Weight applied to lexical overlap score.
        rank_weight: Weight applied to original rank score.
    Returns:
        Reranked results list (stable on ties).
    Raises:
        ValueError: If weights are negative or both are zero.
    """

    if lexical_weight < 0 or rank_weight < 0:
        raise ValueError("rerank weights must be >= 0")
    if lexical_weight == 0 and rank_weight == 0:
        raise ValueError("rerank weights cannot both be zero")
    if not results:
        return []

    query_tokens = set(_tokenize_for_rerank(query))
    if not query_tokens:
        return list(results)

    scored: list[tuple[float, int, ChunkResult | HybridChunkResult]] = []
    for index, result in enumerate(results):
        chunk_tokens = set(_tokenize_for_rerank(result.text))
        if chunk_tokens:
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
        else:
            overlap = 0.0
        rank_score = 1.0 / (index + 1)
        score = (lexical_weight * overlap) + (rank_weight * rank_score)
        scored.append((score, index, result))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored]


def search_vector(
    question: str,
    *,
    top_k: int,
    db_path: Path,
    embeddings_client: EmbeddingsClient,
) -> list[ChunkResult]:
    """Search chunks by cosine similarity using stored embeddings.

    Args:
        question: Query text.
        top_k: Number of results to return.
        db_path: Path to the SQLite database.
        embeddings_client: Client for query embeddings.
    Returns:
        List of ChunkResult rows ordered by cosine similarity.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When top_k is not positive.
        sqlite3.Error: When SQLite operations fail.
    """

    query_embedding = embeddings_client.embed([question]).embeddings[0]
    return search_vector_with_embedding(
        query_embedding,
        top_k=top_k,
        db_path=db_path,
    )


def search_vector_chroma(
    question: str,
    *,
    top_k: int,
    db_path: Path,
    embeddings_client: EmbeddingsClient,
    chroma_config: ChromaConfig,
    chroma_store: ChromaStore | None = None,
) -> list[ChunkResult]:
    """Search chunks by cosine similarity using Chroma.

    Args:
        question: Query text.
        top_k: Number of results to return.
        db_path: Path to the SQLite database.
        embeddings_client: Client for query embeddings.
        chroma_config: Chroma configuration.
        chroma_store: Optional store override for testing.
    Returns:
        List of ChunkResult rows ordered by Chroma distance.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When top_k is not positive.
        sqlite3.Error: When SQLite operations fail.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    query_embedding = embeddings_client.embed([question]).embeddings[0]
    return search_vector_chroma_with_embedding(
        query_embedding,
        top_k=top_k,
        db_path=db_path,
        chroma_config=chroma_config,
        chroma_store=chroma_store,
    )


def search_vector_chroma_with_embedding(
    query_embedding: Sequence[float],
    *,
    top_k: int,
    db_path: Path,
    chroma_config: ChromaConfig,
    chroma_store: ChromaStore | None = None,
) -> list[ChunkResult]:
    """Search chunks by cosine similarity using Chroma and a provided embedding.

    Args:
        query_embedding: Pre-computed query embedding.
        top_k: Number of results to return.
        db_path: Path to the SQLite database.
        chroma_config: Chroma configuration.
        chroma_store: Optional store override for testing.
    Returns:
        List of ChunkResult rows ordered by Chroma distance.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When top_k is not positive.
        sqlite3.Error: When SQLite operations fail.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    normalized_query = normalize_embedding(query_embedding)
    store = chroma_store or ChromaStore(chroma_config)
    ids, distances = store.query(
        query_embeddings=[normalized_query],
        top_k=top_k,
    )
    if not ids:
        return []

    scores_by_uid = {
        uid: distances[index] for index, uid in enumerate(ids) if index < len(distances)
    }
    with sqlite3.connect(db_path) as conn:
        return load_chunks_by_uid(conn, ids, scores_by_uid)


def load_chunks_by_uid(
    conn: sqlite3.Connection,
    chunk_uids: Sequence[str],
    scores_by_uid: Mapping[str, float] | None = None,
) -> list[ChunkResult]:  # sourcery skip: for-append-to-extend
    """Load chunk rows for a set of chunk_uids, preserving order.

    Args:
        conn: SQLite connection.
        chunk_uids: Chunk UID values to load.
        scores_by_uid: Optional map of scores to attach.
    Returns:
        List of ChunkResult entries in chunk_uids order.
    """

    if not chunk_uids:
        return []

    placeholders = ", ".join("?" for _ in chunk_uids)
    sql = f"""
        SELECT chunk_uid, chunk_id, paper_id, page_number, text
        FROM chunks
        WHERE chunk_uid IN ({placeholders})
    """
    rows = conn.execute(sql, list(chunk_uids)).fetchall()
    by_uid = {
        row[0]: ChunkResult(
            chunk_uid=row[0],
            chunk_id=row[1],
            paper_id=row[2],
            page_number=row[3],
            text=row[4],
            score=(scores_by_uid or {}).get(row[0]),
        )
        for row in rows
    }

    ordered: list[ChunkResult] = []
    for uid in chunk_uids:
        if uid in by_uid:
            ordered.append(by_uid[uid])
    return ordered


def search_hybrid(
    question: str,
    *,
    top_k: int,
    db_path: Path,
    embeddings_client: EmbeddingsClient,
    chroma_config: ChromaConfig,
    chroma_store: ChromaStore | None = None,
    rrf_k: int = 60,
    fts_weight: float = 1.0,
    vector_weight: float = 1.0,
    candidate_multiplier: int = 2,
    use_sqlite_fallback: bool = True,
) -> HybridSearchOutput:
    """Search chunks using FTS + vector backends with RRF fusion.

    Args:
        question: Query text.
        top_k: Number of fused results to return.
        db_path: Path to the SQLite database.
        embeddings_client: Client for query embeddings.
        chroma_config: Chroma configuration.
        chroma_store: Optional store override for testing.
        rrf_k: RRF constant; larger values reduce rank bias.
        fts_weight: Weight applied to FTS contributions.
        vector_weight: Weight applied to vector contributions.
        candidate_multiplier: Multiplier for per-backend candidate pools.
        use_sqlite_fallback: Whether to fallback to SQLite embeddings if Chroma fails.
    Returns:
        HybridSearchOutput containing results and warnings.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When parameters are invalid.
        sqlite3.Error: When SQLite operations fail.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be > 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")
    if fts_weight < 0 or vector_weight < 0:
        raise ValueError("weights must be >= 0")
    if fts_weight == 0 and vector_weight == 0:
        raise ValueError("at least one weight must be > 0")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    warnings: list[str] = []
    candidate_k = top_k * candidate_multiplier

    fts_results = (
        search_fts(question, top_k=candidate_k, db_path=db_path)
        if fts_weight > 0
        else []
    )

    vector_results: list[ChunkResult] = []
    if vector_weight > 0:
        try:
            query_embedding = embeddings_client.embed([question]).embeddings[0]
        except Exception as exc:  # pragma: no cover - defensive for API failures
            warnings.append(f"Embedding failed ({exc}); skipping vector search.")
            query_embedding = None

        if query_embedding is not None:
            try:
                vector_results = search_vector_chroma_with_embedding(
                    query_embedding,
                    top_k=candidate_k,
                    db_path=db_path,
                    chroma_config=chroma_config,
                    chroma_store=chroma_store,
                )
                if not vector_results and use_sqlite_fallback:
                    warnings.append(
                        "Chroma returned no results; falling back to SQLite embeddings."
                    )
                    vector_results = search_vector_with_embedding(
                        query_embedding,
                        top_k=candidate_k,
                        db_path=db_path,
                    )
            except ImportError as exc:
                if use_sqlite_fallback:
                    warnings.append(
                        f"Chroma unavailable ({exc}); falling back to SQLite embeddings."
                    )
                    vector_results = search_vector_with_embedding(
                        query_embedding,
                        top_k=candidate_k,
                        db_path=db_path,
                    )
                else:
                    warnings.append(
                        f"Chroma unavailable ({exc}); skipping vector search."
                    )

            if use_sqlite_fallback and vector_results == []:
                warnings.append(
                    "SQLite embedding search returned no results; check stored embeddings."
                )

    results = _fuse_rrf(
        fts_results,
        vector_results,
        top_k=top_k,
        rrf_k=rrf_k,
        fts_weight=fts_weight,
        vector_weight=vector_weight,
    )
    return HybridSearchOutput(results=results, warnings=warnings)


def search_vector_with_embedding(
    query_embedding: Sequence[float],
    *,
    top_k: int,
    db_path: Path,
) -> list[ChunkResult]:
    """Search chunks by cosine similarity using a provided embedding.

    Args:
        query_embedding: Pre-computed query embedding.
        top_k: Number of results to return.
        db_path: Path to the SQLite database.
    Returns:
        List of ChunkResult rows ordered by cosine similarity.
    Raises:
        FileNotFoundError: When the database path does not exist.
        ValueError: When top_k is not positive or embeddings are missing.
        sqlite3.Error: When SQLite operations fail.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    normalized_query = normalize_embedding(query_embedding)
    rows = _load_vector_rows(db_path)
    if not rows:
        return []

    query_array = array("f", normalized_query)
    scored: list[tuple[float, ChunkResult]] = []
    for row in rows:
        embedding = deserialize_embedding_array(row.embedding)
        if len(embedding) != len(query_array):
            continue
        score = _dot_product(query_array, embedding)
        scored.append(
            (
                score,
                ChunkResult(
                    chunk_uid=row.chunk_uid,
                    chunk_id=row.chunk_id,
                    paper_id=row.paper_id,
                    page_number=row.page_number,
                    text=row.text,
                    score=score,
                ),
            )
        )

    if not scored:
        return []

    top_results = heapq.nlargest(top_k, scored, key=lambda pair: pair[0])
    return [result for _, result in top_results]


@dataclass(frozen=True)
class _VectorRow:
    chunk_uid: str
    chunk_id: int
    paper_id: str
    page_number: int
    text: str
    embedding: bytes


def _load_vector_rows(db_path: Path) -> list[_VectorRow]:
    """Load chunk rows with embeddings from SQLite."""

    sql = """
        SELECT chunk_uid, chunk_id, paper_id, page_number, text, embedding
        FROM chunks
        WHERE embedding IS NOT NULL
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [
        _VectorRow(
            chunk_uid=row[0],
            chunk_id=row[1],
            paper_id=row[2],
            page_number=row[3],
            text=row[4],
            embedding=row[5],
        )
        for row in rows
    ]


def _dot_product(left: Iterable[float], right: Iterable[float]) -> float:
    """Compute dot product between two equal-length vectors."""

    return sum(left_value * right_value for left_value, right_value in zip(left, right))


@dataclass
class _RrfEntry:
    """Mutable accumulator for a fused RRF result."""

    chunk: ChunkResult
    rrf_score: float
    provenance: list[BackendProvenance]
    first_seen: int


def _fuse_rrf(
    fts_results: Sequence[ChunkResult],
    vector_results: Sequence[ChunkResult],
    *,
    top_k: int,
    rrf_k: int,
    fts_weight: float,
    vector_weight: float,
) -> list[HybridChunkResult]:
    """Fuse backend results using reciprocal rank fusion (RRF).

    Args:
        fts_results: Results from SQLite FTS search.
        vector_results: Results from vector search.
        top_k: Number of results to return.
        rrf_k: RRF constant.
        fts_weight: Weight applied to FTS contributions.
        vector_weight: Weight applied to vector contributions.
    Returns:
        Ranked hybrid results with provenance.
    Raises:
        ValueError: When parameters are invalid.
    """

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")
    if fts_weight < 0 or vector_weight < 0:
        raise ValueError("weights must be >= 0")
    if fts_weight == 0 and vector_weight == 0:
        raise ValueError("at least one weight must be > 0")

    merged: dict[str, _RrfEntry] = {}
    order_counter = 0

    def add_backend(
        results: Sequence[ChunkResult],
        *,
        backend: Literal["fts", "vector"],
        weight: float,
    ) -> None:
        nonlocal order_counter
        if weight == 0:
            return
        for index, result in enumerate(results):
            rank = index + 1
            normalized_score = 1.0 / rank
            rrf_contribution = weight / (rrf_k + rank)
            entry = merged.get(result.chunk_uid)
            if entry is None:
                entry = _RrfEntry(
                    chunk=result,
                    rrf_score=0.0,
                    provenance=[],
                    first_seen=order_counter,
                )
                merged[result.chunk_uid] = entry
                order_counter += 1

            entry.rrf_score += rrf_contribution
            entry.provenance.append(
                BackendProvenance(
                    backend=backend,
                    rank=rank,
                    raw_score=result.score,
                    normalized_score=normalized_score,
                    rrf_contribution=rrf_contribution,
                )
            )

    add_backend(fts_results, backend="fts", weight=fts_weight)
    add_backend(vector_results, backend="vector", weight=vector_weight)

    if not merged:
        return []

    ranked = sorted(
        merged.values(),
        key=lambda entry: (-entry.rrf_score, entry.first_seen),
    )
    output: list[HybridChunkResult] = []
    for entry in ranked[:top_k]:
        output.append(
            HybridChunkResult(
                chunk_uid=entry.chunk.chunk_uid,
                chunk_id=entry.chunk.chunk_id,
                paper_id=entry.chunk.paper_id,
                page_number=entry.chunk.page_number,
                text=entry.chunk.text,
                rrf_score=entry.rrf_score,
                provenance=entry.provenance,
            )
        )
    return output
