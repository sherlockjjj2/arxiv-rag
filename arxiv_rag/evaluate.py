"""Evaluation utilities for retrieval and citation accuracy."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import sqlite3
from hashlib import sha256
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from string import Template
from typing import Iterable, Literal, Mapping, Sequence

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from arxiv_rag.chroma_client import ChromaConfig, ChromaStore
from arxiv_rag.db import (
    load_chunk_uids_by_page,
    load_paper_ids,
    normalize_embedding,
)
from arxiv_rag.embeddings_client import (
    EmbeddingBatchResult,
    EmbeddingsClient,
    EmbeddingsConfig,
)
from arxiv_rag.generate import (
    Chunk as GenerationChunk,
    load_chunk_citation_prompt_template,
    generate_answer,
    load_quote_first_prompt_template,
    load_quote_selection_prompt_template,
    load_prompt_template,
    load_repair_prompt_template,
    load_selection_prompt_template,
)
from arxiv_rag.retrieve import (
    ChunkResult,
    HybridChunkResult,
    rerank_results_for_generation,
    search_fts,
    search_hybrid,
    search_vector_chroma,
)
from arxiv_rag.verify import CitationRecord, parse_citations

LOGGER = logging.getLogger(__name__)

_DEFAULT_EVAL_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "generate_eval_qa.txt"
)
_MAX_QA_ATTEMPTS_PER_CHUNK = 3
_DISALLOWED_QUESTION_PHRASES = (
    "in the excerpt",
    "from the excerpt",
    "this excerpt",
    "the excerpt",
    "as described in the excerpt",
    "as described in this excerpt",
    "as mentioned in the excerpt",
    "according to the excerpt",
    "in this passage",
    "from this passage",
    "the passage",
    "the text above",
    "the above text",
    "the paper referenced",
)
_OPENING_GROUPS: list[list[str]] = [
    ["Why", "How", "What", "When"],
    [
        "Compared to",
        "In comparison to",
        "How does",
        "What distinguishes",
        "In what way",
    ],
    ["Under what conditions", "In which case", "In what situation", "When would"],
    ["What limitation", "What trade-off", "What constraint", "What drawback"],
    [
        "What is the implication",
        "What does this suggest",
        "What does this imply",
        "What follows from",
    ],
    [
        "What explains",
        "What factors",
        "What mechanism",
        "What role does",
        "To what extent",
    ],
]
_OVERLAP_STOPWORDS = frozenset(
    {
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
        "has",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


def _round_robin_openings(groups: Sequence[list[str]]) -> list[str]:
    """Interleave opening groups in round-robin order."""

    output: list[str] = []
    remaining = True
    while remaining:
        remaining = False
        for group in groups:
            if not group:
                continue
            output.append(group.pop())
            remaining = True
    return output


def _build_openings_round(rng: random.Random) -> list[str]:
    """Build one balanced round of openings."""

    groups = [group.copy() for group in _OPENING_GROUPS]
    for group in groups:
        rng.shuffle(group)
    return _round_robin_openings(groups)


def build_openings_plan(n_questions: int, *, seed: int | None) -> list[str]:
    """Build a balanced, mostly-unique list of openings for eval questions.

    Args:
        n_questions: Number of questions to generate.
        seed: Optional RNG seed for stable ordering.
    Returns:
        List of opening strings, length n_questions.
    Raises:
        ValueError: If n_questions is not positive or exceeds supported capacity.
    """

    if n_questions <= 0:
        raise ValueError("n_questions must be > 0")

    rng = random.Random(0 if seed is None else seed)
    first_round = _build_openings_round(rng)
    if n_questions <= len(first_round):
        return first_round[:n_questions]

    second_round = _build_openings_round(rng)
    combined = first_round + second_round
    if n_questions > len(combined):
        raise ValueError(
            "n_questions exceeds available openings; provide --openings-path or lower n_questions."
        )
    return combined[:n_questions]


def load_openings_from_path(path: Path) -> list[str]:
    """Load openings from a JSON list or newline-delimited text file.

    Args:
        path: Path to the openings file.
    Returns:
        List of opening strings.
    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is empty or malformed.
    """

    if not path.exists():
        raise FileNotFoundError(f"Openings file not found: {path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Openings file is empty: {path}")

    openings: list[str]
    if raw_text.lstrip().startswith("["):
        payload = json.loads(raw_text)
        if not isinstance(payload, list):
            raise ValueError("Openings JSON must be a list of strings.")
        openings = [item.strip() for item in payload if isinstance(item, str)]
    else:
        openings = [line.strip() for line in raw_text.splitlines()]

    openings = [opening for opening in openings if opening]
    if not openings:
        raise ValueError("Openings list must contain at least one entry.")
    return openings


def format_openings_for_prompt(openings: Sequence[str]) -> str:
    """Format openings for prompt injection."""

    return "\n".join(
        f"{index + 1}. {opening}" for index, opening in enumerate(openings)
    )


Difficulty = Literal["factual", "synthesis"]


@dataclass(frozen=True)
class EvalGroundTruth:
    """Ground truth references for an eval query.

    Args:
        chunk_uids: Chunk UIDs that contain the answer.
        papers: arXiv paper IDs referenced by the answer.
        pages: Page numbers per paper in the same order as papers.
        expected_topics: Optional topic hints for manual review.
    """

    chunk_uids: list[str]
    papers: list[str]
    pages: list[list[int]]
    expected_topics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""

        return {
            "chunk_uids": list(self.chunk_uids),
            "papers": list(self.papers),
            "pages": [list(pages) for pages in self.pages],
            "expected_topics": list(self.expected_topics),
        }


@dataclass(frozen=True)
class EvalItem:
    """Evaluation item with a query and ground truth.

    Args:
        query_id: Stable query identifier.
        query: Query text.
        difficulty: Difficulty label.
        ground_truth: Ground truth chunk references.
        reference_answer: Reference answer text.
    """

    query_id: str
    query: str
    difficulty: Difficulty
    ground_truth: EvalGroundTruth
    reference_answer: str

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""

        return {
            "query_id": self.query_id,
            "query": self.query,
            "difficulty": self.difficulty,
            "ground_truth": self.ground_truth.as_dict(),
            "reference_answer": self.reference_answer,
        }


@dataclass(frozen=True)
class EvalMetadata:
    """Metadata describing an eval set.

    Args:
        created: ISO date string of eval set creation.
        corpus_version: Corpus version string.
        n_queries: Total number of queries in the eval set.
    """

    created: str
    corpus_version: str
    n_queries: int

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""

        return {
            "created": self.created,
            "corpus_version": self.corpus_version,
            "n_queries": self.n_queries,
        }


@dataclass(frozen=True)
class EvalSet:
    """Evaluation dataset container.

    Args:
        eval_set: List of eval items.
        metadata: Dataset metadata.
    """

    eval_set: list[EvalItem]
    metadata: EvalMetadata

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""

        return {
            "eval_set": [item.as_dict() for item in self.eval_set],
            "metadata": self.metadata.as_dict(),
        }


@dataclass(frozen=True)
class ChunkSample:
    """Chunk sample used for synthetic QA generation.

    Args:
        chunk_uid: Stable chunk identifier.
        paper_id: arXiv paper ID.
        page_number: 1-based page number.
        text: Chunk text.
        title: Optional paper title.
    """

    chunk_uid: str
    paper_id: str
    page_number: int
    text: str
    title: str | None


@dataclass(frozen=True)
class GeneratedQuestion:
    """Structured QA pair generated from a chunk.

    Args:
        question: Query string.
        expected_answer: Reference answer text.
        difficulty: Difficulty label.
    """

    question: str
    expected_answer: str
    difficulty: Difficulty


@dataclass(frozen=True)
class EvalGenerationConfig:
    """Configuration for eval QA generation requests.

    Args:
        model: OpenAI model name.
        temperature: Sampling temperature.
        top_p: Top-p nucleus sampling value.
        max_output_tokens: Maximum tokens for completion.
        request_timeout_s: Timeout for a single request.
        max_retries: Maximum retry attempts for transient errors.
        prompt_path: Optional override for the prompt template.
        cost_per_1k_tokens: Optional cost per 1k tokens for logging estimates.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.8
    top_p: float = 0.9
    max_output_tokens: int = 800
    request_timeout_s: float = 60.0
    max_retries: int = 5
    prompt_path: Path | None = None
    cost_per_1k_tokens: float | None = None


@dataclass(frozen=True)
class EvalGenerationResult:
    """Result of a QA generation call.

    Args:
        questions: Parsed QA pairs.
        raw_text: Raw model output.
    """

    questions: list[GeneratedQuestion]
    raw_text: str


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval configuration for evaluation.

    Args:
        mode: Retrieval backend name.
        top_k: Number of results to retrieve.
        model: Embedding model name.
        chroma_dir: Chroma persistence directory.
        collection: Chroma collection name.
        distance: Chroma distance metric.
        rrf_k: RRF constant for hybrid fusion.
        rrf_weight_fts: Weight for FTS in RRF.
        rrf_weight_vector: Weight for vector search in RRF.
    """

    mode: Literal["fts", "vector", "hybrid"] = "hybrid"
    top_k: int = 10
    model: str = "text-embedding-3-small"
    chroma_dir: Path = Path("data/chroma")
    collection: str = "arxiv_chunks_te3s_v1"
    distance: Literal["cosine", "l2", "ip"] = "cosine"
    rrf_k: int = 60
    rrf_weight_fts: float = 1.0
    rrf_weight_vector: float = 1.0


@dataclass(frozen=True)
class EvalItemResult:
    """Per-query evaluation result.

    Args:
        query_id: Query identifier.
        query: Query text.
        recall_at_5: Recall@5 score.
        recall_at_10: Recall@10 score.
        mrr: Reciprocal rank of the first correct result.
        first_correct_rank: Rank of first correct chunk, if any.
        retrieved_chunk_uids: Retrieved chunk identifiers.
        ground_truth_chunk_uids: Ground truth chunk identifiers.
        citation_count: Number of parsed citations in generated answer.
        citation_accuracy: Citation accuracy, if generated.
        citation_error: Citation parse or validation error.
        generation_error: Generation error, if any.
        ground_truth_in_generation_context: Whether a ground truth chunk was in the
            generation context.
        citations_in_generation_context: Number of citations that reference a
            retrieved generation chunk page.
        citations_outside_generation_context: Number of citations that reference a
            page outside the generation context.
        warnings: Retrieval warnings.
    """

    query_id: str
    query: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    first_correct_rank: int | None
    retrieved_chunk_uids: list[str]
    ground_truth_chunk_uids: list[str]
    citation_count: int | None = None
    citation_accuracy: float | None = None
    citation_error: str | None = None
    generation_error: str | None = None
    ground_truth_in_generation_context: bool | None = None
    citations_in_generation_context: int | None = None
    citations_outside_generation_context: int | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""

        return {
            "query_id": self.query_id,
            "query": self.query,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "first_correct_rank": self.first_correct_rank,
            "retrieved_chunk_uids": list(self.retrieved_chunk_uids),
            "ground_truth_chunk_uids": list(self.ground_truth_chunk_uids),
            "citation_count": self.citation_count,
            "citation_accuracy": self.citation_accuracy,
            "citation_error": self.citation_error,
            "generation_error": self.generation_error,
            "ground_truth_in_generation_context": (
                self.ground_truth_in_generation_context
            ),
            "citations_in_generation_context": (self.citations_in_generation_context),
            "citations_outside_generation_context": (
                self.citations_outside_generation_context
            ),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EvalSummary:
    """Aggregated evaluation metrics.

    Args:
        recall_at_5: Mean Recall@5.
        recall_at_10: Mean Recall@10.
        mrr: Mean MRR.
        citation_accuracy: Mean citation accuracy (if computed).
        citation_accuracy_when_recall5_hit: Mean citation accuracy for items where
            recall@5 > 0 (if computed).
        n_queries: Number of evaluated queries.
    """

    recall_at_5: float
    recall_at_10: float
    mrr: float
    citation_accuracy: float | None
    citation_accuracy_when_recall5_hit: float | None
    n_queries: int

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""

        return {
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "citation_accuracy": self.citation_accuracy,
            "citation_accuracy_when_recall5_hit": (
                self.citation_accuracy_when_recall5_hit
            ),
            "n_queries": self.n_queries,
        }


@dataclass(frozen=True)
class EvalFailureSummary:
    """Failure mode counts for evaluation.

    Args:
        retrieval_empty: Queries with no retrieval results.
        recall_at_5_zero: Queries with recall@5 == 0.
        mrr_zero: Queries with mrr == 0.
        citation_absent: Generated answers with no parsed citations.
        citation_zero_score: Generated answers with citations but score == 0.
        citation_parse_error: Answers with malformed citations.
        citation_inaccurate: Answers with citation accuracy < 1.
    """

    retrieval_empty: int
    recall_at_5_zero: int
    mrr_zero: int
    citation_absent: int
    citation_zero_score: int
    citation_parse_error: int
    citation_inaccurate: int

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""

        return {
            "retrieval_empty": self.retrieval_empty,
            "recall_at_5_zero": self.recall_at_5_zero,
            "mrr_zero": self.mrr_zero,
            "citation_absent": self.citation_absent,
            "citation_zero_score": self.citation_zero_score,
            "citation_parse_error": self.citation_parse_error,
            "citation_inaccurate": self.citation_inaccurate,
        }


@dataclass(frozen=True)
class EvalReport:
    """Full evaluation report.

    Args:
        created: ISO datetime string for report creation.
        summary: Aggregated metric summary.
        failures: Failure mode summary.
        items: Per-query results.
    """

    created: str
    summary: EvalSummary
    failures: EvalFailureSummary
    items: list[EvalItemResult]

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""

        return {
            "created": self.created,
            "summary": self.summary.as_dict(),
            "failures": self.failures.as_dict(),
            "items": [item.as_dict() for item in self.items],
        }


class EvalCache:
    """Persistent SQLite cache for eval embeddings and generated answers.

    Args:
        db_path: Path to the cache SQLite database.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize and ensure cache schema exists.

        Args:
            db_path: Cache database path.
        Raises:
            sqlite3.Error: If the cache DB cannot be opened or initialized.
        """

        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._ensure_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._conn.close()

    def __enter__(self) -> EvalCache:
        """Enter context manager scope."""

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close cache connection at context exit."""

        self.close()

    def _ensure_schema(self) -> None:
        """Create cache tables when missing."""

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS query_embedding_cache (
                query_text TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (query_text, embedding_model)
            );
            CREATE TABLE IF NOT EXISTS generated_answer_cache (
                query_text TEXT NOT NULL,
                generation_model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                chunk_uids_hash TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (
                    query_text,
                    generation_model,
                    prompt_version,
                    chunk_uids_hash
                )
            );
            """
        )
        self._conn.commit()

    def get_query_embedding(
        self,
        *,
        query: str,
        embedding_model: str,
    ) -> list[float] | None:
        """Load a cached query embedding.

        Args:
            query: Query text.
            embedding_model: Embedding model name.
        Returns:
            Cached embedding vector if present and valid, else None.
        """

        row = self._conn.execute(
            """
            SELECT embedding_json
            FROM query_embedding_cache
            WHERE query_text = ? AND embedding_model = ?
            """,
            (query, embedding_model),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, list) or not payload:
            return None
        return [float(value) for value in payload]

    def set_query_embedding(
        self,
        *,
        query: str,
        embedding_model: str,
        embedding: Sequence[float],
    ) -> None:
        """Store or update a cached query embedding.

        Args:
            query: Query text.
            embedding_model: Embedding model name.
            embedding: Embedding vector.
        """

        self._conn.execute(
            """
            INSERT INTO query_embedding_cache (
                query_text,
                embedding_model,
                embedding_json
            ) VALUES (?, ?, ?)
            ON CONFLICT(query_text, embedding_model)
            DO UPDATE SET
                embedding_json = excluded.embedding_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (query, embedding_model, json.dumps(list(embedding))),
        )
        self._conn.commit()

    def get_generated_answer(
        self,
        *,
        query: str,
        generation_model: str,
        prompt_version: str,
        chunk_uids_hash: str,
    ) -> str | None:
        """Load a cached generated answer.

        Args:
            query: Query text.
            generation_model: Generation model name.
            prompt_version: Prompt content version identifier.
            chunk_uids_hash: Hash of generation chunk UIDs.
        Returns:
            Cached answer text if present, else None.
        """

        row = self._conn.execute(
            """
            SELECT answer_text
            FROM generated_answer_cache
            WHERE query_text = ?
              AND generation_model = ?
              AND prompt_version = ?
              AND chunk_uids_hash = ?
            """,
            (query, generation_model, prompt_version, chunk_uids_hash),
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def set_generated_answer(
        self,
        *,
        query: str,
        generation_model: str,
        prompt_version: str,
        chunk_uids_hash: str,
        answer: str,
    ) -> None:
        """Store or update a cached generated answer."""

        self._conn.execute(
            """
            INSERT INTO generated_answer_cache (
                query_text,
                generation_model,
                prompt_version,
                chunk_uids_hash,
                answer_text
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(
                query_text,
                generation_model,
                prompt_version,
                chunk_uids_hash
            )
            DO UPDATE SET
                answer_text = excluded.answer_text,
                created_at = CURRENT_TIMESTAMP
            """,
            (query, generation_model, prompt_version, chunk_uids_hash, answer),
        )
        self._conn.commit()


class CachedEmbeddingsClient:
    """Embeddings client wrapper backed by EvalCache."""

    def __init__(
        self,
        *,
        base_client: EmbeddingsClient,
        cache: EvalCache | None,
    ) -> None:
        """Initialize caching wrapper.

        Args:
            base_client: Non-caching embeddings client.
            cache: Optional cache backend. No-op when None.
        """

        self._base_client = base_client
        self._cache = cache

    @property
    def config(self) -> EmbeddingsConfig:
        """Return active embeddings config."""

        return self._base_client.config

    def embed(self, inputs: Sequence[str]) -> EmbeddingBatchResult:
        """Embed text inputs with cache lookup for repeated requests.

        Args:
            inputs: Input texts to embed.
        Returns:
            Embeddings in the same order as inputs.
        Raises:
            ValueError: If inputs is empty.
        """

        if self._cache is None:
            return self._base_client.embed(inputs)

        if not inputs:
            raise ValueError("inputs must be non-empty")

        model = self._base_client.config.model
        embeddings_by_input: dict[str, list[float]] = {}
        missing_inputs: list[str] = []
        for text in dict.fromkeys(inputs):
            cached_embedding = self._cache.get_query_embedding(
                query=text,
                embedding_model=model,
            )
            if cached_embedding is None:
                missing_inputs.append(text)
            else:
                embeddings_by_input[text] = cached_embedding

        total_tokens: int | None = None
        if missing_inputs:
            batch_result = self._base_client.embed(missing_inputs)
            total_tokens = batch_result.total_tokens
            for text, embedding in zip(
                missing_inputs,
                batch_result.embeddings,
                strict=False,
            ):
                embeddings_by_input[text] = embedding
                self._cache.set_query_embedding(
                    query=text,
                    embedding_model=model,
                    embedding=embedding,
                )

        return EmbeddingBatchResult(
            embeddings=[embeddings_by_input[text] for text in inputs],
            total_tokens=total_tokens,
        )


@dataclass
class _EvalComputation:
    """Intermediate per-query evaluation state before final report assembly."""

    item: EvalItem
    retrieval_results: list[ChunkResult | HybridChunkResult]
    warnings: list[str]
    retrieved_uids: list[str]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    first_correct_rank: int | None
    citation_count: int | None = None
    citation_accuracy: float | None = None
    citation_error: str | None = None
    generation_error: str | None = None
    ground_truth_in_generation_context: bool | None = None
    citations_in_generation_context: int | None = None
    citations_outside_generation_context: int | None = None


_GenerationCacheKey = tuple[str, str, str, str]


class EvalGenerationClient:
    """Thin wrapper around OpenAI chat completions for eval QA generation."""

    def __init__(
        self,
        config: EvalGenerationConfig,
        client: OpenAI | None = None,
    ) -> None:
        """Initialize the generation client.

        Args:
            config: Generation configuration.
            client: Optional OpenAI client override for testing.
        """

        self._config = config
        self._client = client or OpenAI()

    def generate_questions(
        self, *, prompt: str, openings: Sequence[str] | None = None
    ) -> EvalGenerationResult:
        """Generate QA pairs from a prompt.

        Args:
            prompt: Prompt containing chunk context.
            openings: Optional openings list to validate prefixes.
        Returns:
            EvalGenerationResult with parsed questions.
        Raises:
            ValueError: If prompt is empty or parsing fails.
            APIError: For non-retryable OpenAI API errors.
        """

        if not prompt.strip():
            raise ValueError("prompt must be non-empty")

        response = self._create_completion(prompt=prompt)
        request_id = getattr(response, "id", None)
        if request_id:
            LOGGER.debug("Eval QA request_id=%s", request_id)

        usage = getattr(response, "usage", None)
        if usage is not None:
            LOGGER.debug(
                "Eval QA usage: prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
            if self._config.cost_per_1k_tokens is not None:
                cost = (usage.total_tokens / 1000.0) * self._config.cost_per_1k_tokens
                LOGGER.debug("Eval QA estimated cost: $%.6f", cost)

        if not response.choices:
            raise ValueError("Eval QA response contained no choices.")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Eval QA response contained empty content.")

        questions = parse_generated_questions(content, openings=openings)
        return EvalGenerationResult(questions=questions, raw_text=content)

    def _create_completion(self, *, prompt: str):
        """Call the OpenAI chat completions API with retry logic."""

        retrying = Retrying(
            retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError)),
            stop=stop_after_attempt(self._config.max_retries),
            wait=wait_exponential_jitter(initial=1, max=20),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Generate the questions."},
                    ],
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    max_tokens=self._config.max_output_tokens,
                    timeout=self._config.request_timeout_s,
                )
        raise RuntimeError("Unexpected retry exhaustion for eval QA request.")


def load_eval_prompt_template(path: Path | None = None) -> str:
    """Load the eval QA prompt template from disk.

    Args:
        path: Optional override path for the prompt template.
    Returns:
        Prompt template text.
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt template is empty.
    """

    prompt_path = path or _DEFAULT_EVAL_PROMPT_PATH
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Prompt template is empty: {prompt_path}")
    return template


def render_eval_prompt(
    template: str,
    *,
    chunk: ChunkSample,
    n_questions: int,
    openings: Sequence[str] | None = None,
) -> str:
    """Render the eval QA prompt template.

    Args:
        template: Prompt template text.
        chunk: Chunk sample with metadata and text.
        n_questions: Number of questions to request.
        openings: Optional openings list to enforce question prefixes.
    Returns:
        Rendered prompt string.
    Raises:
        ValueError: If the template is missing required placeholders.
    """

    required_placeholders = [
        "CHUNK_TEXT",
        "PAPER_ID",
        "PAGE_NUMBER",
        "CHUNK_UID",
        "N_QUESTIONS",
    ]
    for placeholder in required_placeholders:
        if f"${placeholder}" not in template:
            raise ValueError(f"Prompt template missing ${placeholder} placeholder")
    if openings is None:
        if "$OPENINGS" in template:
            raise ValueError(
                "Prompt template requires $OPENINGS but none were provided."
            )
    else:
        if "$OPENINGS" not in template:
            raise ValueError(
                "Prompt template missing $OPENINGS placeholder for provided openings."
            )
        if len(openings) != n_questions:
            raise ValueError("Openings length must match n_questions.")

    placeholders = {
        "CHUNK_TEXT": chunk.text.replace("$", "$$"),
        "PAPER_ID": chunk.paper_id.replace("$", "$$"),
        "PAGE_NUMBER": str(chunk.page_number),
        "CHUNK_UID": chunk.chunk_uid.replace("$", "$$"),
        "TITLE": (chunk.title or "").replace("$", "$$"),
        "N_QUESTIONS": str(n_questions),
    }
    if openings is not None:
        placeholders["OPENINGS"] = format_openings_for_prompt(openings).replace(
            "$", "$$"
        )
    rendered = Template(template).safe_substitute(placeholders)
    if "$TITLE" in template and not placeholders["TITLE"]:
        LOGGER.debug("Eval QA prompt rendered without title.")
    return rendered


def _validate_question_openings(
    questions: Sequence[GeneratedQuestion], openings: Sequence[str]
) -> None:
    """Ensure each generated question starts with the expected opening."""

    if len(questions) != len(openings):
        raise ValueError(
            "Generated question count does not match the required openings list."
        )
    for index, (question, opening) in enumerate(zip(questions, openings), start=1):
        normalized = question.question.lstrip()
        if not normalized.startswith(opening):
            raise ValueError(
                f"Question {index} must start with {opening!r}, got {question.question!r}."
            )


def parse_generated_questions(
    raw_text: str, *, openings: Sequence[str] | None = None
) -> list[GeneratedQuestion]:
    """Parse generated QA pairs from model output.

    Args:
        raw_text: Raw model output.
        openings: Optional openings list to validate prefixes.
    Returns:
        List of GeneratedQuestion entries.
    Raises:
        ValueError: If no valid questions can be parsed.
    """

    payload = _parse_json_payload(raw_text)
    if payload is None:
        raise ValueError("Failed to parse JSON from eval QA output.")

    items = payload
    if isinstance(payload, dict):
        if "items" in payload:
            items = payload.get("items")
        elif "questions" in payload:
            items = payload.get("questions")

    if not isinstance(items, list):
        raise ValueError("Eval QA output is not a JSON array.")

    questions: list[GeneratedQuestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        expected_answer = str(item.get("expected_answer", "")).strip()
        difficulty_raw = str(item.get("difficulty", "")).strip().lower()
        if not question or not expected_answer:
            continue
        if difficulty_raw not in {"factual", "synthesis"}:
            continue
        questions.append(
            GeneratedQuestion(
                question=question,
                expected_answer=expected_answer,
                difficulty=difficulty_raw,  # type: ignore[arg-type]
            )
        )

    if not questions:
        raise ValueError("Eval QA output contained no valid questions.")
    if openings is not None:
        _validate_question_openings(questions, openings)
    return questions


def _parse_json_payload(raw_text: str) -> object | None:
    """Parse JSON payload from raw model output.

    Args:
        raw_text: Raw model output.
    Returns:
        Parsed JSON object, or None if parsing fails.
    """

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    array_text = _extract_json_segment(raw_text, "[", "]")
    if array_text is not None:
        try:
            return json.loads(array_text)
        except json.JSONDecodeError:
            return None

    object_text = _extract_json_segment(raw_text, "{", "}")
    if object_text is not None:
        try:
            return json.loads(object_text)
        except json.JSONDecodeError:
            return None

    return None


def _normalize_question_key(question: str) -> str:
    """Normalize a question for de-duplication."""

    collapsed = re.sub(r"\s+", " ", question.strip().lower())
    return re.sub(r"[^\w\s]", "", collapsed)


def _has_banned_context_reference(question: str) -> bool:
    """Check whether a question references external context deictically.

    Args:
        question: Generated question text.
    Returns:
        True when the question contains phrases like "in the excerpt".
    """

    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return any(phrase in normalized for phrase in _DISALLOWED_QUESTION_PHRASES)


def _content_tokens(text: str) -> set[str]:
    """Extract content-bearing tokens for lexical grounding checks.

    Args:
        text: Raw text to tokenize.
    Returns:
        Lower-cased token set excluding short/common stop words.
    """

    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {
        token for token in tokens if len(token) >= 3 and token not in _OVERLAP_STOPWORDS
    }


def _has_minimum_grounding_overlap(expected_answer: str, chunk_text: str) -> bool:
    """Check whether an expected answer is lexically grounded in a chunk.

    Args:
        expected_answer: Generated reference answer.
        chunk_text: Source chunk text used for QA generation.
    Returns:
        True when the answer shares enough content tokens with the chunk.
    """

    answer_tokens = _content_tokens(expected_answer)
    if not answer_tokens:
        return False

    overlap = len(answer_tokens & _content_tokens(chunk_text))
    if len(answer_tokens) <= 4:
        return overlap >= 1
    return overlap >= 2 and (overlap / len(answer_tokens)) >= 0.25


def _validation_error_for_generated_question(
    question: GeneratedQuestion,
    *,
    chunk_text: str,
) -> str | None:
    """Validate generated QA quality and return a rejection reason.

    Args:
        question: Generated QA item.
        chunk_text: Source chunk text.
    Returns:
        None if valid, otherwise a short rejection reason.
    """

    if _has_banned_context_reference(question.question):
        return "question is not standalone"
    if not _has_minimum_grounding_overlap(question.expected_answer, chunk_text):
        return "expected answer is weakly grounded in chunk text"
    return None


def _validation_error_for_generated_batch(
    questions: Sequence[GeneratedQuestion],
    *,
    chunk_text: str,
    seen_questions: set[str],
) -> str | None:
    """Validate a batch of generated questions.

    Args:
        questions: Generated questions for one chunk.
        chunk_text: Source chunk text.
        seen_questions: Normalized question keys already accepted.
    Returns:
        None if valid, otherwise a short rejection reason.
    """

    seen_batch: set[str] = set()
    for question in questions:
        question_key = _normalize_question_key(question.question)
        if question_key in seen_questions:
            return "duplicate question in eval set"
        if question_key in seen_batch:
            return "duplicate question in batch"
        rejection_reason = _validation_error_for_generated_question(
            question,
            chunk_text=chunk_text,
        )
        if rejection_reason is not None:
            return rejection_reason
        seen_batch.add(question_key)
    return None


def _extract_json_segment(
    raw_text: str, start_token: str, end_token: str
) -> str | None:
    """Extract a JSON segment bounded by start/end tokens.

    Args:
        raw_text: Raw model output.
        start_token: Opening token.
        end_token: Closing token.
    Returns:
        Substring containing the JSON segment, or None if missing.
    """

    start_index = raw_text.find(start_token)
    end_index = raw_text.rfind(end_token)
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return None
    return raw_text[start_index : end_index + 1]


def load_eval_set(path: Path) -> EvalSet:
    """Load an eval set from disk.

    Args:
        path: Path to the eval set JSON.
    Returns:
        EvalSet instance.
    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the JSON structure is invalid.
    """

    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    eval_items = payload.get("eval_set")
    metadata = payload.get("metadata")
    if not isinstance(eval_items, list) or not isinstance(metadata, dict):
        raise ValueError("Eval set JSON missing required fields.")

    items: list[EvalItem] = []
    for item in eval_items:
        ground_truth = item.get("ground_truth", {}) if isinstance(item, dict) else {}
        items.append(
            EvalItem(
                query_id=str(item.get("query_id", "")),
                query=str(item.get("query", "")),
                difficulty=str(item.get("difficulty", "factual")),  # type: ignore[arg-type]
                ground_truth=EvalGroundTruth(
                    chunk_uids=list(ground_truth.get("chunk_uids", [])),
                    papers=list(ground_truth.get("papers", [])),
                    pages=[list(pages) for pages in ground_truth.get("pages", [])],
                    expected_topics=list(ground_truth.get("expected_topics", [])),
                ),
                reference_answer=str(item.get("reference_answer", "")),
            )
        )

    metadata_obj = EvalMetadata(
        created=str(metadata.get("created", "")),
        corpus_version=str(metadata.get("corpus_version", "")),
        n_queries=int(metadata.get("n_queries", len(items))),
    )
    return EvalSet(eval_set=items, metadata=metadata_obj)


def save_eval_set(eval_set: EvalSet, path: Path) -> None:
    """Save an eval set to disk.

    Args:
        eval_set: EvalSet instance to serialize.
        path: Destination path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(eval_set.as_dict(), indent=2), encoding="utf-8")


def generate_eval_set(
    *,
    db_path: Path,
    output_path: Path,
    n_questions: int,
    questions_per_chunk: int,
    seed: int | None,
    min_chars: int,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    request_timeout_s: float,
    max_retries: int,
    prompt_path: Path | None,
    corpus_version: str,
    openings_plan: Sequence[str] | None,
) -> EvalSet:
    """Generate a synthetic eval set from stored chunks.

    Args:
        db_path: SQLite database path.
        output_path: Destination path for the eval set JSON.
        n_questions: Total number of questions to generate.
        questions_per_chunk: Questions to request per chunk.
        seed: Optional RNG seed for chunk sampling.
        min_chars: Minimum chunk text length.
        model: OpenAI model name.
        temperature: Sampling temperature.
        top_p: Top-p nucleus sampling value.
        max_output_tokens: Completion token limit.
        request_timeout_s: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        prompt_path: Optional prompt template override.
        corpus_version: Corpus version label.
        openings_plan: Optional list of question openings to enforce.
    Returns:
        Generated EvalSet instance.
    Raises:
        ValueError: If inputs are invalid or generation is incomplete.
        FileNotFoundError: If the database path does not exist.
        sqlite3.Error: If database queries fail.
    """

    if n_questions <= 0:
        raise ValueError("n_questions must be > 0")
    if questions_per_chunk <= 0:
        raise ValueError("questions_per_chunk must be > 0")
    if min_chars <= 0:
        raise ValueError("min_chars must be > 0")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    openings_list: list[str] | None = None
    if openings_plan is not None:
        openings_list = [
            opening.strip() for opening in openings_plan if opening.strip()
        ]
        if len(openings_list) < n_questions:
            raise ValueError("openings_plan must provide at least n_questions entries.")

    template = load_eval_prompt_template(prompt_path)
    config = EvalGenerationConfig(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        request_timeout_s=request_timeout_s,
        max_retries=max_retries,
        prompt_path=prompt_path,
    )
    client = EvalGenerationClient(config)

    candidate_uids = _load_candidate_chunk_uids(db_path, min_chars=min_chars)
    if not candidate_uids:
        raise ValueError("No chunks available for eval generation.")
    if n_questions > len(candidate_uids) * questions_per_chunk:
        raise ValueError("Not enough chunks to satisfy requested questions per chunk.")

    rng = random.Random(0 if seed is None else seed)
    rng.shuffle(candidate_uids)

    eval_items: list[EvalItem] = []
    seen_questions: set[str] = set()
    query_index = 1
    opening_index = 0

    with sqlite3.connect(db_path) as conn:
        for chunk_uid in candidate_uids:
            if len(eval_items) >= n_questions:
                break
            chunk = _load_chunk_sample(conn, chunk_uid)
            if chunk is None:
                continue

            remaining = n_questions - len(eval_items)
            chunk_question_count = min(questions_per_chunk, remaining)
            chunk_openings: list[str] | None = None
            if openings_list is not None:
                chunk_openings = openings_list[
                    opening_index : opening_index + chunk_question_count
                ]
                if len(chunk_openings) < chunk_question_count:
                    raise ValueError(
                        "Not enough openings to cover remaining questions."
                    )

            prompt = render_eval_prompt(
                template,
                chunk=chunk,
                n_questions=chunk_question_count,
                openings=chunk_openings,
            )
            LOGGER.debug(
                "Eval QA prompt chunk_uid=%s text_chars=%s",
                chunk.chunk_uid,
                len(chunk.text),
            )
            accepted_questions: list[GeneratedQuestion] = []
            seen_chunk_questions: set[str] = set()
            for attempt_index in range(_MAX_QA_ATTEMPTS_PER_CHUNK):
                if len(accepted_questions) >= chunk_question_count:
                    break
                try:
                    result = client.generate_questions(
                        prompt=prompt,
                        openings=chunk_openings,
                    )
                except (ValueError, APIError, APITimeoutError, RateLimitError) as exc:
                    LOGGER.warning(
                        "Eval QA generation failed for %s (attempt %s/%s): %s",
                        chunk.chunk_uid,
                        attempt_index + 1,
                        _MAX_QA_ATTEMPTS_PER_CHUNK,
                        exc,
                    )
                    continue

                if chunk_openings is not None:
                    rejection_reason = _validation_error_for_generated_batch(
                        result.questions,
                        chunk_text=chunk.text,
                        seen_questions=seen_questions,
                    )
                    if rejection_reason is not None:
                        LOGGER.debug(
                            "Rejected eval QA batch chunk_uid=%s reason=%s",
                            chunk.chunk_uid,
                            rejection_reason,
                        )
                        continue
                    accepted_questions = list(result.questions)
                    break

                for question in result.questions:
                    if len(accepted_questions) >= chunk_question_count:
                        break
                    question_key = _normalize_question_key(question.question)
                    if question_key in seen_chunk_questions:
                        continue
                    if question_key in seen_questions:
                        continue

                    rejection_reason = _validation_error_for_generated_question(
                        question,
                        chunk_text=chunk.text,
                    )
                    if rejection_reason is not None:
                        LOGGER.debug(
                            "Rejected eval QA chunk_uid=%s reason=%s question=%r",
                            chunk.chunk_uid,
                            rejection_reason,
                            question.question,
                        )
                        continue

                    seen_chunk_questions.add(question_key)
                    seen_questions.add(question_key)
                    accepted_questions.append(question)

            for question in accepted_questions:
                if len(eval_items) >= n_questions:
                    break
                question_key = _normalize_question_key(question.question)
                seen_questions.add(question_key)
                query_id = f"q{query_index:03d}"
                query_index += 1
                eval_items.append(
                    EvalItem(
                        query_id=query_id,
                        query=question.question,
                        difficulty=question.difficulty,
                        ground_truth=EvalGroundTruth(
                            chunk_uids=[chunk.chunk_uid],
                            papers=[chunk.paper_id],
                            pages=[[chunk.page_number]],
                            expected_topics=[],
                        ),
                        reference_answer=question.expected_answer,
                    )
                )
            if chunk_openings is not None:
                opening_index += len(accepted_questions)

    if len(eval_items) < n_questions:
        raise ValueError(
            f"Generated {len(eval_items)} questions; expected {n_questions}."
        )

    metadata = EvalMetadata(
        created=date.today().isoformat(),
        corpus_version=corpus_version,
        n_queries=len(eval_items),
    )
    eval_set = EvalSet(eval_set=eval_items, metadata=metadata)
    save_eval_set(eval_set, output_path)
    return eval_set


def _load_candidate_chunk_uids(db_path: Path, *, min_chars: int) -> list[str]:
    """Load candidate chunk UIDs for eval generation.

    Args:
        db_path: SQLite database path.
        min_chars: Minimum chunk text length.
    Returns:
        List of chunk UIDs.
    """

    sql = """
        SELECT chunk_uid
        FROM chunks
        WHERE LENGTH(text) >= ?
        ORDER BY chunk_uid
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, (min_chars,)).fetchall()
    return [row[0] for row in rows]


def _load_chunk_sample(conn: sqlite3.Connection, chunk_uid: str) -> ChunkSample | None:
    """Load a chunk sample by UID.

    Args:
        conn: SQLite connection.
        chunk_uid: Chunk UID to load.
    Returns:
        ChunkSample or None if missing.
    """

    sql = """
        SELECT chunks.chunk_uid, chunks.paper_id, chunks.page_number, chunks.text, papers.title
        FROM chunks
        LEFT JOIN papers ON chunks.paper_id = papers.paper_id
        WHERE chunks.chunk_uid = ?
    """
    row = conn.execute(sql, (chunk_uid,)).fetchone()
    if row is None:
        return None
    return ChunkSample(
        chunk_uid=row[0],
        paper_id=row[1],
        page_number=row[2],
        text=row[3],
        title=row[4],
    )


def compute_recall_at_k(
    retrieved: Sequence[str],
    ground_truth: Iterable[str],
    *,
    k: int,
) -> float:
    """Compute Recall@K for retrieved chunk UIDs.

    Args:
        retrieved: Retrieved chunk UIDs in rank order.
        ground_truth: Ground truth chunk UIDs.
        k: Rank cutoff.
    Returns:
        Recall@K score.
    Raises:
        ValueError: If k is not positive.
    """

    if k <= 0:
        raise ValueError("k must be > 0")

    ground_truth_set = {uid for uid in ground_truth if uid}
    if not ground_truth_set:
        return 0.0

    retrieved_set = set(retrieved[:k])
    return len(retrieved_set & ground_truth_set) / len(ground_truth_set)


def compute_mrr(retrieved: Sequence[str], ground_truth: Iterable[str]) -> float:
    """Compute reciprocal rank for retrieved chunk UIDs.

    Args:
        retrieved: Retrieved chunk UIDs in rank order.
        ground_truth: Ground truth chunk UIDs.
    Returns:
        Reciprocal rank of the first correct result.
    """

    ground_truth_set = {uid for uid in ground_truth if uid}
    if not ground_truth_set:
        return 0.0

    for rank, uid in enumerate(retrieved, start=1):
        if uid in ground_truth_set:
            return 1.0 / rank
    return 0.0


def find_first_correct_rank(
    retrieved: Sequence[str],
    ground_truth: Iterable[str],
) -> int | None:
    """Find the first rank containing a ground truth chunk.

    Args:
        retrieved: Retrieved chunk UIDs in rank order.
        ground_truth: Ground truth chunk UIDs.
    Returns:
        Rank of the first correct result, or None if none found.
    """

    ground_truth_set = {uid for uid in ground_truth if uid}
    if not ground_truth_set:
        return None

    for rank, uid in enumerate(retrieved, start=1):
        if uid in ground_truth_set:
            return rank
    return None


def compute_citation_accuracy(
    citations: Sequence[CitationRecord],
    *,
    ground_truth_chunk_uids: Iterable[str],
    chunk_uids_by_page: Mapping[tuple[str, int], Sequence[str]],
) -> float:
    """Compute citation accuracy at the chunk level.

    A citation is accurate if any chunk on the cited page matches ground truth.

    Args:
        citations: Parsed citation records.
        ground_truth_chunk_uids: Ground truth chunk UIDs.
        chunk_uids_by_page: Mapping from (paper_id, page_number) to chunk UIDs.
    Returns:
        Citation accuracy in [0, 1].
    """

    if not citations:
        return 0.0

    ground_truth_set = {uid for uid in ground_truth_chunk_uids if uid}
    if not ground_truth_set:
        return 0.0

    correct = 0
    for citation in citations:
        page_key = (citation.paper_id, citation.page_number)
        page_chunk_uids = set(chunk_uids_by_page.get(page_key, []))
        if page_chunk_uids & ground_truth_set:
            correct += 1
    return correct / len(citations)


def run_eval(
    *,
    eval_set: EvalSet,
    db_path: Path,
    retrieval_config: RetrievalConfig,
    generate: bool,
    generate_model: str,
    generation_top_k: int,
    generation_rerank: Literal["none", "lexical"] = "none",
    generation_select_evidence: bool = False,
    generation_select_k: int = 3,
    generation_quote_first: bool = False,
    generation_cite_chunk_index: bool = False,
    generation_repair_citations: bool = False,
    generation_repair_max_attempts: int = 1,
    generation_concurrency: int = 4,
    cache_db_path: Path | None = None,
) -> EvalReport:
    """Run evaluation for an eval set.

    Args:
        eval_set: EvalSet to evaluate.
        db_path: SQLite database path.
        retrieval_config: Retrieval settings.
        generate: Whether to run generation and compute citation accuracy.
        generate_model: Model name for answer generation.
        generation_top_k: Number of chunks to pass to the generator.
        generation_rerank: Rerank strategy for generation chunks.
        generation_select_evidence: Whether to select evidence chunks before answering.
        generation_select_k: Max chunks to keep when selection is enabled.
        generation_quote_first: Whether to force quote-first generation.
        generation_cite_chunk_index: Whether to force chunk-index citations.
        generation_repair_citations: Whether to repair missing/invalid citations.
        generation_repair_max_attempts: Max attempts for citation repair.
        generation_concurrency: Maximum concurrent generation requests.
        cache_db_path: Optional SQLite cache path for embeddings/answers.
    Returns:
        EvalReport with per-item results and summary metrics.
    Raises:
        FileNotFoundError: If the database path does not exist.
        ValueError: If generation_top_k is invalid.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if generation_top_k <= 0:
        raise ValueError("generation_top_k must be > 0")
    if generation_concurrency <= 0:
        raise ValueError("generation_concurrency must be > 0")
    if generation_select_k <= 0:
        raise ValueError("generation_select_k must be > 0")
    if generation_repair_max_attempts <= 0:
        raise ValueError("generation_repair_max_attempts must be > 0")
    if generation_cite_chunk_index and generation_quote_first:
        raise ValueError(
            "generation_cite_chunk_index and generation_quote_first cannot both be True."
        )

    cache = EvalCache(cache_db_path) if cache_db_path is not None else None
    prompt_version = (
        _compute_prompt_version(
            generation_select_evidence=generation_select_evidence,
            generation_select_k=generation_select_k,
            generation_quote_first=generation_quote_first,
            generation_cite_chunk_index=generation_cite_chunk_index,
            generation_repair_citations=generation_repair_citations,
            generation_repair_max_attempts=generation_repair_max_attempts,
        )
        if generate and cache is not None
        else None
    )
    embeddings_client: CachedEmbeddingsClient | None = None
    chroma_config: ChromaConfig | None = None
    if retrieval_config.mode in {"vector", "hybrid"}:
        embeddings_client = CachedEmbeddingsClient(
            base_client=EmbeddingsClient(
                EmbeddingsConfig(model=retrieval_config.model)
            ),
            cache=cache,
        )
        chroma_config = ChromaConfig(
            persist_dir=retrieval_config.chroma_dir,
            collection_name=retrieval_config.collection,
            distance=retrieval_config.distance,
        )

    computations: list[_EvalComputation] = []

    try:
        for item in eval_set.eval_set:
            retrieval_results, warnings = _run_retrieval(
                query=item.query,
                db_path=db_path,
                retrieval_config=retrieval_config,
                embeddings_client=embeddings_client,
                chroma_config=chroma_config,
            )
            retrieved_uids = [result.chunk_uid for result in retrieval_results]

            recall_5 = compute_recall_at_k(
                retrieved_uids,
                item.ground_truth.chunk_uids,
                k=5,
            )
            recall_10 = compute_recall_at_k(
                retrieved_uids,
                item.ground_truth.chunk_uids,
                k=10,
            )
            mrr = compute_mrr(retrieved_uids, item.ground_truth.chunk_uids)
            first_rank = find_first_correct_rank(
                retrieved_uids,
                item.ground_truth.chunk_uids,
            )

            computations.append(
                _EvalComputation(
                    item=item,
                    retrieval_results=retrieval_results,
                    warnings=warnings,
                    retrieved_uids=retrieved_uids,
                    recall_at_5=recall_5,
                    recall_at_10=recall_10,
                    mrr=mrr,
                    first_correct_rank=first_rank,
                )
            )

        if generate and computations:
            _run_generation_stage(
                computations=computations,
                db_path=db_path,
                generate_model=generate_model,
                generation_top_k=generation_top_k,
                generation_rerank=generation_rerank,
                generation_select_evidence=generation_select_evidence,
                generation_select_k=generation_select_k,
                generation_quote_first=generation_quote_first,
                generation_cite_chunk_index=generation_cite_chunk_index,
                generation_repair_citations=generation_repair_citations,
                generation_repair_max_attempts=generation_repair_max_attempts,
                generation_concurrency=generation_concurrency,
                cache=cache,
                prompt_version=prompt_version,
            )
    finally:
        if cache is not None:
            cache.close()

    results = [
        EvalItemResult(
            query_id=computation.item.query_id,
            query=computation.item.query,
            recall_at_5=computation.recall_at_5,
            recall_at_10=computation.recall_at_10,
            mrr=computation.mrr,
            first_correct_rank=computation.first_correct_rank,
            retrieved_chunk_uids=computation.retrieved_uids,
            ground_truth_chunk_uids=list(computation.item.ground_truth.chunk_uids),
            citation_count=computation.citation_count,
            citation_accuracy=computation.citation_accuracy,
            citation_error=computation.citation_error,
            generation_error=computation.generation_error,
            ground_truth_in_generation_context=(
                computation.ground_truth_in_generation_context
            ),
            citations_in_generation_context=(
                computation.citations_in_generation_context
            ),
            citations_outside_generation_context=(
                computation.citations_outside_generation_context
            ),
            warnings=computation.warnings,
        )
        for computation in computations
    ]

    recall_5_scores = [computation.recall_at_5 for computation in computations]
    recall_10_scores = [computation.recall_at_10 for computation in computations]
    mrr_scores = [computation.mrr for computation in computations]
    citation_scores: list[float] = []
    citation_scores_recall5_hit: list[float] = []
    for computation in computations:
        if computation.citation_accuracy is not None:
            citation_scores.append(computation.citation_accuracy)
            if computation.recall_at_5 > 0.0:
                citation_scores_recall5_hit.append(computation.citation_accuracy)

    summary = EvalSummary(
        recall_at_5=_mean(recall_5_scores),
        recall_at_10=_mean(recall_10_scores),
        mrr=_mean(mrr_scores),
        citation_accuracy=_mean(citation_scores) if citation_scores else None,
        citation_accuracy_when_recall5_hit=(
            _mean(citation_scores_recall5_hit) if citation_scores_recall5_hit else None
        ),
        n_queries=len(results),
    )
    failures = _summarize_failures(results)

    return EvalReport(
        created=datetime.now().isoformat(timespec="seconds"),
        summary=summary,
        failures=failures,
        items=results,
    )


def _run_generation_stage(
    *,
    computations: Sequence[_EvalComputation],
    db_path: Path,
    generate_model: str,
    generation_top_k: int,
    generation_rerank: Literal["none", "lexical"],
    generation_select_evidence: bool,
    generation_select_k: int,
    generation_quote_first: bool,
    generation_cite_chunk_index: bool,
    generation_repair_citations: bool,
    generation_repair_max_attempts: int,
    generation_concurrency: int,
    cache: EvalCache | None,
    prompt_version: str | None,
) -> None:
    """Run generation for eval items concurrently and update computations in place.

    Args:
        computations: Per-item retrieval/metric computations.
        db_path: SQLite database path for citation scoring.
        generate_model: Generation model name.
        generation_top_k: Number of chunks to pass to generation.
        generation_rerank: Rerank strategy for generation chunks.
        generation_select_evidence: Whether to select evidence chunks before answering.
        generation_select_k: Max chunks to keep when selection is enabled.
        generation_quote_first: Whether to force quote-first generation.
        generation_cite_chunk_index: Whether to force chunk-index citations.
        generation_repair_citations: Whether to repair missing/invalid citations.
        generation_repair_max_attempts: Max attempts for citation repair.
        generation_concurrency: Maximum number of in-flight generation requests.
        cache: Optional eval cache.
        prompt_version: Prompt hash for generated answer cache keys.
    Raises:
        ValueError: If generation_concurrency is not positive.
    """

    if generation_concurrency <= 0:
        raise ValueError("generation_concurrency must be > 0")

    if _has_running_event_loop():
        LOGGER.debug(
            "Running event loop detected; using synchronous generation fallback."
        )
        for computation in computations:
            _generate_for_computation_sync(
                computation=computation,
                db_path=db_path,
                generate_model=generate_model,
                generation_top_k=generation_top_k,
                generation_rerank=generation_rerank,
                generation_select_evidence=generation_select_evidence,
                generation_select_k=generation_select_k,
                generation_quote_first=generation_quote_first,
                generation_cite_chunk_index=generation_cite_chunk_index,
                generation_repair_citations=generation_repair_citations,
                generation_repair_max_attempts=generation_repair_max_attempts,
                cache=cache,
                prompt_version=prompt_version,
            )
        return

    asyncio.run(
        _run_generation_stage_async(
            computations=computations,
            db_path=db_path,
            generate_model=generate_model,
            generation_top_k=generation_top_k,
            generation_rerank=generation_rerank,
            generation_select_evidence=generation_select_evidence,
            generation_select_k=generation_select_k,
            generation_quote_first=generation_quote_first,
            generation_cite_chunk_index=generation_cite_chunk_index,
            generation_repair_citations=generation_repair_citations,
            generation_repair_max_attempts=generation_repair_max_attempts,
            generation_concurrency=generation_concurrency,
            cache=cache,
            prompt_version=prompt_version,
        )
    )


def _has_running_event_loop() -> bool:
    """Return whether the current thread already has an active event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def _run_generation_stage_async(
    *,
    computations: Sequence[_EvalComputation],
    db_path: Path,
    generate_model: str,
    generation_top_k: int,
    generation_rerank: Literal["none", "lexical"],
    generation_select_evidence: bool,
    generation_select_k: int,
    generation_quote_first: bool,
    generation_cite_chunk_index: bool,
    generation_repair_citations: bool,
    generation_repair_max_attempts: int,
    generation_concurrency: int,
    cache: EvalCache | None,
    prompt_version: str | None,
) -> None:
    """Asynchronously generate answers and citation metrics for eval computations.

    Args:
        computations: Per-item retrieval/metric computations.
        db_path: SQLite database path for citation scoring.
        generate_model: Generation model name.
        generation_top_k: Number of chunks to pass to generation.
        generation_rerank: Rerank strategy for generation chunks.
        generation_select_evidence: Whether to select evidence chunks before answering.
        generation_select_k: Max chunks to keep when selection is enabled.
        generation_quote_first: Whether to force quote-first generation.
        generation_cite_chunk_index: Whether to force chunk-index citations.
        generation_repair_citations: Whether to repair missing/invalid citations.
        generation_repair_max_attempts: Max attempts for citation repair.
        generation_concurrency: Maximum number of in-flight generation requests.
        cache: Optional eval cache.
        prompt_version: Prompt hash for generated answer cache keys.
    """

    semaphore = asyncio.Semaphore(generation_concurrency)
    answer_locks: dict[_GenerationCacheKey, asyncio.Lock] = {}
    tasks = [
        _generate_for_computation(
            computation=computation,
            db_path=db_path,
            generate_model=generate_model,
            generation_top_k=generation_top_k,
            generation_rerank=generation_rerank,
            generation_select_evidence=generation_select_evidence,
            generation_select_k=generation_select_k,
            generation_quote_first=generation_quote_first,
            generation_cite_chunk_index=generation_cite_chunk_index,
            generation_repair_citations=generation_repair_citations,
            generation_repair_max_attempts=generation_repair_max_attempts,
            cache=cache,
            prompt_version=prompt_version,
            semaphore=semaphore,
            answer_locks=answer_locks,
        )
        for computation in computations
    ]
    await asyncio.gather(*tasks)


async def _generate_for_computation(
    *,
    computation: _EvalComputation,
    db_path: Path,
    generate_model: str,
    generation_top_k: int,
    generation_rerank: Literal["none", "lexical"],
    generation_select_evidence: bool,
    generation_select_k: int,
    generation_quote_first: bool,
    generation_cite_chunk_index: bool,
    generation_repair_citations: bool,
    generation_repair_max_attempts: int,
    cache: EvalCache | None,
    prompt_version: str | None,
    semaphore: asyncio.Semaphore,
    answer_locks: dict[_GenerationCacheKey, asyncio.Lock],
) -> None:
    """Generate one answer, then parse and score citations for an eval computation.

    This function also uses a per-key async lock when cache is enabled to avoid
    duplicate cold-miss model calls for the same generation key within one run.
    """

    generation_chunks = _build_generation_chunks(
        computation.retrieval_results,
        generation_top_k,
        query=computation.item.query,
        generation_rerank=generation_rerank,
    )
    generation_chunk_uids = {chunk.chunk_uid for chunk in generation_chunks}
    computation.ground_truth_in_generation_context = bool(
        generation_chunk_uids
        & {uid for uid in computation.item.ground_truth.chunk_uids if uid}
    )
    generation_pages = {
        (chunk.paper_id, chunk.page_number) for chunk in generation_chunks
    }
    chunk_uids_hash = _hash_chunk_uids(generation_chunks)
    cache_key = _build_generation_cache_key(
        query=computation.item.query,
        generate_model=generate_model,
        prompt_version=prompt_version,
        chunk_uids_hash=chunk_uids_hash,
    )

    answer: str | None = None
    if cache is not None and cache_key is not None:
        answer = cache.get_generated_answer(
            query=cache_key[0],
            generation_model=cache_key[1],
            prompt_version=cache_key[2],
            chunk_uids_hash=cache_key[3],
        )

    if answer is None and cache is not None and cache_key is not None:
        lock = answer_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            answer = cache.get_generated_answer(
                query=cache_key[0],
                generation_model=cache_key[1],
                prompt_version=cache_key[2],
                chunk_uids_hash=cache_key[3],
            )
            if answer is None:
                answer = await _generate_answer_with_semaphore(
                    computation=computation,
                    generate_model=generate_model,
                    generation_chunks=generation_chunks,
                    generation_select_evidence=generation_select_evidence,
                    generation_select_k=generation_select_k,
                    generation_quote_first=generation_quote_first,
                    generation_cite_chunk_index=generation_cite_chunk_index,
                    generation_repair_citations=generation_repair_citations,
                    generation_repair_max_attempts=generation_repair_max_attempts,
                    semaphore=semaphore,
                )
                if answer is not None:
                    cache.set_generated_answer(
                        query=cache_key[0],
                        generation_model=cache_key[1],
                        prompt_version=cache_key[2],
                        chunk_uids_hash=cache_key[3],
                        answer=answer,
                    )
    elif answer is None:
        answer = await _generate_answer_with_semaphore(
            computation=computation,
            generate_model=generate_model,
            generation_chunks=generation_chunks,
            generation_select_evidence=generation_select_evidence,
            generation_select_k=generation_select_k,
            generation_quote_first=generation_quote_first,
            generation_cite_chunk_index=generation_cite_chunk_index,
            generation_repair_citations=generation_repair_citations,
            generation_repair_max_attempts=generation_repair_max_attempts,
            semaphore=semaphore,
        )

    if answer is None:
        return

    try:
        citations = parse_citations(answer)
        computation.citation_count = len(citations)
        citations_in_context = sum(
            1
            for citation in citations
            if (citation.paper_id, citation.page_number) in generation_pages
        )
        computation.citations_in_generation_context = citations_in_context
        computation.citations_outside_generation_context = (
            len(citations) - citations_in_context
        )
        computation.citation_accuracy = _score_citations_for_item(
            citations,
            ground_truth_chunk_uids=computation.item.ground_truth.chunk_uids,
            db_path=db_path,
        )
    except ValueError as exc:
        computation.citation_error = str(exc)


def _generate_for_computation_sync(
    *,
    computation: _EvalComputation,
    db_path: Path,
    generate_model: str,
    generation_top_k: int,
    generation_rerank: Literal["none", "lexical"],
    generation_select_evidence: bool,
    generation_select_k: int,
    generation_quote_first: bool,
    generation_cite_chunk_index: bool,
    generation_repair_citations: bool,
    generation_repair_max_attempts: int,
    cache: EvalCache | None,
    prompt_version: str | None,
) -> None:
    """Generate one answer synchronously and update citation fields."""

    generation_chunks = _build_generation_chunks(
        computation.retrieval_results,
        generation_top_k,
        query=computation.item.query,
        generation_rerank=generation_rerank,
    )
    generation_chunk_uids = {chunk.chunk_uid for chunk in generation_chunks}
    computation.ground_truth_in_generation_context = bool(
        generation_chunk_uids
        & {uid for uid in computation.item.ground_truth.chunk_uids if uid}
    )
    generation_pages = {
        (chunk.paper_id, chunk.page_number) for chunk in generation_chunks
    }
    chunk_uids_hash = _hash_chunk_uids(generation_chunks)
    cache_key = _build_generation_cache_key(
        query=computation.item.query,
        generate_model=generate_model,
        prompt_version=prompt_version,
        chunk_uids_hash=chunk_uids_hash,
    )

    answer: str | None = None
    if cache is not None and cache_key is not None:
        answer = cache.get_generated_answer(
            query=cache_key[0],
            generation_model=cache_key[1],
            prompt_version=cache_key[2],
            chunk_uids_hash=cache_key[3],
        )

    if answer is None:
        answer = _generate_answer_sync(
            computation=computation,
            generate_model=generate_model,
            generation_chunks=generation_chunks,
            generation_select_evidence=generation_select_evidence,
            generation_select_k=generation_select_k,
            generation_quote_first=generation_quote_first,
            generation_cite_chunk_index=generation_cite_chunk_index,
            generation_repair_citations=generation_repair_citations,
            generation_repair_max_attempts=generation_repair_max_attempts,
        )
        if answer is not None and cache is not None and cache_key is not None:
            cache.set_generated_answer(
                query=cache_key[0],
                generation_model=cache_key[1],
                prompt_version=cache_key[2],
                chunk_uids_hash=cache_key[3],
                answer=answer,
            )

    if answer is None:
        return

    try:
        citations = parse_citations(answer)
        computation.citation_count = len(citations)
        citations_in_context = sum(
            1
            for citation in citations
            if (citation.paper_id, citation.page_number) in generation_pages
        )
        computation.citations_in_generation_context = citations_in_context
        computation.citations_outside_generation_context = (
            len(citations) - citations_in_context
        )
        computation.citation_accuracy = _score_citations_for_item(
            citations,
            ground_truth_chunk_uids=computation.item.ground_truth.chunk_uids,
            db_path=db_path,
        )
    except ValueError as exc:
        computation.citation_error = str(exc)


def _generate_answer_sync(
    *,
    computation: _EvalComputation,
    generate_model: str,
    generation_chunks: list[GenerationChunk],
    generation_select_evidence: bool,
    generation_select_k: int,
    generation_quote_first: bool,
    generation_cite_chunk_index: bool,
    generation_repair_citations: bool,
    generation_repair_max_attempts: int,
) -> str | None:
    """Generate an answer synchronously and capture generation errors."""

    try:
        return generate_answer(
            computation.item.query,
            generation_chunks,
            model=generate_model,
            select_evidence=generation_select_evidence,
            selection_max_chunks=generation_select_k,
            quote_first=generation_quote_first,
            cite_chunk_index=generation_cite_chunk_index,
            repair_citations=generation_repair_citations,
            repair_max_attempts=generation_repair_max_attempts,
        )
    except ValueError as exc:
        computation.generation_error = str(exc)
    except RuntimeError as exc:
        computation.generation_error = str(exc)
    return None


async def _generate_answer_with_semaphore(
    *,
    computation: _EvalComputation,
    generate_model: str,
    generation_chunks: list[GenerationChunk],
    generation_select_evidence: bool,
    generation_select_k: int,
    generation_quote_first: bool,
    generation_cite_chunk_index: bool,
    generation_repair_citations: bool,
    generation_repair_max_attempts: int,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Generate an answer under a concurrency semaphore.

    Args:
        computation: Per-item eval computation state.
        generate_model: Generation model name.
        generation_chunks: Retrieved evidence chunks for generation.
        semaphore: Concurrency limiter for generation calls.
    Returns:
        Generated answer text, or None when generation fails.
    """

    async with semaphore:
        try:
            return await asyncio.to_thread(
                generate_answer,
                computation.item.query,
                generation_chunks,
                generate_model,
                select_evidence=generation_select_evidence,
                selection_max_chunks=generation_select_k,
                quote_first=generation_quote_first,
                cite_chunk_index=generation_cite_chunk_index,
                repair_citations=generation_repair_citations,
                repair_max_attempts=generation_repair_max_attempts,
            )
        except ValueError as exc:
            computation.generation_error = str(exc)
        except RuntimeError as exc:
            computation.generation_error = str(exc)
    return None


def _build_generation_cache_key(
    *,
    query: str,
    generate_model: str,
    prompt_version: str | None,
    chunk_uids_hash: str,
) -> _GenerationCacheKey | None:
    """Build a generated-answer cache key when prompt_version is available."""

    if prompt_version is None:
        return None
    return (query, generate_model, prompt_version, chunk_uids_hash)


def _run_retrieval(
    *,
    query: str,
    db_path: Path,
    retrieval_config: RetrievalConfig,
    embeddings_client: CachedEmbeddingsClient | None = None,
    chroma_config: ChromaConfig | None = None,
) -> tuple[list[ChunkResult | HybridChunkResult], list[str]]:
    """Run retrieval for evaluation."""

    warnings: list[str] = []
    results: list[ChunkResult | HybridChunkResult] = []
    if retrieval_config.mode == "fts":
        results = list(search_fts(query, top_k=retrieval_config.top_k, db_path=db_path))
    elif retrieval_config.mode == "vector":
        active_embeddings_client = embeddings_client or CachedEmbeddingsClient(
            base_client=EmbeddingsClient(
                EmbeddingsConfig(model=retrieval_config.model)
            ),
            cache=None,
        )
        active_chroma_config = chroma_config or ChromaConfig(
            persist_dir=retrieval_config.chroma_dir,
            collection_name=retrieval_config.collection,
            distance=retrieval_config.distance,
        )
        try:
            results = list(
                search_vector_chroma(
                    query,
                    top_k=retrieval_config.top_k,
                    db_path=db_path,
                    embeddings_client=active_embeddings_client,
                    chroma_config=active_chroma_config,
                )
            )
        except ImportError as exc:
            raise ValueError(str(exc)) from exc
    else:
        active_embeddings_client = embeddings_client or CachedEmbeddingsClient(
            base_client=EmbeddingsClient(
                EmbeddingsConfig(model=retrieval_config.model)
            ),
            cache=None,
        )
        active_chroma_config = chroma_config or ChromaConfig(
            persist_dir=retrieval_config.chroma_dir,
            collection_name=retrieval_config.collection,
            distance=retrieval_config.distance,
        )
        output = search_hybrid(
            query,
            top_k=retrieval_config.top_k,
            db_path=db_path,
            embeddings_client=active_embeddings_client,
            chroma_config=active_chroma_config,
            rrf_k=retrieval_config.rrf_k,
            fts_weight=retrieval_config.rrf_weight_fts,
            vector_weight=retrieval_config.rrf_weight_vector,
        )
        warnings.extend(output.warnings)
        results = list(output.results)

    return results, warnings


def _compute_prompt_version(
    *,
    generation_select_evidence: bool,
    generation_select_k: int,
    generation_quote_first: bool,
    generation_cite_chunk_index: bool,
    generation_repair_citations: bool,
    generation_repair_max_attempts: int,
) -> str:
    """Compute a stable prompt version hash for generation cache keys."""

    payload_parts = [load_prompt_template()]
    payload_parts.append(f"selection_enabled={generation_select_evidence}")
    payload_parts.append(f"selection_max_chunks={generation_select_k}")
    payload_parts.append(f"quote_first_enabled={generation_quote_first}")
    payload_parts.append(f"chunk_citation_enabled={generation_cite_chunk_index}")
    payload_parts.append(f"repair_enabled={generation_repair_citations}")
    payload_parts.append(f"repair_max_attempts={generation_repair_max_attempts}")
    if generation_select_evidence:
        payload_parts.append(load_selection_prompt_template())
    if generation_quote_first:
        payload_parts.append(load_quote_selection_prompt_template())
        payload_parts.append(load_quote_first_prompt_template())
    if generation_cite_chunk_index:
        payload_parts.append(load_chunk_citation_prompt_template())
    if generation_repair_citations:
        payload_parts.append(load_repair_prompt_template())
    payload = "\n".join(payload_parts)
    return sha256(payload.encode("utf-8")).hexdigest()


def _hash_chunk_uids(chunks: Sequence[GenerationChunk]) -> str:
    """Compute a stable hash for ordered generation chunk UIDs."""

    payload = "\n".join(chunk.chunk_uid for chunk in chunks)
    return sha256(payload.encode("utf-8")).hexdigest()


def _build_generation_chunks(
    results: Sequence[ChunkResult | HybridChunkResult],
    top_k: int,
    *,
    query: str,
    generation_rerank: Literal["none", "lexical"],
) -> list[GenerationChunk]:
    """Build generation chunks from retrieval results.

    Args:
        results: Retrieved chunk results.
        top_k: Number of chunks to include.
        query: User query text used for reranking.
        generation_rerank: Rerank strategy for generation chunks.
    Returns:
        List of GenerationChunk entries.
    """

    if generation_rerank == "lexical":
        results = rerank_results_for_generation(query, results)

    return [
        GenerationChunk(
            chunk_uid=result.chunk_uid,
            paper_id=result.paper_id,
            page_number=result.page_number,
            text=result.text,
        )
        for result in results[:top_k]
    ]


def _score_citations_for_item(
    citations: Sequence[CitationRecord],
    *,
    ground_truth_chunk_uids: Iterable[str],
    db_path: Path,
) -> float:
    """Score citation accuracy for an item."""

    if not citations:
        return 0.0

    pages = [(citation.paper_id, citation.page_number) for citation in citations]
    chunk_uids_by_page = load_chunk_uids_by_page(db_path, pages)
    return compute_citation_accuracy(
        citations,
        ground_truth_chunk_uids=ground_truth_chunk_uids,
        chunk_uids_by_page=chunk_uids_by_page,
    )


def _summarize_failures(results: Sequence[EvalItemResult]) -> EvalFailureSummary:
    """Aggregate failure modes from per-item results."""

    retrieval_empty = len([item for item in results if not item.retrieved_chunk_uids])
    recall_at_5_zero = len([item for item in results if item.recall_at_5 == 0.0])
    mrr_zero = len([item for item in results if item.mrr == 0.0])
    citation_absent = len(
        [
            item
            for item in results
            if item.citation_count is not None and item.citation_count == 0
        ]
    )
    citation_zero_score = len(
        [
            item
            for item in results
            if (
                item.citation_accuracy is not None
                and item.citation_count is not None
                and item.citation_count > 0
                and item.citation_accuracy == 0.0
            )
        ]
    )
    citation_parse_error = len([item for item in results if item.citation_error])
    citation_inaccurate = len(
        [
            item
            for item in results
            if item.citation_accuracy is not None and item.citation_accuracy < 1.0
        ]
    )

    return EvalFailureSummary(
        retrieval_empty=retrieval_empty,
        recall_at_5_zero=recall_at_5_zero,
        mrr_zero=mrr_zero,
        citation_absent=citation_absent,
        citation_zero_score=citation_zero_score,
        citation_parse_error=citation_parse_error,
        citation_inaccurate=citation_inaccurate,
    )


def _mean(values: Sequence[float]) -> float:
    """Compute the mean of a list of floats."""

    return sum(values) / len(values) if values else 0.0


def render_report_markdown(report: EvalReport) -> str:
    """Render a Markdown evaluation report.

    Args:
        report: EvalReport instance.
    Returns:
        Markdown-formatted report string.
    """

    summary = report.summary
    lines = [
        "# Eval Report",
        "",
        f"Created: {report.created}",
        "",
        "## Summary",
        f"- Recall@5: {summary.recall_at_5:.3f}",
        f"- Recall@10: {summary.recall_at_10:.3f}",
        f"- MRR: {summary.mrr:.3f}",
    ]
    if summary.citation_accuracy is not None:
        lines.append(f"- Citation accuracy: {summary.citation_accuracy:.3f}")
    if summary.citation_accuracy_when_recall5_hit is not None:
        lines.append(
            "- Citation accuracy (Recall@5 > 0): "
            f"{summary.citation_accuracy_when_recall5_hit:.3f}"
        )
    lines.append(f"- Queries: {summary.n_queries}")

    lines.extend(
        [
            "",
            "## Failure Modes",
            f"- Retrieval empty: {report.failures.retrieval_empty}",
            f"- Recall@5 = 0: {report.failures.recall_at_5_zero}",
            f"- MRR = 0: {report.failures.mrr_zero}",
            f"- Citation absent: {report.failures.citation_absent}",
            f"- Citation zero score: {report.failures.citation_zero_score}",
            f"- Citation parse errors: {report.failures.citation_parse_error}",
            f"- Citation inaccurate: {report.failures.citation_inaccurate}",
        ]
    )

    context_items = [
        item
        for item in report.items
        if item.ground_truth_in_generation_context is not None
    ]
    if context_items:
        gt_missing = len(
            [
                item
                for item in context_items
                if not item.ground_truth_in_generation_context
            ]
        )
        citations_outside = len(
            [
                item
                for item in context_items
                if item.citations_outside_generation_context
                and item.citations_outside_generation_context > 0
            ]
        )
        lines.extend(
            [
                "",
                "## Generation Context Diagnostics",
                f"- Ground truth missing from generation context: {gt_missing}",
                f"- Citations outside generation context: {citations_outside}",
            ]
        )

    if worst := [item for item in report.items if item.recall_at_5 == 0.0][:5]:
        lines.extend(["", "## Example Misses (Recall@5 = 0)"])
        lines.extend(f"- {item.query_id}: {item.query}" for item in worst)

    if diagnostics := [
        item
        for item in report.items
        if item.recall_at_5 == 0.0
        or item.citation_error
        or item.generation_error
        or item.citation_count == 0
        or item.ground_truth_in_generation_context is False
        or (
            item.citations_outside_generation_context is not None
            and item.citations_outside_generation_context > 0
        )
    ]:
        lines.extend(["", "## Per-Query Diagnostics"])
        for item in diagnostics:
            parts = [
                f"recall@5={item.recall_at_5:.3f}",
                f"mrr={item.mrr:.3f}",
                f"first_rank={item.first_correct_rank}",
            ]
            if item.citation_error:
                parts.append(f"citation_error={item.citation_error}")
            if item.generation_error:
                parts.append(f"generation_error={item.generation_error}")
            if item.ground_truth_in_generation_context is not None:
                parts.append(f"gt_in_gen_ctx={item.ground_truth_in_generation_context}")
            if item.citations_outside_generation_context is not None:
                parts.append(
                    "citations_outside_gen_ctx="
                    f"{item.citations_outside_generation_context}"
                )
            lines.append(f"- {item.query_id}: {item.query} ({', '.join(parts)})")

    return "\n".join(lines)


def save_eval_report(report: EvalReport, *, output_path: Path) -> None:
    """Save evaluation report as JSON and Markdown.

    Args:
        report: EvalReport instance.
        output_path: Base path (without extension) for report files.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    md_path = output_path.with_suffix(".md")
    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    md_path.write_text(render_report_markdown(report), encoding="utf-8")


@dataclass(frozen=True)
class PaperCoverage:
    """Helper for checking eval paper IDs in a DB.

    Args:
        missing_papers: Paper IDs missing from the database.
        total_papers: Total number of papers referenced.
    """

    missing_papers: list[str]
    total_papers: int


def check_eval_set_coverage(eval_set: EvalSet, *, db_path: Path) -> PaperCoverage:
    """Check whether eval set papers exist in the database.

    Args:
        eval_set: EvalSet instance.
        db_path: SQLite database path.
    Returns:
        PaperCoverage with missing IDs.
    Raises:
        FileNotFoundError: If the database does not exist.
    """

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    known_papers = load_paper_ids(db_path)
    expected_papers = {
        paper_id
        for item in eval_set.eval_set
        for paper_id in item.ground_truth.papers
        if paper_id
    }
    missing = sorted({paper for paper in expected_papers if paper not in known_papers})
    return PaperCoverage(missing_papers=missing, total_papers=len(expected_papers))


def normalize_query_embeddings(
    embeddings: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Normalize query embeddings to unit length.

    Args:
        embeddings: Raw embeddings.
    Returns:
        List of normalized embeddings.
    """

    normalized: list[list[float]] = []
    for embedding in embeddings:
        normalized.append(list(normalize_embedding(embedding)))
    return normalized


def inspect_chroma_available() -> bool:
    """Check whether Chroma can be imported.

    Returns:
        True if Chroma can be used, False otherwise.
    """

    try:
        _ = ChromaStore
    except ImportError:
        return False
    return True


__all__ = [
    "EvalGroundTruth",
    "EvalItem",
    "EvalMetadata",
    "EvalSet",
    "EvalItemResult",
    "EvalSummary",
    "EvalReport",
    "EvalGenerationConfig",
    "EvalGenerationClient",
    "RetrievalConfig",
    "load_eval_set",
    "save_eval_set",
    "generate_eval_set",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_citation_accuracy",
    "run_eval",
    "render_report_markdown",
    "save_eval_report",
    "check_eval_set_coverage",
]
