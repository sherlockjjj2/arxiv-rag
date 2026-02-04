"""Evaluation utilities for retrieval and citation accuracy."""

from __future__ import annotations

import json
import logging
import random
import sqlite3
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
from arxiv_rag.embeddings_client import EmbeddingsClient, EmbeddingsConfig
from arxiv_rag.generate import Chunk as GenerationChunk, generate_answer
from arxiv_rag.retrieve import (
    ChunkResult,
    HybridChunkResult,
    search_fts,
    search_hybrid,
    search_vector_chroma,
)
from arxiv_rag.verify import CitationRecord, parse_citations

LOGGER = logging.getLogger(__name__)

_DEFAULT_EVAL_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "generate_eval_qa.txt"
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
        max_output_tokens: Maximum tokens for completion.
        request_timeout_s: Timeout for a single request.
        max_retries: Maximum retry attempts for transient errors.
        prompt_path: Optional override for the prompt template.
        cost_per_1k_tokens: Optional cost per 1k tokens for logging estimates.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.2
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

    def generate_questions(self, *, prompt: str) -> EvalGenerationResult:
        """Generate QA pairs from a prompt.

        Args:
            prompt: Prompt containing chunk context.
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

        questions = parse_generated_questions(content)
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


def render_eval_prompt(template: str, *, chunk: ChunkSample, n_questions: int) -> str:
    """Render the eval QA prompt template.

    Args:
        template: Prompt template text.
        chunk: Chunk sample with metadata and text.
        n_questions: Number of questions to request.
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

    placeholders = {
        "CHUNK_TEXT": chunk.text.replace("$", "$$"),
        "PAPER_ID": chunk.paper_id.replace("$", "$$"),
        "PAGE_NUMBER": str(chunk.page_number),
        "CHUNK_UID": chunk.chunk_uid.replace("$", "$$"),
        "TITLE": (chunk.title or "").replace("$", "$$"),
        "N_QUESTIONS": str(n_questions),
    }
    rendered = Template(template).safe_substitute(placeholders)
    if "$TITLE" in template and not placeholders["TITLE"]:
        LOGGER.debug("Eval QA prompt rendered without title.")
    return rendered


def parse_generated_questions(raw_text: str) -> list[GeneratedQuestion]:
    """Parse generated QA pairs from model output.

    Args:
        raw_text: Raw model output.
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
    max_output_tokens: int,
    request_timeout_s: float,
    max_retries: int,
    prompt_path: Path | None,
    corpus_version: str,
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
        max_output_tokens: Completion token limit.
        request_timeout_s: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        prompt_path: Optional prompt template override.
        corpus_version: Corpus version label.
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

    template = load_eval_prompt_template(prompt_path)
    config = EvalGenerationConfig(
        model=model,
        temperature=temperature,
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
    query_index = 1

    with sqlite3.connect(db_path) as conn:
        for chunk_uid in candidate_uids:
            if len(eval_items) >= n_questions:
                break
            chunk = _load_chunk_sample(conn, chunk_uid)
            if chunk is None:
                continue

            prompt = render_eval_prompt(
                template, chunk=chunk, n_questions=questions_per_chunk
            )
            LOGGER.debug(
                "Eval QA prompt chunk_uid=%s text_chars=%s",
                chunk.chunk_uid,
                len(chunk.text),
            )
            try:
                result = client.generate_questions(prompt=prompt)
            except (ValueError, APIError, APITimeoutError, RateLimitError) as exc:
                LOGGER.warning(
                    "Eval QA generation failed for %s: %s", chunk.chunk_uid, exc
                )
                continue

            for question in result.questions:
                if len(eval_items) >= n_questions:
                    break
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
) -> EvalReport:
    """Run evaluation for an eval set.

    Args:
        eval_set: EvalSet to evaluate.
        db_path: SQLite database path.
        retrieval_config: Retrieval settings.
        generate: Whether to run generation and compute citation accuracy.
        generate_model: Model name for answer generation.
        generation_top_k: Number of chunks to pass to the generator.
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

    results: list[EvalItemResult] = []
    recall_5_scores: list[float] = []
    recall_10_scores: list[float] = []
    mrr_scores: list[float] = []
    citation_scores: list[float] = []
    citation_scores_recall5_hit: list[float] = []

    for item in eval_set.eval_set:
        retrieval_results, warnings = _run_retrieval(
            query=item.query,
            db_path=db_path,
            retrieval_config=retrieval_config,
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

        recall_5_scores.append(recall_5)
        recall_10_scores.append(recall_10)
        mrr_scores.append(mrr)

        citation_count = None
        citation_accuracy = None
        citation_error = None
        generation_error = None

        if generate:
            try:
                answer = generate_answer(
                    item.query,
                    _build_generation_chunks(retrieval_results, generation_top_k),
                    model=generate_model,
                )
            except ValueError as exc:
                generation_error = str(exc)
            except RuntimeError as exc:
                generation_error = str(exc)
            else:
                try:
                    citations = parse_citations(answer)
                    citation_count = len(citations)
                    citation_accuracy = _score_citations_for_item(
                        citations,
                        ground_truth_chunk_uids=item.ground_truth.chunk_uids,
                        db_path=db_path,
                    )
                    citation_scores.append(citation_accuracy)
                    if recall_5 > 0.0:
                        citation_scores_recall5_hit.append(citation_accuracy)
                except ValueError as exc:
                    citation_error = str(exc)

        results.append(
            EvalItemResult(
                query_id=item.query_id,
                query=item.query,
                recall_at_5=recall_5,
                recall_at_10=recall_10,
                mrr=mrr,
                first_correct_rank=first_rank,
                retrieved_chunk_uids=retrieved_uids,
                ground_truth_chunk_uids=list(item.ground_truth.chunk_uids),
                citation_count=citation_count,
                citation_accuracy=citation_accuracy,
                citation_error=citation_error,
                generation_error=generation_error,
                warnings=warnings,
            )
        )

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


def _run_retrieval(
    *,
    query: str,
    db_path: Path,
    retrieval_config: RetrievalConfig,
) -> tuple[list[ChunkResult | HybridChunkResult], list[str]]:
    """Run retrieval for evaluation."""

    warnings: list[str] = []
    if retrieval_config.mode == "fts":
        results = search_fts(query, top_k=retrieval_config.top_k, db_path=db_path)
    elif retrieval_config.mode == "vector":
        embeddings_client = EmbeddingsClient(
            EmbeddingsConfig(model=retrieval_config.model)
        )
        chroma_config = ChromaConfig(
            persist_dir=retrieval_config.chroma_dir,
            collection_name=retrieval_config.collection,
            distance=retrieval_config.distance,
        )
        try:
            results = search_vector_chroma(
                query,
                top_k=retrieval_config.top_k,
                db_path=db_path,
                embeddings_client=embeddings_client,
                chroma_config=chroma_config,
            )
        except ImportError as exc:
            raise ValueError(str(exc)) from exc
    else:
        embeddings_client = EmbeddingsClient(
            EmbeddingsConfig(model=retrieval_config.model)
        )
        chroma_config = ChromaConfig(
            persist_dir=retrieval_config.chroma_dir,
            collection_name=retrieval_config.collection,
            distance=retrieval_config.distance,
        )
        output = search_hybrid(
            query,
            top_k=retrieval_config.top_k,
            db_path=db_path,
            embeddings_client=embeddings_client,
            chroma_config=chroma_config,
            rrf_k=retrieval_config.rrf_k,
            fts_weight=retrieval_config.rrf_weight_fts,
            vector_weight=retrieval_config.rrf_weight_vector,
        )
        warnings.extend(output.warnings)
        results = output.results

    return results, warnings


def _build_generation_chunks(
    results: Sequence[ChunkResult | HybridChunkResult],
    top_k: int,
) -> list[GenerationChunk]:
    """Build generation chunks from retrieval results.

    Args:
        results: Retrieved chunk results.
        top_k: Number of chunks to include.
    Returns:
        List of GenerationChunk entries.
    """

    chunks: list[GenerationChunk] = []
    for result in results[:top_k]:
        chunks.append(
            GenerationChunk(
                chunk_uid=result.chunk_uid,
                paper_id=result.paper_id,
                page_number=result.page_number,
                text=result.text,
            )
        )
    return chunks


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

    retrieval_empty = sum(1 for item in results if not item.retrieved_chunk_uids)
    recall_at_5_zero = sum(1 for item in results if item.recall_at_5 == 0.0)
    mrr_zero = sum(1 for item in results if item.mrr == 0.0)
    citation_absent = sum(
        1
        for item in results
        if item.citation_count is not None and item.citation_count == 0
    )
    citation_zero_score = sum(
        1
        for item in results
        if (
            item.citation_accuracy is not None
            and item.citation_count is not None
            and item.citation_count > 0
            and item.citation_accuracy == 0.0
        )
    )
    citation_parse_error = sum(1 for item in results if item.citation_error)
    citation_inaccurate = sum(
        1
        for item in results
        if item.citation_accuracy is not None and item.citation_accuracy < 1.0
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

    if not values:
        return 0.0
    return sum(values) / len(values)


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

    worst = [item for item in report.items if item.recall_at_5 == 0.0][:5]
    if worst:
        lines.extend(["", "## Example Misses (Recall@5 = 0)"])
        for item in worst:
            lines.append(f"- {item.query_id}: {item.query}")

    diagnostics = [
        item
        for item in report.items
        if item.recall_at_5 == 0.0 or item.citation_error or item.generation_error
    ]
    if diagnostics:
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
