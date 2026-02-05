"""Answer generation with citation-aware prompts."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Sequence

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from arxiv_rag.verify import (
    CITATION_PATTERN,
    normalize_whitespace,
    parse_citation_quotes_with_errors,
    parse_citations,
)

LOGGER = logging.getLogger(__name__)
_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "generate_with_citations.txt"
)
_DEFAULT_CHUNK_CITATION_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "generate_with_chunk_citations.txt"
)
_DEFAULT_SELECTION_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "select_evidence.txt"
)
_DEFAULT_QUOTE_SELECTION_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "select_quote.txt"
)
_DEFAULT_REPAIR_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "repair_citations.txt"
)
_DEFAULT_QUOTE_FIRST_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "generate_quote_first.txt"
)
_MIN_QUOTE_OVERLAP = 0.6
_MIN_SNIPPET_OVERLAP = 0.25
_SELECTION_MAX_OUTPUT_TOKENS = 300
_QUOTE_SELECTION_MAX_OUTPUT_TOKENS = 300
_REPAIR_MAX_OUTPUT_TOKENS = 1200
_CHUNK_CITATION_PATTERN = re.compile(r"\[chunk:(?P<index>\d+)\]", re.IGNORECASE)
_SNIPPET_STOPWORDS = frozenset(
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


@dataclass(frozen=True)
class Chunk:
    """Chunk payload required for answer generation.

    Args:
        paper_id: Base arXiv identifier.
        page_number: 1-based page number in the source PDF.
        text: Chunk text content.
        chunk_uid: Stable chunk identifier.
        title: Optional paper title for display.
    """

    paper_id: str
    page_number: int
    text: str
    chunk_uid: str
    title: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for OpenAI generation requests.

    Args:
        model: Chat completion model name.
        temperature: Sampling temperature for generation.
        max_output_tokens: Maximum output tokens to generate.
        request_timeout_s: Timeout for a single generation request in seconds.
        max_retries: Maximum retry attempts for transient errors.
        prompt_path: Optional path to the system prompt template.
        cost_per_1k_tokens: Optional cost per 1k tokens for logging estimates.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_output_tokens: int = 1000
    request_timeout_s: float = 60.0
    max_retries: int = 5
    prompt_path: Path | None = None
    cost_per_1k_tokens: float | None = None


class GenerationClient:
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(
        self,
        config: GenerationConfig,
        client: OpenAI | None = None,
    ) -> None:
        """Initialize the generation client.

        Args:
            config: Generation configuration.
            client: Optional OpenAI client override for testing.
        """

        self._config = config
        self._client = client or OpenAI()

    @property
    def config(self) -> GenerationConfig:
        """Return the active generation configuration."""

        return self._config

    def generate(self, *, prompt: str, query: str) -> str:
        """Generate an answer using the configured model.

        Args:
            prompt: System prompt containing chunk context.
            query: User question.
        Returns:
            Generated answer text.
        Raises:
            ValueError: If prompt or query is empty.
            APIError: For non-retryable OpenAI API errors.
        """

        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not query.strip():
            raise ValueError("query must be non-empty")

        response = self._create_completion(prompt=prompt, query=query)
        request_id = getattr(response, "id", None)
        if request_id:
            LOGGER.debug("Generation request_id=%s", request_id)

        usage = getattr(response, "usage", None)
        if usage is not None:
            LOGGER.debug(
                "Generation usage: prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
            if self._config.cost_per_1k_tokens is not None:
                cost = (usage.total_tokens / 1000.0) * self._config.cost_per_1k_tokens
                LOGGER.debug("Generation estimated cost: $%.6f", cost)

        if not response.choices:
            raise ValueError("Generation response contained no choices.")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Generation response contained empty content.")
        return content

    def _create_completion(self, *, prompt: str, query: str):
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
                        {"role": "user", "content": query},
                    ],
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_output_tokens,
                    timeout=self._config.request_timeout_s,
                )
        raise RuntimeError("Unexpected retry exhaustion for generation request.")


def load_prompt_template(path: Path | None = None) -> str:
    """Load the prompt template from disk.

    Args:
        path: Optional override for the prompt template path.
    Returns:
        Prompt template text.
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt template is empty.
    """

    if path is not None:
        prompt_path = path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = _load_packaged_prompt_template()
        if template is None:
            prompt_path = _DEFAULT_PROMPT_PATH
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
            template = prompt_path.read_text(encoding="utf-8")
    if not template.strip():
        location = path or _DEFAULT_PROMPT_PATH
        raise ValueError(f"Prompt template is empty: {location}")
    return template


def load_chunk_citation_prompt_template(path: Path | None = None) -> str:
    """Load the chunk-citation prompt template from disk.

    Args:
        path: Optional override for the prompt template path.
    Returns:
        Prompt template text.
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt template is empty.
    """

    if path is not None:
        prompt_path = path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = _load_packaged_prompt("generate_with_chunk_citations.txt")
        if template is None:
            prompt_path = _DEFAULT_CHUNK_CITATION_PROMPT_PATH
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
            template = prompt_path.read_text(encoding="utf-8")
    if not template.strip():
        location = path or _DEFAULT_CHUNK_CITATION_PROMPT_PATH
        raise ValueError(f"Prompt template is empty: {location}")
    return template


def _load_packaged_prompt_template() -> str | None:
    """Load the prompt template from package resources.

    Returns:
        Prompt template string if found, otherwise None.
    """

    try:
        resource = resources.files("arxiv_rag").joinpath(
            "prompts",
            "generate_with_citations.txt",
        )
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        return None
    return None


def _load_packaged_prompt(name: str) -> str | None:
    """Load a prompt template from packaged resources.

    Args:
        name: Prompt filename under arxiv_rag/prompts.
    Returns:
        Prompt template string if found, otherwise None.
    """

    try:
        resource = resources.files("arxiv_rag").joinpath("prompts", name)
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        return None
    return None


def load_selection_prompt_template(path: Path | None = None) -> str:
    """Load the evidence-selection prompt template from disk.

    Args:
        path: Optional override for the prompt template path.
    Returns:
        Prompt template text.
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt template is empty.
    """

    if path is not None:
        prompt_path = path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = _load_packaged_prompt("select_evidence.txt")
        if template is None:
            prompt_path = _DEFAULT_SELECTION_PROMPT_PATH
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
            template = prompt_path.read_text(encoding="utf-8")
    if not template.strip():
        location = path or _DEFAULT_SELECTION_PROMPT_PATH
        raise ValueError(f"Prompt template is empty: {location}")
    return template


def load_quote_selection_prompt_template(path: Path | None = None) -> str:
    """Load the quote-selection prompt template from disk.

    Args:
        path: Optional override for the prompt template path.
    Returns:
        Prompt template text.
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt template is empty.
    """

    if path is not None:
        prompt_path = path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = _load_packaged_prompt("select_quote.txt")
        if template is None:
            prompt_path = _DEFAULT_QUOTE_SELECTION_PROMPT_PATH
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
            template = prompt_path.read_text(encoding="utf-8")
    if not template.strip():
        location = path or _DEFAULT_QUOTE_SELECTION_PROMPT_PATH
        raise ValueError(f"Prompt template is empty: {location}")
    return template


def load_quote_first_prompt_template(path: Path | None = None) -> str:
    """Load the quote-first generation prompt template from disk.

    Args:
        path: Optional override for the prompt template path.
    Returns:
        Prompt template text.
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt template is empty.
    """

    if path is not None:
        prompt_path = path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = _load_packaged_prompt("generate_quote_first.txt")
        if template is None:
            prompt_path = _DEFAULT_QUOTE_FIRST_PROMPT_PATH
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
            template = prompt_path.read_text(encoding="utf-8")
    if not template.strip():
        location = path or _DEFAULT_QUOTE_FIRST_PROMPT_PATH
        raise ValueError(f"Prompt template is empty: {location}")
    return template


def load_repair_prompt_template(path: Path | None = None) -> str:
    """Load the citation-repair prompt template from disk.

    Args:
        path: Optional override for the prompt template path.
    Returns:
        Prompt template text.
    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the prompt template is empty.
    """

    if path is not None:
        prompt_path = path
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        template = prompt_path.read_text(encoding="utf-8")
    else:
        template = _load_packaged_prompt("repair_citations.txt")
        if template is None:
            prompt_path = _DEFAULT_REPAIR_PROMPT_PATH
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
            template = prompt_path.read_text(encoding="utf-8")
    if not template.strip():
        location = path or _DEFAULT_REPAIR_PROMPT_PATH
        raise ValueError(f"Prompt template is empty: {location}")
    return template


def validate_chunks(chunks: Sequence[Chunk]) -> None:
    """Validate chunk payloads for generation.

    Args:
        chunks: Chunk payloads to validate.
    Raises:
        ValueError: When chunks are missing required fields.
    """

    if not chunks:
        raise ValueError("chunks must be non-empty")

    for index, chunk in enumerate(chunks, start=1):
        if not chunk.paper_id:
            raise ValueError(f"Chunk {index} missing paper_id")
        if not chunk.chunk_uid:
            raise ValueError(f"Chunk {index} missing chunk_uid")
        if chunk.page_number <= 0:
            raise ValueError(
                f"Chunk {index} has invalid page_number={chunk.page_number}"
            )
        if not chunk.text.strip():
            raise ValueError(f"Chunk {index} missing text")


def format_chunks(chunks: Sequence[Chunk]) -> str:
    """Format chunks for insertion into the prompt template.

    Args:
        chunks: Chunk payloads to format.
    Returns:
        Rendered chunk context string.
    """

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = f" {chunk.title}" if chunk.title else ""
        header = (
            f"Chunk {index}: [arXiv:{chunk.paper_id} p.{chunk.page_number}]"
            f" (chunk_uid={chunk.chunk_uid}){title}"
        )
        lines.append(f"{header}\n{chunk.text}")
    return "\n\n".join(lines)


def render_prompt(template: str, chunks_text: str) -> str:
    """Render the system prompt with formatted chunk context.

    Args:
        template: Prompt template text containing a "{chunks}" placeholder.
        chunks_text: Formatted chunk context.
    Returns:
        Rendered prompt string.
    Raises:
        ValueError: If the template does not contain the placeholder.
    """

    if "{chunks}" not in template:
        raise ValueError("Prompt template missing '{chunks}' placeholder")
    return template.replace("{chunks}", chunks_text)


def render_chunk_citation_prompt(template: str, chunks_text: str) -> str:
    """Render the chunk-citation prompt with formatted chunk context.

    Args:
        template: Prompt template text containing a "{chunks}" placeholder.
        chunks_text: Formatted chunk context.
    Returns:
        Rendered prompt string.
    Raises:
        ValueError: If the template does not contain the placeholder.
    """

    if "{chunks}" not in template:
        raise ValueError("Chunk citation prompt missing '{chunks}' placeholder")
    return template.replace("{chunks}", chunks_text)


def render_selection_prompt(
    template: str,
    chunks_text: str,
    *,
    query: str,
    max_chunks: int,
) -> str:
    """Render the evidence-selection prompt template.

    Args:
        template: Prompt template text.
        chunks_text: Formatted chunk context.
        query: User question text.
        max_chunks: Maximum number of chunks to select.
    Returns:
        Rendered prompt string.
    Raises:
        ValueError: If required placeholders are missing.
    """

    if "{chunks}" not in template:
        raise ValueError("Selection prompt missing '{chunks}' placeholder")
    if "{query}" not in template:
        raise ValueError("Selection prompt missing '{query}' placeholder")
    if "{max_chunks}" not in template:
        raise ValueError("Selection prompt missing '{max_chunks}' placeholder")
    return template.format(chunks=chunks_text, query=query, max_chunks=max_chunks)


def render_repair_prompt(
    template: str,
    chunks_text: str,
    *,
    query: str,
    answer: str,
) -> str:
    """Render the citation-repair prompt template.

    Args:
        template: Prompt template text.
        chunks_text: Formatted chunk context.
        query: User question text.
        answer: Answer text to repair.
    Returns:
        Rendered prompt string.
    Raises:
        ValueError: If required placeholders are missing.
    """

    if "{chunks}" not in template:
        raise ValueError("Repair prompt missing '{chunks}' placeholder")
    if "{query}" not in template:
        raise ValueError("Repair prompt missing '{query}' placeholder")
    if "{answer}" not in template:
        raise ValueError("Repair prompt missing '{answer}' placeholder")
    return template.format(chunks=chunks_text, query=query, answer=answer)


def render_quote_selection_prompt(
    template: str,
    chunks_text: str,
    *,
    query: str,
) -> str:
    """Render the quote-selection prompt template.

    Args:
        template: Prompt template text.
        chunks_text: Formatted chunk context.
        query: User question text.
    Returns:
        Rendered prompt string.
    Raises:
        ValueError: If required placeholders are missing.
    """

    if "{chunks}" not in template:
        raise ValueError("Quote selection prompt missing '{chunks}' placeholder")
    if "{query}" not in template:
        raise ValueError("Quote selection prompt missing '{query}' placeholder")
    return template.format(chunks=chunks_text, query=query)


def render_quote_first_prompt(
    template: str,
    chunks_text: str,
    *,
    query: str,
    quote: str,
) -> str:
    """Render the quote-first generation prompt template.

    Args:
        template: Prompt template text.
        chunks_text: Formatted chunk context.
        query: User question text.
        quote: Selected verbatim quote to include in the answer.
    Returns:
        Rendered prompt string.
    Raises:
        ValueError: If required placeholders are missing.
    """

    if "{chunks}" not in template:
        raise ValueError("Quote-first prompt missing '{chunks}' placeholder")
    if "{query}" not in template:
        raise ValueError("Quote-first prompt missing '{query}' placeholder")
    if "{quote}" not in template:
        raise ValueError("Quote-first prompt missing '{quote}' placeholder")
    return template.format(chunks=chunks_text, query=query, quote=quote)


def _parse_selection_indices(selection_text: str, *, max_index: int) -> list[int]:
    """Parse chunk indices from selection output.

    Args:
        selection_text: Raw model output for selection.
        max_index: Maximum allowed index.
    Returns:
        Ordered list of unique indices within [1, max_index].
    """

    if not selection_text.strip():
        return []

    candidates: list[int] = []
    try:
        payload = json.loads(selection_text)
    except json.JSONDecodeError:
        payload = None

    if payload is None:
        start = selection_text.find("{")
        end = selection_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = selection_text[start : end + 1]
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, dict):
        selected = payload.get("selected")
        if isinstance(selected, list):
            for item in selected:
                if isinstance(item, int):
                    candidates.append(item)
                elif isinstance(item, str) and item.isdigit():
                    candidates.append(int(item))

    if not candidates:
        for match in re.findall(r"\d+", selection_text):
            candidates.append(int(match))

    seen: set[int] = set()
    indices: list[int] = []
    for idx in candidates:
        if 1 <= idx <= max_index and idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _parse_quote_selection(
    selection_text: str,
    *,
    max_index: int,
) -> tuple[int, str] | None:
    """Parse a quote selection payload from model output.

    Args:
        selection_text: Raw model output for quote selection.
        max_index: Maximum allowed chunk index.
    Returns:
        Tuple of (chunk_index, quote) or None when parsing fails.
    """

    if not selection_text.strip():
        return None

    payload = None
    try:
        payload = json.loads(selection_text)
    except json.JSONDecodeError:
        start = selection_text.find("{")
        end = selection_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = selection_text[start : end + 1]
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        return None

    chunk_index = payload.get("chunk_index")
    quote = payload.get("quote")
    if isinstance(chunk_index, str) and chunk_index.isdigit():
        chunk_index = int(chunk_index)

    if not isinstance(chunk_index, int) or not isinstance(quote, str):
        return None
    if not (1 <= chunk_index <= max_index):
        return None

    quote = normalize_whitespace(quote)
    if not quote:
        return None
    return chunk_index, quote


def _quote_matches_chunk(quote: str, chunk_text: str) -> bool:
    """Return whether a quote is a substring of the chunk text."""

    normalized_quote = normalize_whitespace(quote).lower()
    normalized_chunk = normalize_whitespace(chunk_text).lower()
    if not normalized_quote or not normalized_chunk:
        return False
    return normalized_quote in normalized_chunk


def select_quote_snippet(
    query: str,
    chunks: Sequence[Chunk],
    *,
    model: str = "gpt-4o-mini",
) -> tuple[Chunk, str] | None:
    """Select a verbatim quote snippet from the provided chunks.

    Args:
        query: User question text.
        chunks: Candidate chunks for selection.
        model: OpenAI model name for quote selection.
    Returns:
        Tuple of (selected chunk, quote) or None when selection fails.
    Raises:
        ValueError: If inputs are invalid.
    """

    if not query.strip():
        raise ValueError("query must be non-empty")
    validate_chunks(chunks)

    template = load_quote_selection_prompt_template()
    chunks_text = format_chunks(chunks)
    prompt = render_quote_selection_prompt(template, chunks_text, query=query)

    config = GenerationConfig(
        model=model,
        temperature=0.0,
        max_output_tokens=_QUOTE_SELECTION_MAX_OUTPUT_TOKENS,
    )
    client = GenerationClient(config)
    selection_text = client.generate(prompt=prompt, query="Return JSON only.")
    selection = _parse_quote_selection(selection_text, max_index=len(chunks))
    if selection is None:
        return None

    chunk_index, quote = selection
    chunk = chunks[chunk_index - 1]
    if not _quote_matches_chunk(quote, chunk.text):
        return None
    return chunk, quote


def select_evidence_chunks(
    query: str,
    chunks: Sequence[Chunk],
    *,
    model: str = "gpt-4o-mini",
    max_chunks: int = 3,
) -> list[Chunk]:
    """Select the most relevant evidence chunks for answering a query.

    Args:
        query: User question text.
        chunks: Candidate chunks for selection.
        model: OpenAI model name for selection.
        max_chunks: Maximum number of chunks to select.
    Returns:
        List of selected chunks (at least one when chunks are provided).
        Always includes the top-ranked chunk as a guardrail, subject to max_chunks.
    Raises:
        ValueError: If inputs are invalid.
    """

    if not query.strip():
        raise ValueError("query must be non-empty")
    validate_chunks(chunks)
    if max_chunks <= 0:
        raise ValueError("max_chunks must be > 0")

    max_chunks = min(max_chunks, len(chunks))
    template = load_selection_prompt_template()
    chunks_text = format_chunks(chunks)
    prompt = render_selection_prompt(
        template,
        chunks_text,
        query=query,
        max_chunks=max_chunks,
    )

    config = GenerationConfig(
        model=model,
        temperature=0.0,
        max_output_tokens=_SELECTION_MAX_OUTPUT_TOKENS,
    )
    client = GenerationClient(config)
    selection_text = client.generate(prompt=prompt, query="Return JSON only.")
    indices = _parse_selection_indices(selection_text, max_index=len(chunks))
    if not indices:
        return [chunks[0]]
    ordered_indices: list[int] = [1]
    for index in indices:
        if index not in ordered_indices:
            ordered_indices.append(index)
        if len(ordered_indices) >= max_chunks:
            break
    return [chunks[index - 1] for index in ordered_indices[:max_chunks]]


def remap_citations_by_quote_overlap(answer: str, chunks: Sequence[Chunk]) -> str:
    """Remap citation paper/page pairs to best matching retrieved chunks.

    Args:
        answer: Generated answer with citations and adjacent quote snippets.
        chunks: Retrieved chunks in ranking order used for generation.
    Returns:
        Answer text with citation targets updated when a better quote-supported
        chunk match is found.
    Edge cases:
        Returns the original answer when citations cannot be parsed.
    """

    if not answer.strip() or not chunks:
        return answer

    try:
        citations = parse_citations(answer)
    except ValueError:
        return answer
    if not citations:
        return answer

    quote_map, _ = parse_citation_quotes_with_errors(answer)
    matches = list(CITATION_PATTERN.finditer(answer))
    if len(matches) != len(citations):
        return answer

    replacements: list[tuple[int, int, str]] = []
    for citation, match in zip(citations, matches, strict=False):
        quote = quote_map.get(citation.citation_id)
        if quote is not None:
            best_chunk = _select_best_chunk_for_quote(quote, chunks)
        else:
            snippet = _extract_sentence_snippet(
                answer,
                start=match.start(),
                end=match.end(),
            )
            best_chunk = _select_best_chunk_for_snippet(snippet, chunks)
        if best_chunk is None:
            continue
        if (
            best_chunk.paper_id == citation.paper_id
            and best_chunk.page_number == citation.page_number
        ):
            continue

        replacements.append(
            (
                match.start(),
                match.end(),
                f"[arXiv:{best_chunk.paper_id} p.{best_chunk.page_number}]",
            )
        )

    if not replacements:
        return answer

    return _apply_citation_replacements(answer, replacements)


def _select_best_chunk_for_quote(quote: str, chunks: Sequence[Chunk]) -> Chunk | None:
    """Choose the highest-overlap chunk for a quote.

    Args:
        quote: Extracted supporting quote snippet.
        chunks: Retrieved chunks in rank order.
    Returns:
        Best matching chunk, or None when no chunk is a reasonable match.
    """

    normalized_quote = normalize_whitespace(quote).lower()
    quote_tokens = _tokenize_for_overlap(normalized_quote)
    if not normalized_quote or not quote_tokens:
        return None

    best_chunk: Chunk | None = None
    best_score = 0.0
    for chunk in chunks:
        score = _score_quote_overlap(
            normalized_quote=normalized_quote,
            quote_tokens=quote_tokens,
            chunk_text=chunk.text,
        )
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score < _MIN_QUOTE_OVERLAP:
        return None
    return best_chunk


def _select_best_chunk_for_snippet(
    snippet: str | None,
    chunks: Sequence[Chunk],
) -> Chunk | None:
    """Choose the highest-overlap chunk for a sentence snippet.

    Args:
        snippet: Sentence fragment near a citation.
        chunks: Retrieved chunks in rank order.
    Returns:
        Best matching chunk, or None when no chunk is a reasonable match.
    """

    if snippet is None:
        return None
    normalized = normalize_whitespace(snippet).lower()
    tokens = _tokenize_for_snippet(normalized)
    if len(tokens) < 3:
        return None

    best_chunk: Chunk | None = None
    best_score = 0.0
    for chunk in chunks:
        score = _score_quote_overlap(
            normalized_quote=normalized,
            quote_tokens=tokens,
            chunk_text=chunk.text,
        )
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score < _MIN_SNIPPET_OVERLAP:
        return None
    return best_chunk


def _extract_sentence_snippet(answer: str, *, start: int, end: int) -> str | None:
    """Extract the sentence surrounding a citation span.

    Args:
        answer: Full answer text.
        start: Start index of the citation match.
        end: End index of the citation match.
    Returns:
        Sentence snippet with citation markers removed, or None when unavailable.
    """

    if not answer:
        return None

    boundaries = ".!?\n"
    before = [answer.rfind(ch, 0, start) for ch in boundaries]
    start_index = max(before)
    if start_index == -1:
        start_index = 0
    else:
        start_index += 1

    after_candidates = [answer.find(ch, end) for ch in boundaries]
    after_candidates = [idx for idx in after_candidates if idx != -1]
    end_index = min(after_candidates) if after_candidates else len(answer)

    snippet = answer[start_index:end_index]
    snippet = CITATION_PATTERN.sub("", snippet)
    snippet = re.sub(r'\*\"quote\"\\*', "", snippet, flags=re.IGNORECASE)
    snippet = normalize_whitespace(snippet).strip()
    if not snippet:
        return None
    return snippet


def _score_quote_overlap(
    *,
    normalized_quote: str,
    quote_tokens: Sequence[str],
    chunk_text: str,
) -> float:
    """Score quote overlap against one chunk.

    Args:
        normalized_quote: Normalized quote text.
        quote_tokens: Tokenized quote terms.
        chunk_text: Candidate chunk text.
    Returns:
        Overlap score in [0, 1], where 1 means exact normalized substring.
    """

    normalized_chunk = normalize_whitespace(chunk_text).lower()
    if not normalized_chunk:
        return 0.0
    if normalized_quote in normalized_chunk:
        return 1.0

    chunk_tokens = set(_tokenize_for_overlap(normalized_chunk))
    if not chunk_tokens:
        return 0.0
    overlap = sum(1 for token in quote_tokens if token in chunk_tokens)
    return overlap / len(quote_tokens)


def _tokenize_for_overlap(text: str) -> list[str]:
    """Tokenize text for lexical overlap checks."""

    return [token for token in re.findall(r"[a-z0-9]+", text) if len(token) >= 2]


def _tokenize_for_snippet(text: str) -> list[str]:
    """Tokenize text for snippet overlap checks, excluding stop words."""

    tokens = re.findall(r"[a-z0-9]+", text)
    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in _SNIPPET_STOPWORDS
    ]


def _apply_citation_replacements(
    answer: str,
    replacements: Sequence[tuple[int, int, str]],
) -> str:
    """Apply citation string replacements based on character spans."""

    if not replacements:
        return answer

    parts: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(replacements, key=lambda item: item[0]):
        if start < cursor:
            continue
        parts.append(answer[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(answer[cursor:])
    return "".join(parts)


def map_chunk_citations(answer: str, chunks: Sequence[Chunk]) -> str:
    """Map [chunk:N] citations to [arXiv:ID p.PAGE] citations.

    Args:
        answer: Generated answer containing chunk citations.
        chunks: Chunks used to generate the answer.
    Returns:
        Answer with chunk citations replaced when possible.
    """

    if not answer.strip() or not chunks:
        return answer

    replacements: list[tuple[int, int, str]] = []
    for match in _CHUNK_CITATION_PATTERN.finditer(answer):
        index = int(match.group("index"))
        if not (1 <= index <= len(chunks)):
            continue
        chunk = chunks[index - 1]
        replacements.append(
            (
                match.start(),
                match.end(),
                f"[arXiv:{chunk.paper_id} p.{chunk.page_number}]",
            )
        )

    if not replacements:
        return answer
    return _apply_citation_replacements(answer, replacements)


def _needs_citation_repair(answer: str) -> bool:
    """Check whether the answer needs citation repair."""

    if not answer.strip():
        return False
    try:
        citations = parse_citations(answer)
    except ValueError:
        return True
    if not citations:
        return True
    _, quote_errors = parse_citation_quotes_with_errors(answer)
    if quote_errors:
        return True
    return False


def repair_answer_with_citations(
    answer: str,
    *,
    query: str,
    chunks: Sequence[Chunk],
    model: str,
    max_attempts: int = 1,
) -> str:
    """Repair citations in an answer using the provided chunks.

    Args:
        answer: Generated answer text.
        query: User question text.
        chunks: Evidence chunks to cite.
        model: OpenAI model name for repair.
        max_attempts: Maximum repair attempts when citations are invalid.
    Returns:
        Repaired answer text (may be unchanged).
    Raises:
        ValueError: If inputs are invalid.
    """

    if max_attempts <= 0:
        return answer
    if not answer.strip():
        return answer
    validate_chunks(chunks)

    template = load_repair_prompt_template()
    chunks_text = format_chunks(chunks)
    prompt = render_repair_prompt(template, chunks_text, query=query, answer=answer)

    config = GenerationConfig(
        model=model,
        temperature=0.0,
        max_output_tokens=_REPAIR_MAX_OUTPUT_TOKENS,
    )
    client = GenerationClient(config)

    repaired = answer
    for _ in range(max_attempts):
        if not _needs_citation_repair(repaired):
            break
        repaired = client.generate(prompt=prompt, query="Return the corrected answer.")
        repaired = remap_citations_by_quote_overlap(repaired, chunks)
    return repaired


def generate_answer(
    query: str,
    chunks: list[Chunk],
    model: str = "gpt-4o-mini",
    *,
    select_evidence: bool = False,
    selection_max_chunks: int = 3,
    cite_chunk_index: bool = False,
    quote_first: bool = False,
    repair_citations: bool = False,
    repair_max_attempts: int = 1,
) -> str:
    """Generate an answer using provided chunks with citations.

    Args:
        query: User question.
        chunks: Chunk payloads used as evidence.
        model: OpenAI model name for generation.
        select_evidence: Whether to select a smaller evidence subset before answering.
        selection_max_chunks: Maximum number of chunks to select when enabled.
        cite_chunk_index: Whether to force citations to use [chunk:N] then map.
        quote_first: Whether to force quote-first generation anchored to one chunk.
        repair_citations: Whether to attempt citation repair on invalid outputs.
        repair_max_attempts: Maximum citation repair attempts.
    Returns:
        Generated answer string.
    Raises:
        ValueError: If inputs are invalid.
    """

    if not query.strip():
        raise ValueError("query must be non-empty")

    validate_chunks(chunks)

    working_chunks = list(chunks)
    if select_evidence:
        working_chunks = select_evidence_chunks(
            query,
            working_chunks,
            model=model,
            max_chunks=selection_max_chunks,
        )

    quote = None
    if cite_chunk_index and quote_first:
        raise ValueError("cite_chunk_index and quote_first cannot both be enabled.")

    if quote_first:
        selection = select_quote_snippet(query, working_chunks, model=model)
        if selection is not None:
            quote_chunk, quote = selection
            working_chunks = [quote_chunk]

    config = GenerationConfig(model=model)
    if quote_first and quote is not None:
        template = load_quote_first_prompt_template()
        chunks_text = format_chunks(working_chunks)
        prompt = render_quote_first_prompt(
            template,
            chunks_text,
            query=query,
            quote=quote,
        )
    elif cite_chunk_index:
        template = load_chunk_citation_prompt_template()
        chunks_text = format_chunks(working_chunks)
        prompt = render_chunk_citation_prompt(template, chunks_text)
    else:
        template = load_prompt_template(config.prompt_path)
        chunks_text = format_chunks(working_chunks)
        prompt = render_prompt(template, chunks_text)

    LOGGER.debug("Generation input question=%s", query)
    LOGGER.debug(
        "Generation context chunks=%s total_chars=%s",
        len(chunks),
        len(chunks_text),
    )
    for chunk in working_chunks:
        LOGGER.debug(
            "Generation chunk uid=%s paper_id=%s page=%s text_chars=%s",
            chunk.chunk_uid,
            chunk.paper_id,
            chunk.page_number,
            len(chunk.text),
        )

    client = GenerationClient(config)
    LOGGER.debug(
        "Generation request model=%s temperature=%s max_tokens=%s",
        config.model,
        config.temperature,
        config.max_output_tokens,
    )
    answer = client.generate(prompt=prompt, query=query)
    if cite_chunk_index:
        answer = map_chunk_citations(answer, working_chunks)
    answer = remap_citations_by_quote_overlap(answer, working_chunks)
    if repair_citations:
        answer = repair_answer_with_citations(
            answer,
            query=query,
            chunks=working_chunks,
            model=model,
            max_attempts=repair_max_attempts,
        )
    LOGGER.debug("Generation output chars=%s", len(answer))
    return answer
