"""Answer generation with citation-aware prompts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

LOGGER = logging.getLogger(__name__)
_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "generate_with_citations.txt"
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

    prompt_path = path or _DEFAULT_PROMPT_PATH
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    template = prompt_path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Prompt template is empty: {prompt_path}")
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
            raise ValueError(f"Chunk {index} has invalid page_number={chunk.page_number}")
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


def generate_answer(
    query: str,
    chunks: list[Chunk],
    model: str = "gpt-4o-mini",
) -> str:
    """Generate an answer using provided chunks with citations.

    Args:
        query: User question.
        chunks: Chunk payloads used as evidence.
        model: OpenAI model name for generation.
    Returns:
        Generated answer string.
    Raises:
        ValueError: If inputs are invalid.
    """

    if not query.strip():
        raise ValueError("query must be non-empty")

    validate_chunks(chunks)

    config = GenerationConfig(model=model)
    template = load_prompt_template(config.prompt_path)
    chunks_text = format_chunks(chunks)
    prompt = render_prompt(template, chunks_text)

    LOGGER.debug("Generation input question=%s", query)
    LOGGER.debug(
        "Generation context chunks=%s total_chars=%s",
        len(chunks),
        len(chunks_text),
    )
    for chunk in chunks:
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
    LOGGER.debug("Generation output chars=%s", len(answer))
    return answer
