"""OpenAI embeddings client wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Configuration for OpenAI embeddings requests.

    Args:
        model: Embedding model name.
        request_timeout_s: Timeout for a single embeddings request in seconds.
        max_retries: Maximum retry attempts for transient errors.
        max_request_tokens: Token cap for all inputs in a single request.
        max_input_tokens: Maximum tokens allowed for any single input.
        cost_per_1k_tokens: Optional cost per 1k tokens for logging estimates.
    """

    model: str = "text-embedding-3-small"
    request_timeout_s: float = 30.0
    max_retries: int = 5
    max_request_tokens: int = 32000
    max_input_tokens: int = 8192
    cost_per_1k_tokens: float | None = None


@dataclass(frozen=True)
class EmbeddingBatchResult:
    """Embeddings result with optional token usage."""

    embeddings: list[list[float]]
    total_tokens: int | None


class EmbeddingsClientLike(Protocol):
    """Structural type for embeddings clients used throughout the project."""

    @property
    def config(self) -> EmbeddingsConfig:  # pragma: no cover - protocol
        """Return the active embeddings configuration."""

    def embed(self, inputs: Sequence[str]) -> EmbeddingBatchResult:  # pragma: no cover
        """Embed input texts using the configured model."""


class EmbeddingsClient:
    """Thin wrapper around the OpenAI embeddings API."""

    def __init__(
        self,
        config: EmbeddingsConfig,
        client: OpenAI | None = None,
    ) -> None:
        """Initialize the embeddings client.

        Args:
            config: Embeddings configuration.
            client: Optional OpenAI client override for testing.
        """

        self._config = config
        self._client = client or OpenAI()

    @property
    def config(self) -> EmbeddingsConfig:
        """Return the active embeddings configuration."""

        return self._config

    def embed(self, inputs: Sequence[str]) -> EmbeddingBatchResult:
        """Embed input texts using the configured model.

        Args:
            inputs: Text inputs to embed.
        Returns:
            EmbeddingBatchResult with embeddings and token usage.
        Raises:
            ValueError: If inputs is empty.
            APIError: For non-retryable OpenAI API errors.
        """

        if not inputs:
            raise ValueError("inputs must be non-empty")

        response = self._create_embeddings(inputs)
        embeddings = [item.embedding for item in response.data]
        usage = getattr(response, "usage", None)
        total_tokens = usage.total_tokens if usage is not None else None

        if total_tokens is not None:
            LOGGER.debug("Embeddings usage: total_tokens=%s", total_tokens)
            if self._config.cost_per_1k_tokens is not None:
                cost = (total_tokens / 1000.0) * self._config.cost_per_1k_tokens
                LOGGER.debug("Embeddings estimated cost: $%.6f", cost)

        return EmbeddingBatchResult(embeddings=embeddings, total_tokens=total_tokens)

    def _create_embeddings(self, inputs: Sequence[str]):
        """Call the OpenAI embeddings API with retry logic."""

        retrying = Retrying(
            retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError)),
            stop=stop_after_attempt(self._config.max_retries),
            wait=wait_exponential_jitter(initial=1, max=20),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self._client.embeddings.create(
                    model=self._config.model,
                    input=list(inputs),
                    timeout=self._config.request_timeout_s,
                )
        raise RuntimeError("Unexpected retry exhaustion for embeddings request.")
