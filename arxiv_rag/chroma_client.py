"""Chroma client wrapper for local vector storage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from chromadb.api import ClientAPI
    from chromadb.api.models import Collection

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChromaConfig:
    """Configuration for connecting to a local Chroma collection.

    Args:
        persist_dir: Path for Chroma persistence.
        collection_name: Collection name to store embeddings in.
        distance: Distance metric for vector search.
    """

    persist_dir: Path
    collection_name: str
    distance: Literal["cosine", "l2", "ip"] = "cosine"


class ChromaStore:
    """Thin wrapper around a local Chroma persistent client."""

    def __init__(
        self,
        config: ChromaConfig,
        client: ClientAPI | None = None,
    ) -> None:
        """Initialize the Chroma store.

        Args:
            config: Chroma configuration.
            client: Optional client override for testing.
        """

        self._config = config
        self._config.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client: ClientAPI = client or self._create_client()
        self._collection: Collection | None = None

    @property
    def config(self) -> ChromaConfig:
        """Return the active Chroma configuration."""

        return self._config

    def get_or_create_collection(self):
        """Return the Chroma collection, creating it if missing."""

        if self._collection is None:
            metadata = {"hnsw:space": self._config.distance}
            self._collection = self._client.get_or_create_collection(
                name=self._config.collection_name,
                metadata=metadata,
            )
        return self._collection

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all vectors for a doc_id.

        Args:
            doc_id: Document identifier stored in Chroma metadata.
        Returns:
            Number of vectors deleted (best-effort count).
        """

        collection = self.get_or_create_collection()
        result = collection.get(where={"doc_id": doc_id}, include=[])
        ids = _flatten_ids(result.get("ids"))
        if not ids:
            return 0
        collection.delete(where={"doc_id": doc_id})
        return len(ids)

    def list_ids_by_doc_id(self, doc_id: str) -> list[str]:
        """List vector IDs stored for a doc_id.

        Args:
            doc_id: Document identifier stored in Chroma metadata.
        Returns:
            List of vector IDs for the doc_id.
        """

        collection = self.get_or_create_collection()
        result = collection.get(where={"doc_id": doc_id}, include=[])
        return _flatten_ids(result.get("ids"))

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        """Delete vectors by explicit IDs.

        Args:
            ids: Chroma IDs to delete.
        Returns:
            Number of vectors deleted (best-effort count).
        """

        if not ids:
            return 0
        collection = self.get_or_create_collection()
        collection.delete(ids=list(ids))
        return len(ids)

    def upsert_embeddings(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, object]],
    ) -> None:
        """Upsert embeddings into Chroma.

        Args:
            ids: Chroma IDs (chunk_uid values).
            embeddings: Embedding vectors.
            metadatas: Per-embedding metadata.
        Raises:
            ValueError: When input lengths do not match.
        """

        if len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError("ids, embeddings, and metadatas must be the same length")
        collection = self.get_or_create_collection()
        collection.upsert(
            ids=list(ids), embeddings=list(embeddings), metadatas=list(metadatas)
        )

    def count_by_doc_id(self, doc_id: str) -> int:
        """Count vectors in Chroma for a doc_id.

        Args:
            doc_id: Document identifier stored in Chroma metadata.
        Returns:
            Number of vectors matching the doc_id.
        """

        collection = self.get_or_create_collection()
        result = collection.get(where={"doc_id": doc_id}, include=[])
        ids = _flatten_ids(result.get("ids"))
        return len(ids)

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        top_k: int,
        where: dict[str, object] | None = None,
    ) -> tuple[list[str], list[float]]:
        """Query Chroma for nearest neighbors.

        Args:
            query_embeddings: Embedding vectors to query.
            top_k: Number of neighbors to return.
            where: Optional metadata filter.
        Returns:
            Tuple of (ids, distances) for the first query vector.
        Raises:
            ValueError: When top_k is not positive.
        """

        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        collection = self.get_or_create_collection()
        result = collection.query(
            query_embeddings=list(query_embeddings),
            n_results=top_k,
            where=where,
        )
        ids = _first_list(result.get("ids"))
        distances = _first_list(result.get("distances"))
        return ids, distances

    def _create_client(self) -> ClientAPI:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - requires optional dep
            raise ImportError(
                "chromadb is required for vector indexing. Install it via uv/pip."
            ) from exc
        return chromadb.PersistentClient(path=str(self._config.persist_dir))


def _first_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def _flatten_ids(value: object) -> list[str]:
    ids = _first_list(value)
    return [str(item) for item in ids]
