"""Chunk identifier helpers."""

from __future__ import annotations

import hashlib


def compute_chunk_uid(
    *,
    doc_id: str,
    page_number: int,
    chunk_index: int,
    char_start: int | None,
    char_end: int | None,
) -> str:
    """Compute a stable chunk UID for cross-index joins.

    Args:
        doc_id: SHA1 of the PDF bytes.
        page_number: 1-based page number.
        chunk_index: 0-based chunk index within the page.
        char_start: Character offset for the chunk start, if available.
        char_end: Character offset for the chunk end, if available.
    Returns:
        SHA1 hash string identifying the chunk.
    Edge cases:
        Uses -1 for missing char offsets to keep the ID deterministic.
    """

    safe_char_start = char_start if char_start is not None else -1
    safe_char_end = char_end if char_end is not None else -1
    raw = f"{doc_id}:{page_number}:{chunk_index}:{safe_char_start}:{safe_char_end}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
