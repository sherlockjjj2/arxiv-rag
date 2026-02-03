"""Utilities for normalizing arXiv identifiers."""

from __future__ import annotations

import re
from typing import Iterable

_NEW_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_OLD_ID_RE = re.compile(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}$", re.IGNORECASE)
_VERSIONED_NEW_ID_RE = re.compile(r"^(?P<base>\d{4}\.\d{4,5})v\d+$")
_VERSIONED_OLD_ID_RE = re.compile(
    r"^(?P<base>[a-z-]+(\.[A-Z]{2})?/\d{7})v\d+$",
    re.IGNORECASE,
)


def base_id_from_versioned(versioned_id: str) -> str:
    """Strip version suffixes from arXiv IDs.

    Args:
        versioned_id: ID that may include a version suffix.
    Returns:
        Base arXiv ID without version suffix.
    Edge cases:
        Returns the input unchanged when no version suffix is present.
    """

    if match := _VERSIONED_NEW_ID_RE.match(versioned_id):
        return match.group("base")
    if match := _VERSIONED_OLD_ID_RE.match(versioned_id):
        return match.group("base")
    return versioned_id


def is_valid_base_id(base_id: str) -> bool:
    """Check whether a base arXiv ID matches known patterns.

    Args:
        base_id: Base arXiv ID to validate.
    Returns:
        True if the ID matches a valid format, otherwise False.
    """

    return bool(_NEW_ID_RE.match(base_id) or _OLD_ID_RE.match(base_id))


def validate_base_ids(ids: Iterable[str]) -> list[str]:
    """Collect invalid base IDs from an iterable.

    Args:
        ids: Iterable of base IDs to validate.
    Returns:
        List of invalid base IDs.
    Edge cases:
        Returns an empty list when all IDs are valid.
    """

    return [base_id for base_id in ids if not is_valid_base_id(base_id)]


def normalize_base_id_for_lookup(base_id: str) -> str:
    """Normalize an arXiv base ID for case-insensitive matching.

    Args:
        base_id: Base arXiv ID to normalize.
    Returns:
        Normalized identifier for lookup comparisons.
    """

    return base_id.lower()
