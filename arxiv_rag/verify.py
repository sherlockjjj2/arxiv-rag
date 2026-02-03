"""Verification utilities for citation checking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from arxiv_rag.arxiv_ids import base_id_from_versioned, is_valid_base_id
from arxiv_rag.db import load_page_numbers_by_paper, load_paper_ids

_ARXIV_ID_PATTERN = (
    r"(?P<paper_id>"
    r"\d{4}\.\d{4,5}(?:v\d+)?"
    r"|"
    r"[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
    r")"
)

CITATION_PATTERN = re.compile(
    rf"\[arXiv:{_ARXIV_ID_PATTERN} p\.(?P<page_number>\d+)\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationRecord:
    """Structured citation parsed from an answer.

    Args:
        citation_id: Stable identifier for the citation in the answer.
        paper_id: arXiv paper identifier (base or versioned).
        page_number: 1-based page number referenced in the citation.
        quote: Supporting quote snippet (verbatim).
        chunk_uid: Optional chunk UID tied to the citation.
        quote_start: Optional character offset for the quote within the page text.
        quote_end: Optional character offset for the quote within the page text.
    """

    citation_id: str
    paper_id: str
    page_number: int
    quote: str | None = None
    chunk_uid: str | None = None
    quote_start: int | None = None
    quote_end: int | None = None


@dataclass(frozen=True)
class SentenceCheck:
    """Verification metadata for a single sentence.

    Args:
        sentence_index: 0-based sentence index within the answer.
        text: Normalized sentence text.
        citations: List of citation IDs associated with this sentence.
        judge: Optional LLM-judge payload for semantic verification.
    """

    sentence_index: int
    text: str
    citations: list[str]
    judge: dict[str, Any] | None = None


@dataclass(frozen=True)
class VerifyError:
    """Deterministic verification error.

    Args:
        code: Machine-readable error code.
        message: Human-readable error message.
        citation_id: Optional citation ID associated with the error.
        sentence_index: Optional sentence index associated with the error.
    """

    code: str
    message: str
    citation_id: str | None = None
    sentence_index: int | None = None


@dataclass(frozen=True)
class VerifyReport:
    """Verification report for a generated answer.

    Args:
        status: "pass", "fail", or "error".
        errors: List of deterministic validation errors.
        sentences: Per-sentence verification metadata.
        citations: Parsed citation records.
    """

    status: Literal["pass", "fail", "error"]
    errors: list[VerifyError]
    sentences: list[SentenceCheck]
    citations: list[CitationRecord]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for stable matching.

    Args:
        text: Input text to normalize.
    Returns:
        Text with collapsed whitespace and trimmed edges.
    Edge cases:
        Returns an empty string when text contains only whitespace.
    """

    if not text:
        return ""
    cleaned = text.replace("\u00a0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using a lightweight heuristic.

    Args:
        text: Raw answer text.
    Returns:
        List of normalized sentences in display order.
    Edge cases:
        Returns an empty list when no sentence-like content is present.
    """

    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", normalized)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences


def parse_citations(answer: str) -> list[CitationRecord]:
    """Parse citations from an answer string.

    Args:
        answer: Generated answer text containing citations.
    Returns:
        Ordered list of parsed citations with normalized paper IDs.
    Raises:
        ValueError: If malformed citations are found or IDs are invalid.
    Edge cases:
        Returns an empty list when no citations are present.
    """

    malformed = find_malformed_citations(answer)
    if malformed:
        joined = ", ".join(malformed)
        raise ValueError(f"Malformed citations found: {joined}")

    citations: list[CitationRecord] = []
    for index, match in enumerate(CITATION_PATTERN.finditer(answer), start=1):
        raw_paper_id = match.group("paper_id")
        paper_id = base_id_from_versioned(raw_paper_id)
        if not is_valid_base_id(paper_id):
            raise ValueError(f"Invalid arXiv ID: {paper_id}")
        page_number = int(match.group("page_number"))
        if page_number < 1:
            raise ValueError(f"Invalid page number: {page_number}")
        citations.append(
            CitationRecord(
                citation_id=f"c{index}",
                paper_id=paper_id,
                page_number=page_number,
            )
        )
    return citations


def find_malformed_citations(answer: str) -> list[str]:
    """Find citation-like substrings that do not match the required format.

    Args:
        answer: Generated answer text.
    Returns:
        List of malformed citation substrings in appearance order.
    Edge cases:
        Returns an empty list when no malformed citations are detected.
    """

    malformed: list[str] = []

    for match in re.finditer(r"\[arXiv:[^\]]*\]", answer, flags=re.IGNORECASE):
        if not CITATION_PATTERN.fullmatch(match.group(0)):
            malformed.append(match.group(0))

    if re.search(r"\[\[arXiv:", answer, flags=re.IGNORECASE):
        malformed.append("[[arXiv:")
    if re.search(r"arXiv:[^\]]*\]\]", answer, flags=re.IGNORECASE):
        malformed.append("arXiv:...]]")

    return malformed


def validate_citations(
    citations: Iterable[CitationRecord],
    known_paper_ids: Iterable[str] | None = None,
    max_pages_by_paper: Mapping[str, int] | None = None,
) -> list[VerifyError]:
    """Validate parsed citations against known IDs and page ranges.

    Args:
        citations: Parsed citations to validate.
        known_paper_ids: Optional iterable of known arXiv IDs.
        max_pages_by_paper: Optional mapping of paper_id to max page count.
    Returns:
        List of validation errors (empty when all citations are valid).
    Edge cases:
        When both validation inputs are omitted, only format validation occurs.
    """

    errors: list[VerifyError] = []
    known_ids = set(known_paper_ids) if known_paper_ids is not None else None

    for citation in citations:
        if not is_valid_base_id(citation.paper_id):
            errors.append(
                VerifyError(
                    code="invalid_paper_id",
                    message=f"Invalid arXiv ID format: {citation.paper_id}",
                    citation_id=citation.citation_id,
                )
            )
            continue

        if known_ids is not None and citation.paper_id not in known_ids:
            errors.append(
                VerifyError(
                    code="unknown_paper_id",
                    message=f"Unknown arXiv ID: {citation.paper_id}",
                    citation_id=citation.citation_id,
                )
            )

        if citation.page_number < 1:
            errors.append(
                VerifyError(
                    code="page_out_of_range",
                    message=(
                        f"Page {citation.page_number} out of range for "
                        f"{citation.paper_id} (page numbers start at 1)"
                    ),
                    citation_id=citation.citation_id,
                )
            )

        if max_pages_by_paper is not None:
            max_pages = max_pages_by_paper.get(citation.paper_id)
            if max_pages is None:
                errors.append(
                    VerifyError(
                        code="missing_page_count",
                        message=f"Missing page count for: {citation.paper_id}",
                        citation_id=citation.citation_id,
                    )
                )
            elif citation.page_number < 1 or citation.page_number > max_pages:
                errors.append(
                    VerifyError(
                        code="page_out_of_range",
                        message=(
                            f"Page {citation.page_number} out of range for "
                            f"{citation.paper_id} (1-{max_pages})"
                        ),
                        citation_id=citation.citation_id,
                    )
                )

    return errors


def validate_citations_in_db(
    citations: Iterable[CitationRecord],
    *,
    db_path: Path,
) -> None:
    """Validate citations against the database for known IDs and pages.

    Args:
        citations: Parsed citation records to validate.
        db_path: SQLite database path.
    Raises:
        FileNotFoundError: If the database path does not exist.
        ValueError: If a citation references an unknown paper or missing page.
        sqlite3.Error: If database queries fail.
    """

    citations_list = list(citations)
    if not citations_list:
        return

    paper_ids = {citation.paper_id for citation in citations_list}
    known_ids = load_paper_ids(db_path)
    for citation in citations_list:
        if citation.paper_id not in known_ids:
            raise ValueError(f"Unknown arXiv ID: {citation.paper_id}")

    pages_by_paper = load_page_numbers_by_paper(db_path, sorted(paper_ids))
    for citation in citations_list:
        pages = pages_by_paper.get(citation.paper_id, set())
        if citation.page_number not in pages:
            raise ValueError(
                "Missing page "
                f"{citation.page_number} for {citation.paper_id}"
            )


def parse_and_validate_citations(answer: str, *, db_path: Path) -> list[CitationRecord]:
    """Parse citations and validate against database records.

    Args:
        answer: Generated answer text containing citations.
        db_path: SQLite database path.
    Returns:
        Parsed citation records if validation succeeds.
    Raises:
        ValueError: If citations are malformed or unresolved.
    """

    citations = parse_citations(answer)
    validate_citations_in_db(citations, db_path=db_path)
    return citations


def parse_citation_quotes(answer: str) -> dict[str, str]:
    """Extract supporting quotes for each citation occurrence.

    Expected format: [arXiv:ID p.N] *"quote"*

    Args:
        answer: Generated answer text.
    Returns:
        Mapping from citation_id (c1, c2, ...) to extracted quote text.
    Raises:
        ValueError: If a citation is missing a following quote snippet.
    Edge cases:
        Quotes are required to appear after the citation; earlier quotes are ignored.
    """

    quotes: dict[str, str] = {}
    matches = list(CITATION_PATTERN.finditer(answer))
    for index, match in enumerate(matches, start=1):
        window_end = matches[index].start() if index < len(matches) else len(answer)
        after = answer[match.end() : window_end]
        quote_match = re.search(r'\*"(.*?)"\*', after, flags=re.DOTALL)
        if not quote_match:
            raise ValueError(f"Missing quote after citation c{index}")
        quote = normalize_whitespace(quote_match.group(1))
        quotes[f"c{index}"] = quote
    return quotes


def attach_quotes(
    citations: Iterable[CitationRecord],
    quotes_by_id: Mapping[str, str],
) -> list[CitationRecord]:
    """Attach quote snippets to citation records.

    Args:
        citations: Parsed citations.
        quotes_by_id: Mapping of citation IDs to quote strings.
    Returns:
        New citation records with quote fields populated when available.
    Raises:
        ValueError: If a citation is missing a quote entry.
    """

    updated: list[CitationRecord] = []
    for citation in citations:
        quote = quotes_by_id.get(citation.citation_id)
        if quote is None:
            raise ValueError(f"Missing quote for {citation.citation_id}")
        updated.append(
            CitationRecord(
                citation_id=citation.citation_id,
                paper_id=citation.paper_id,
                page_number=citation.page_number,
                quote=quote,
                chunk_uid=citation.chunk_uid,
                quote_start=citation.quote_start,
                quote_end=citation.quote_end,
            )
        )
    return updated


def match_quote_in_text(
    quote: str,
    page_text: str,
) -> tuple[int, int] | None:
    """Find a quote in page text using normalized whitespace matching.

    Args:
        quote: Quote snippet to locate.
        page_text: Full page text content.
    Returns:
        Tuple of (start, end) offsets in normalized page text, or None if not found.
    Edge cases:
        Returns None when either input is empty after normalization.
    """

    normalized_quote = normalize_whitespace(quote)
    normalized_text = normalize_whitespace(page_text)
    if not normalized_quote or not normalized_text:
        return None

    start = normalized_text.find(normalized_quote)
    if start == -1:
        return None
    end = start + len(normalized_quote)
    return start, end
