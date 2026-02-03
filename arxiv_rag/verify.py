"""Verification utilities for citation checking."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from arxiv_rag.arxiv_ids import (
    base_id_from_versioned,
    is_valid_base_id,
    normalize_base_id_for_lookup,
)
from arxiv_rag.db import (
    load_page_numbers_by_paper,
    load_paper_ids,
    load_paper_pdf_paths,
)

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
        paragraph_index: Optional paragraph index associated with the error.
    """

    code: str
    message: str
    citation_id: str | None = None
    sentence_index: int | None = None
    paragraph_index: int | None = None


@dataclass(frozen=True)
class VerifyReport:
    """Verification report for a generated answer.

    Args:
        status: "pass", "fail", or "error".
        errors: List of deterministic validation errors.
        sentences: Per-sentence verification metadata.
        paragraphs: Per-paragraph verification metadata.
        citations: Parsed citation records.
    """

    status: Literal["pass", "fail", "error"]
    errors: list[VerifyError]
    sentences: list[SentenceCheck]
    paragraphs: list["ParagraphCheck"]
    citations: list[CitationRecord]


@dataclass(frozen=True)
class ParagraphCheck:
    """Verification metadata for a single paragraph.

    Args:
        paragraph_index: 0-based paragraph index within the answer.
        text: Paragraph text.
        citations: List of citation IDs associated with this paragraph.
    """

    paragraph_index: int
    text: str
    citations: list[str]


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
    known_ids = (
        {normalize_base_id_for_lookup(paper_id) for paper_id in known_paper_ids}
        if known_paper_ids is not None
        else None
    )
    max_pages_norm = (
        {
            normalize_base_id_for_lookup(paper_id): max_pages
            for paper_id, max_pages in max_pages_by_paper.items()
        }
        if max_pages_by_paper is not None
        else None
    )

    for citation in citations:
        normalized_id = normalize_base_id_for_lookup(citation.paper_id)
        if not is_valid_base_id(citation.paper_id):
            errors.append(
                VerifyError(
                    code="invalid_paper_id",
                    message=f"Invalid arXiv ID format: {citation.paper_id}",
                    citation_id=citation.citation_id,
                )
            )
            continue

        if known_ids is not None and normalized_id not in known_ids:
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

        if max_pages_norm is not None:
            max_pages = max_pages_norm.get(normalized_id)
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

    known_ids = load_paper_ids(db_path)
    known_ids_by_normalized = {
        normalize_base_id_for_lookup(paper_id): paper_id for paper_id in known_ids
    }
    resolved_paper_ids: set[str] = set()
    for citation in citations_list:
        normalized_id = normalize_base_id_for_lookup(citation.paper_id)
        stored_id = known_ids_by_normalized.get(normalized_id)
        if stored_id is None:
            raise ValueError(f"Unknown arXiv ID: {citation.paper_id}")
        resolved_paper_ids.add(stored_id)

    pages_by_paper = load_page_numbers_by_paper(db_path, sorted(resolved_paper_ids))
    pages_by_paper_normalized = {
        normalize_base_id_for_lookup(paper_id): pages
        for paper_id, pages in pages_by_paper.items()
    }
    for citation in citations_list:
        normalized_id = normalize_base_id_for_lookup(citation.paper_id)
        pages = pages_by_paper_normalized.get(normalized_id, set())
        if citation.page_number not in pages:
            raise ValueError(
                f"Missing page {citation.page_number} for {citation.paper_id}"
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
        quote_match = re.match(r'\s*\*"(.*?)"\*', after, flags=re.DOTALL)
        if not quote_match:
            raise ValueError(f"Missing adjacent quote after citation c{index}")
        quote = normalize_whitespace(quote_match.group(1))
        quotes[f"c{index}"] = quote
    return quotes


def parse_citation_quotes_with_errors(
    answer: str,
) -> tuple[dict[str, str], list[VerifyError]]:
    """Extract supporting quotes for each citation occurrence without failing fast.

    Expected format: [arXiv:ID p.N] *"quote"*

    Args:
        answer: Generated answer text.
    Returns:
        Mapping from citation_id (c1, c2, ...) to extracted quote text plus errors.
    Edge cases:
        Missing quotes are reported as errors but other quotes are preserved.
    """

    quotes: dict[str, str] = {}
    errors: list[VerifyError] = []
    matches = list(CITATION_PATTERN.finditer(answer))
    for index, match in enumerate(matches, start=1):
        window_end = matches[index].start() if index < len(matches) else len(answer)
        after = answer[match.end() : window_end]
        quote_match = re.match(r'\s*\*"(.*?)"\*', after, flags=re.DOTALL)
        if not quote_match:
            errors.append(
                VerifyError(
                    code="missing_quote",
                    message=f"Missing adjacent quote after citation c{index}",
                    citation_id=f"c{index}",
                )
            )
            continue
        quote = normalize_whitespace(quote_match.group(1))
        quotes[f"c{index}"] = quote
    return quotes, errors


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


def _resolve_parsed_json_path(
    parsed_dir: Path,
    pdf_path: Path | None,
    paper_id: str,
) -> Path:
    """Resolve the parsed JSON path for a paper.

    Args:
        parsed_dir: Directory containing parsed JSON files.
        pdf_path: Optional PDF path from the papers table.
        paper_id: Base arXiv ID.
    Returns:
        Path to the parsed JSON file.
    """

    if pdf_path is not None:
        return parsed_dir / f"{pdf_path.stem}.json"
    safe_id = paper_id.replace("/", "_")
    return parsed_dir / f"{safe_id}.json"


def _load_page_texts(parsed_path: Path) -> dict[int, str]:
    """Load page text from a parsed JSON file.

    Args:
        parsed_path: Path to the parsed JSON file.
    Returns:
        Mapping of page number to page text.
    Raises:
        ValueError: If the parsed JSON is missing required fields.
        json.JSONDecodeError: If the JSON cannot be decoded.
        OSError: If the file cannot be read.
    """

    payload = json.loads(parsed_path.read_text(encoding="utf-8"))
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"Parsed JSON missing pages: {parsed_path}")
    page_texts: dict[int, str] = {}
    for page in pages:
        page_number = page.get("page")
        if page_number is None:
            continue
        text = page.get("text", "")
        page_texts[int(page_number)] = text
    return page_texts


def load_page_texts_for_citations(
    citations: Iterable[CitationRecord],
    *,
    db_path: Path,
    parsed_dir: Path = Path("data/parsed"),
) -> tuple[dict[tuple[str, int], str], list[VerifyError]]:
    """Load page text for each citation's paper and page number.

    Args:
        citations: Citation records to resolve.
        db_path: SQLite database path.
        parsed_dir: Directory containing parsed JSON files.
    Returns:
        Mapping from (paper_id, page_number) to page text plus any errors.
    Raises:
        FileNotFoundError: If the database path does not exist.
    """

    citations_list = list(citations)
    if not citations_list:
        return {}, []

    parsed_dir = parsed_dir or Path("data/parsed")
    paper_ids = sorted({citation.paper_id for citation in citations_list})
    pdf_paths = load_paper_pdf_paths(db_path, paper_ids)
    parsed_cache: dict[Path, dict[int, str]] = {}
    page_texts: dict[tuple[str, int], str] = {}
    errors: list[VerifyError] = []

    for citation in citations_list:
        normalized_id = normalize_base_id_for_lookup(citation.paper_id)
        pdf_path = pdf_paths.get(normalized_id)
        parsed_path = _resolve_parsed_json_path(parsed_dir, pdf_path, citation.paper_id)

        if not parsed_path.exists():
            errors.append(
                VerifyError(
                    code="missing_parsed_file",
                    message=f"Parsed JSON not found: {parsed_path}",
                    citation_id=citation.citation_id,
                )
            )
            continue

        if parsed_path not in parsed_cache:
            try:
                parsed_cache[parsed_path] = _load_page_texts(parsed_path)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                errors.append(
                    VerifyError(
                        code="invalid_parsed_json",
                        message=str(exc),
                        citation_id=citation.citation_id,
                    )
                )
                continue

        page_text = parsed_cache[parsed_path].get(citation.page_number)
        if page_text is None:
            errors.append(
                VerifyError(
                    code="missing_page_text",
                    message=(
                        "Parsed JSON missing page "
                        f"{citation.page_number} for {citation.paper_id}"
                    ),
                    citation_id=citation.citation_id,
                )
            )
            continue

        page_texts[(citation.paper_id, citation.page_number)] = page_text

    return page_texts, errors


def locate_quotes_in_pages(
    citations: Iterable[CitationRecord],
    page_texts: Mapping[tuple[str, int], str],
    *,
    emit_missing_quote: bool = True,
) -> tuple[list[CitationRecord], list[VerifyError]]:
    """Locate each citation quote within the corresponding page text.

    Args:
        citations: Citation records with quote snippets.
        page_texts: Mapping from (paper_id, page_number) to page text.
        emit_missing_quote: Whether to emit errors for missing quote snippets.
    Returns:
        Updated citations with quote offsets plus any errors.
    """

    updated: list[CitationRecord] = []
    errors: list[VerifyError] = []

    for citation in citations:
        quote = citation.quote
        if not quote:
            if emit_missing_quote:
                errors.append(
                    VerifyError(
                        code="missing_quote",
                        message=f"Missing quote for {citation.citation_id}",
                        citation_id=citation.citation_id,
                    )
                )
            updated.append(citation)
            continue

        page_text = page_texts.get((citation.paper_id, citation.page_number))
        if page_text is None:
            errors.append(
                VerifyError(
                    code="missing_page_text",
                    message=(
                        "Missing page text for "
                        f"{citation.paper_id} p.{citation.page_number}"
                    ),
                    citation_id=citation.citation_id,
                )
            )
            updated.append(citation)
            continue

        match = match_quote_in_text(quote, page_text)
        if match is None:
            errors.append(
                VerifyError(
                    code="quote_not_found",
                    message=(
                        "Quote not found for "
                        f"{citation.paper_id} p.{citation.page_number}"
                    ),
                    citation_id=citation.citation_id,
                )
            )
            updated.append(citation)
            continue

        quote_start, quote_end = match
        updated.append(
            CitationRecord(
                citation_id=citation.citation_id,
                paper_id=citation.paper_id,
                page_number=citation.page_number,
                quote=citation.quote,
                chunk_uid=citation.chunk_uid,
                quote_start=quote_start,
                quote_end=quote_end,
            )
        )

    return updated, errors


def _split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Split normalized text into sentences with offsets.

    Args:
        text: Raw answer text.
    Returns:
        List of tuples containing (sentence, start_offset, end_offset).
    """

    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    splits: list[tuple[str, int, int]] = []
    boundaries: list[tuple[int, int]] = []
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char in ".!?":
            end = index + 1
            while end < len(normalized) and normalized[end] in {'"', "'", "*"}:
                end += 1
            if end < len(normalized) and normalized[end].isspace():
                next_start = end
                while next_start < len(normalized) and normalized[next_start].isspace():
                    next_start += 1
                if next_start < len(normalized) and (
                    normalized[next_start].isupper()
                    or normalized[next_start].isdigit()
                    or normalized[next_start] == "["
                ):
                    boundaries.append((end, next_start))
                    index = next_start
                    continue
        index += 1

    start = 0
    for end, next_start in boundaries:
        segment = normalized[start:end]
        trimmed = segment.strip()
        if trimmed:
            leading = len(segment) - len(segment.lstrip())
            trailing = len(segment) - len(segment.rstrip())
            span_start = start + leading
            span_end = end - trailing
            splits.append((trimmed, span_start, span_end))
        start = next_start

    segment = normalized[start:]
    trimmed = segment.strip()
    if trimmed:
        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment) - len(segment.rstrip())
        span_start = start + leading
        span_end = len(normalized) - trailing
        splits.append((trimmed, span_start, span_end))

    return splits


def map_sentence_citations(answer: str) -> list[SentenceCheck]:
    """Map citations to their containing sentences.

    Args:
        answer: Generated answer text.
    Returns:
        Sentence checks with associated citation IDs.
    """

    normalized = normalize_whitespace(answer)
    if not normalized:
        return []

    citation_positions = [
        (f"c{index}", match.start(), match.end())
        for index, match in enumerate(CITATION_PATTERN.finditer(normalized), start=1)
    ]

    sentences: list[SentenceCheck] = []
    for sentence_index, (sentence, start, end) in enumerate(
        _split_sentences_with_spans(answer)
    ):
        citation_ids = [
            citation_id
            for citation_id, cite_start, cite_end in citation_positions
            if cite_start >= start and cite_end <= end
        ]
        sentences.append(
            SentenceCheck(
                sentence_index=sentence_index,
                text=sentence,
                citations=citation_ids,
            )
        )
    return sentences


def _split_paragraphs_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Split raw text into paragraphs with offsets.

    Args:
        text: Raw answer text.
    Returns:
        List of tuples containing (paragraph, start_offset, end_offset).
    """

    if not text.strip():
        return []

    splits: list[tuple[str, int, int]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n+", text):
        end = match.start()
        segment = text[start:end]
        trimmed = normalize_whitespace(segment)
        if trimmed:
            splits.append((trimmed, start, end))
        start = match.end()

    segment = text[start:]
    trimmed = normalize_whitespace(segment)
    if trimmed:
        splits.append((trimmed, start, len(text)))

    return splits


def map_paragraph_citations(answer: str) -> list[ParagraphCheck]:
    """Map citations to their containing paragraphs.

    Args:
        answer: Generated answer text.
    Returns:
        Paragraph checks with associated citation IDs.
    """

    if not answer.strip():
        return []

    citation_positions = [
        (f"c{index}", match.start(), match.end())
        for index, match in enumerate(CITATION_PATTERN.finditer(answer), start=1)
    ]

    paragraphs: list[ParagraphCheck] = []
    for paragraph_index, (paragraph, start, end) in enumerate(
        _split_paragraphs_with_spans(answer)
    ):
        citation_ids = [
            citation_id
            for citation_id, cite_start, cite_end in citation_positions
            if cite_start >= start and cite_end <= end
        ]
        paragraphs.append(
            ParagraphCheck(
                paragraph_index=paragraph_index,
                text=paragraph,
                citations=citation_ids,
            )
        )
    return paragraphs


def validate_paragraph_coverage(
    paragraphs: Iterable[ParagraphCheck],
) -> list[VerifyError]:
    """Validate that paragraphs contain citations.

    Args:
        paragraphs: Paragraph checks to validate.
    Returns:
        List of coverage errors.
    """

    errors: list[VerifyError] = []
    for paragraph in paragraphs:
        if paragraph.citations:
            continue
        errors.append(
            VerifyError(
                code="missing_paragraph_citation",
                message="Missing citation for paragraph.",
                paragraph_index=paragraph.paragraph_index,
            )
        )
    return errors


def extract_quotes(
    answer: str,
    *,
    db_path: Path,
    parsed_dir: Path = Path("data/parsed"),
) -> list[CitationRecord]:
    """Extract and locate quotes for each citation in an answer.

    Args:
        answer: Generated answer text.
        db_path: SQLite database path.
        parsed_dir: Directory containing parsed JSON files.
    Returns:
        Citation records populated with quote and offsets.
    Raises:
        ValueError: If citations or quotes are malformed.
        FileNotFoundError: If the database path does not exist.
    """

    citations = parse_citations(answer)
    validate_citations_in_db(citations, db_path=db_path)
    quotes_by_id = parse_citation_quotes(answer)
    citations_with_quotes = attach_quotes(citations, quotes_by_id)
    page_texts, errors = load_page_texts_for_citations(
        citations_with_quotes,
        db_path=db_path,
        parsed_dir=parsed_dir,
    )
    if errors:
        first_error = errors[0]
        raise ValueError(first_error.message)
    located, quote_errors = locate_quotes_in_pages(citations_with_quotes, page_texts)
    if quote_errors:
        first_error = quote_errors[0]
        raise ValueError(first_error.message)
    return located


def verify_answer(
    answer: str,
    *,
    db_path: Path,
    parsed_dir: Path = Path("data/parsed"),
) -> VerifyReport:
    """Run deterministic verification on a generated answer.

    Args:
        answer: Generated answer text.
        db_path: SQLite database path.
        parsed_dir: Directory containing parsed JSON files.
    Returns:
        Verification report with status, errors, and parsed citations.
    Raises:
        FileNotFoundError: If the database path does not exist.
    """

    errors: list[VerifyError] = []
    try:
        citations = parse_citations(answer)
    except ValueError as exc:
        return VerifyReport(
            status="fail",
            errors=[
                VerifyError(
                    code="malformed_citation",
                    message=str(exc),
                )
            ],
            sentences=[],
            paragraphs=[],
            citations=[],
        )

    try:
        validate_citations_in_db(citations, db_path=db_path)
    except ValueError as exc:
        return VerifyReport(
            status="fail",
            errors=[
                VerifyError(
                    code="unresolved_citation",
                    message=str(exc),
                )
            ],
            sentences=[],
            paragraphs=[],
            citations=citations,
        )

    try:
        quotes_by_id, quote_errors = parse_citation_quotes_with_errors(answer)
        errors.extend(quote_errors)
        updated: list[CitationRecord] = []
        for citation in citations:
            updated.append(
                CitationRecord(
                    citation_id=citation.citation_id,
                    paper_id=citation.paper_id,
                    page_number=citation.page_number,
                    quote=quotes_by_id.get(citation.citation_id),
                    chunk_uid=citation.chunk_uid,
                    quote_start=citation.quote_start,
                    quote_end=citation.quote_end,
                )
            )
        citations = updated
    except ValueError as exc:
        errors.append(
            VerifyError(
                code="missing_quote",
                message=str(exc),
            )
        )

    page_texts, page_errors = load_page_texts_for_citations(
        citations,
        db_path=db_path,
        parsed_dir=parsed_dir,
    )
    errors.extend(page_errors)

    located, quote_errors = locate_quotes_in_pages(
        citations,
        page_texts,
        emit_missing_quote=False,
    )
    errors.extend(quote_errors)

    sentences = map_sentence_citations(answer)
    paragraphs = map_paragraph_citations(answer)
    errors.extend(validate_paragraph_coverage(paragraphs))

    status = "pass" if not errors else "fail"
    return VerifyReport(
        status=status,
        errors=errors,
        sentences=sentences,
        paragraphs=paragraphs,
        citations=located,
    )
