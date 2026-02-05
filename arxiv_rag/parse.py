from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import fitz

LOGGER = logging.getLogger(__name__)
_SHA1_CHUNK_SIZE = 1024 * 1024
_LOW_RISK_MUPDF_WARNING_PATTERNS = (
    "bogus font ascent/descent values",
    "invalid marked content and clip nesting",
    "could not parse color space",
    "ignoring page blending colorspace",
)


def _compute_sha1(pdf_path: Path) -> str:
    """Return the SHA1 hash of the PDF file contents."""
    sha1 = hashlib.sha1()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA1_CHUNK_SIZE), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def _clean_text(text: str) -> str:
    """Normalize extracted PDF text for downstream parsing."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("-\n", "")
    lines = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"[ \t]+", " ", line)
        lines.append(line)
    return "\n".join(lines)


def _find_repeated_headers_footers(pages: list[dict]) -> set[str]:
    """Return repeated header/footer lines based on frequency thresholds."""
    total_pages = len(pages)
    if total_pages == 0:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = page["text"].split("\n") if page["text"] else []
        if not lines:
            continue
        candidates = {lines[0], lines[-1]}
        for line in candidates:
            if line and len(line) <= 120:
                counts[line] += 1

    threshold = math.ceil(0.6 * total_pages)
    return {line for line, count in counts.items() if count >= threshold}


def _remove_repeated_lines(pages: list[dict], repeated_lines: set[str]) -> int:
    """Remove repeated header/footer lines and return number of removals."""
    if not repeated_lines:
        return 0

    removed = 0
    for page in pages:
        lines = page["text"].split("\n") if page["text"] else []
        filtered = [line for line in lines if line not in repeated_lines]
        removed += len(lines) - len(filtered)
        page["text"] = "\n".join(filtered)
    return removed


@contextmanager
def _mute_stderr_fd() -> Iterator[None]:
    """Temporarily redirect process-level stderr to os.devnull.

    Some MuPDF parser diagnostics bypass Python logging and write directly to
    file descriptor 2. Redirecting stderr at the descriptor level suppresses
    those low-level messages while parsing.
    """

    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd = os.dup(2)
    except OSError:
        yield
        return

    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            os.dup2(saved_stderr_fd, 2)
        finally:
            os.close(saved_stderr_fd)
            os.close(devnull_fd)


@contextmanager
def _mute_mupdf_diagnostics() -> Iterator[None]:
    """Temporarily silence MuPDF stderr diagnostics during extraction.

    PyMuPDF can emit parser diagnostics directly to stderr for malformed-but-readable
    PDFs. This context manager keeps CLI progress output readable while preserving
    explicit Python-level exceptions and warnings.
    """

    previous_error_display = bool(fitz.TOOLS.mupdf_display_errors())
    previous_warning_display = bool(fitz.TOOLS.mupdf_display_warnings())
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    fitz.TOOLS.reset_mupdf_warnings()
    try:
        with _mute_stderr_fd():
            yield
    finally:
        fitz.TOOLS.mupdf_display_errors(previous_error_display)
        fitz.TOOLS.mupdf_display_warnings(previous_warning_display)


def _summarize_mupdf_warnings() -> str | None:
    """Return a compact summary of high-risk MuPDF warnings, if any."""

    warning_blob = fitz.TOOLS.mupdf_warnings()
    if not warning_blob:
        return None

    lines = [line.strip() for line in warning_blob.splitlines() if line.strip()]
    if not lines:
        return None

    filtered_lines = [line for line in lines if not _is_low_risk_mupdf_warning(line)]
    if not filtered_lines:
        return None

    preview_count = min(3, len(filtered_lines))
    preview = "; ".join(filtered_lines[:preview_count])
    if len(filtered_lines) == preview_count:
        return f"MuPDF warnings: {preview}"
    return f"MuPDF warnings: {preview} (+{len(filtered_lines) - preview_count} more)"


def _is_low_risk_mupdf_warning(line: str) -> bool:
    """Return True when a MuPDF warning line matches low-risk parser noise.

    Low-risk warnings are typically font metadata or graphics-structure issues
    that do not indicate severe text extraction failures.
    """

    normalized = line.strip().casefold()
    return any(pattern in normalized for pattern in _LOW_RISK_MUPDF_WARNING_PATTERNS)


def _parse_pdf_with_warnings(
    pdf_path: Path, *, remove_headers_footers: bool
) -> tuple[dict, list[str]]:
    """Parse a PDF and collect non-fatal extraction warnings.

    Args:
        pdf_path: Path to the source PDF file.
        remove_headers_footers: Whether to remove repeated page boundary lines.
    Returns:
        Tuple of parsed payload and warning messages.
    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If the PDF cannot be opened or read.
    Edge cases:
        Pages that fail extraction are skipped with warnings.
    """

    warnings: list[str] = []

    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc_id = _compute_sha1(pdf_path)

    try:
        with _mute_mupdf_diagnostics():
            with fitz.open(pdf_path) as doc:
                num_pages = doc.page_count
                pages: list[dict] = []
                for page_index in range(num_pages):
                    page_number = page_index + 1
                    try:
                        page = doc.load_page(page_index)
                        text = page.get_text("text")
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - PyMuPDF error types vary
                        warning = f"Skipped page {page_number}: {exc}"
                        warnings.append(warning)
                        LOGGER.warning(warning)
                        continue
                    cleaned = _clean_text(text)
                    pages.append({"page": page_number, "text": cleaned})
    except Exception as exc:  # pragma: no cover - PyMuPDF error types vary
        raise ValueError(f"Failed to read PDF: {pdf_path}") from exc

    if warning_summary := _summarize_mupdf_warnings():
        warnings.append(warning_summary)

    if remove_headers_footers and pages:
        if repeated := _find_repeated_headers_footers(pages):
            removed = _remove_repeated_lines(pages, repeated)
            warnings.append(
                "Removed repeated headers/footers: "
                f"{len(repeated)} lines across {removed} occurrences."
            )

    result = {
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "num_pages": num_pages,
        "pages": pages,
    }
    return result, warnings


def parse_pdf(pdf_path: str, *, remove_headers_footers: bool = False) -> dict:
    """Parse an arXiv-style PDF into per-page text.

    Args:
        pdf_path: Path to a PDF file on disk.
        remove_headers_footers: Remove repeated header/footer lines when True.

    Returns:
        Parsed PDF metadata with per-page text (doc_id, pdf_path, num_pages, pages).

    Raises:
        FileNotFoundError: When the PDF file does not exist.
        ValueError: When the PDF cannot be opened or read.

    Edge cases:
        Pages that fail to extract are skipped; the pages list can be shorter than
        num_pages.
    """
    result, _warnings = _parse_pdf_with_warnings(
        Path(pdf_path), remove_headers_footers=remove_headers_footers
    )
    return result


def parse_pdf_with_warnings(
    pdf_path: str | Path,
    *,
    remove_headers_footers: bool = False,
) -> tuple[dict, list[str]]:
    """Parse an arXiv-style PDF into per-page text and warnings.

    Args:
        pdf_path: Path to a PDF file on disk.
        remove_headers_footers: Remove repeated header/footer lines when True.

    Returns:
        Tuple of parsed metadata and warning messages.

    Raises:
        FileNotFoundError: When the PDF file does not exist.
        ValueError: When the PDF cannot be opened or read.

    Edge cases:
        Pages that fail to extract are skipped; warnings include skipped page info.
    """

    return _parse_pdf_with_warnings(
        Path(pdf_path),
        remove_headers_footers=remove_headers_footers,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse PDF text with PyMuPDF.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument(
        "--out",
        help="Output JSON path. Defaults to data/parsed/<pdf_stem>.json.",
    )
    parser.add_argument(
        "--remove-headers-footers",
        action="store_true",
        help="Remove repeated header/footer lines.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        result, warnings = _parse_pdf_with_warnings(
            Path(args.pdf), remove_headers_footers=args.remove_headers_footers
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    pdf_path = Path(args.pdf)
    out_path = (
        Path(args.out)
        if args.out
        else Path("data") / "parsed" / f"{pdf_path.stem}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    pages = result["pages"]
    total_chars = sum(len(page["text"]) for page in pages)
    avg_chars = total_chars / len(pages) if pages else 0
    print(f"Parsed {len(pages)} pages. Avg chars/page: {avg_chars:.0f}.")

    if warnings:
        print("Warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"- {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
