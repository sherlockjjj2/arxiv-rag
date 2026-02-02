#!/usr/bin/env python3
"""Search and download arXiv PDFs via the arxiv package."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import arxiv
import requests

PDF_DIR = Path("data/arxiv-papers")
ID_INDEX = PDF_DIR / "arxiv_ids.txt"
DEFAULT_DB_PATH = Path("data/arxiv_rag.db")

_NEW_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_OLD_ID_RE = re.compile(
    r"^[a-z-]+(\.[A-Z]{2})?/\d{7}$",
    re.IGNORECASE,
)
_VERSIONED_NEW_ID_RE = re.compile(r"^(?P<base>\d{4}\.\d{4,5})v\d+$")
_VERSIONED_OLD_ID_RE = re.compile(
    r"^(?P<base>[a-z-]+(\.[A-Z]{2})?/\d{7})v\d+$",
    re.IGNORECASE,
)
_USER_AGENT = "arxiv-rag/0.1.0"
_CLIENT_DELAY_SECONDS = 10.0
_CLIENT_NUM_RETRIES = 5
SOURCE_TYPE_ARXIV = "arxiv"

SORT_CRITERIA: dict[str, arxiv.SortCriterion] = {
    "relevance": arxiv.SortCriterion.Relevance,
    "submittedDate": arxiv.SortCriterion.SubmittedDate,
    "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
}


@dataclass(frozen=True)
class DownloadConfig:
    """Runtime configuration for download operations.

    Attributes:
        pdf_dir: Directory for downloaded PDFs.
        id_index: Path to text file of downloaded base IDs.
        user_agent: User-Agent string for PDF requests.
        client_delay_seconds: Delay between arXiv API calls.
        client_num_retries: Retry count for arXiv API calls.
        default_db_path: Default metadata SQLite path.
    """

    pdf_dir: Path
    id_index: Path
    user_agent: str
    client_delay_seconds: float
    client_num_retries: int
    default_db_path: Path


@dataclass(frozen=True)
class PaperMetadata:
    """Metadata for a paper suitable for SQLite ingestion."""

    paper_id: str
    title: str
    authors: str
    abstract: str | None
    categories: str
    published_date: str | None
    pdf_path: str
    total_pages: int | None
    source_type: str = SOURCE_TYPE_ARXIV

    def as_row(self) -> tuple[object, ...]:
        """Return the metadata as a SQLite row tuple.

        Returns:
            Tuple matching the papers table column order.
        """

        return (
            self.paper_id,
            self.title,
            self.authors,
            self.abstract,
            self.categories,
            self.published_date,
            self.pdf_path,
            self.total_pages,
            self.source_type,
        )


@dataclass(frozen=True)
class ArxivApiClient:
    """Wrapper around the arxiv Client for structured access."""

    client: arxiv.Client

    def search(
        self,
        query: str,
        max_results: int,
        sort: str,
    ) -> Iterable[arxiv.Result]:
        """Yield search results for a query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            sort: Sort key matching SORT_CRITERIA.
        Returns:
            Iterable of arxiv.Result objects.
        Raises:
            ValueError: If the sort key is not recognized.
        Edge cases:
            Empty results yield an empty iterator.
        """

        if (sort_by := SORT_CRITERIA.get(sort)) is None:
            raise ValueError(f"Unsupported sort key: {sort}")
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )
        return self.client.results(search)

    def fetch_results(self, ids: Iterable[str]) -> dict[str, arxiv.Result]:
        """Fetch metadata for a list of arXiv IDs.

        Args:
            ids: Iterable of base arXiv IDs.
        Returns:
            Mapping of base ID to arxiv.Result.
        Edge cases:
            Returns an empty dict if ids is empty.
        """

        search = arxiv.Search(id_list=list(ids))
        results: dict[str, arxiv.Result] = {}
        for result in self.client.results(search):
            short_id = result.get_short_id()
            base_id = _base_id_from_versioned(short_id)
            results[base_id] = result
        return results


def default_config() -> DownloadConfig:
    """Build the default configuration from module constants.

    Returns:
        DownloadConfig populated with module defaults.
    Edge cases:
        Uses current module-level paths, which may be monkeypatched in tests.
    """

    return DownloadConfig(
        pdf_dir=PDF_DIR,
        id_index=ID_INDEX,
        user_agent=_USER_AGENT,
        client_delay_seconds=_CLIENT_DELAY_SECONDS,
        client_num_retries=_CLIENT_NUM_RETRIES,
        default_db_path=DEFAULT_DB_PATH,
    )


def create_arxiv_client(config: DownloadConfig) -> ArxivApiClient:
    """Create a configured arXiv API client wrapper.

    Args:
        config: Runtime configuration for API calls.
    Returns:
        ArxivApiClient ready for search and fetch operations.
    Raises:
        ValueError: If the arxiv client cannot be created.
    """

    client = arxiv.Client(
        delay_seconds=config.client_delay_seconds,
        num_retries=config.client_num_retries,
    )
    return ArxivApiClient(client=client)


def _resolve_config(config: DownloadConfig | None) -> DownloadConfig:
    """Return the provided config or fall back to defaults.

    Args:
        config: Optional configuration override.
    Returns:
        DownloadConfig to use for the operation.
    Edge cases:
        Returns a fresh default configuration when config is None.
    """

    return config or default_config()


def _resolve_client(
    client: ArxivApiClient | None,
    config: DownloadConfig,
) -> ArxivApiClient:
    """Return the provided client or create one from config.

    Args:
        client: Optional arXiv client wrapper.
        config: Runtime configuration for building a client.
    Returns:
        ArxivApiClient to use for API calls.
    """

    return client or create_arxiv_client(config)


def _ensure_paths(config: DownloadConfig | None = None) -> None:
    """Ensure the PDF directory and ID index file exist.

    Args:
        config: Optional configuration override.
    Raises:
        OSError: If directory or file creation fails.
    """

    resolved_config = _resolve_config(config)
    resolved_config.pdf_dir.mkdir(parents=True, exist_ok=True)
    if not resolved_config.id_index.exists():
        resolved_config.id_index.touch()


def _load_id_index(config: DownloadConfig | None = None) -> set[str]:
    """Load base IDs from the index file.

    Args:
        config: Optional configuration override.
    Returns:
        Set of base arXiv IDs recorded in the index.
    Raises:
        OSError: If the index file cannot be read.
    Edge cases:
        Returns an empty set when the index is missing.
    """

    resolved_config = _resolve_config(config)
    if not resolved_config.id_index.exists():
        return set()
    with resolved_config.id_index.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _sync_id_index(
    existing: set[str],
    config: DownloadConfig | None = None,
) -> None:
    """Rebuild the ID index from PDFs on disk.

    Args:
        existing: Mutable set updated in-place with discovered IDs.
        config: Optional configuration override.
    Raises:
        OSError: If directory traversal or file writing fails.
    Edge cases:
        Ignores PDFs whose filenames are not valid arXiv IDs.
    """

    resolved_config = _resolve_config(config)
    current_ids: set[str] = set()
    if resolved_config.pdf_dir.exists():
        for pdf_path in resolved_config.pdf_dir.glob("*.pdf"):
            base_id = _base_id_from_versioned(pdf_path.stem)
            if _is_valid_base_id(base_id):
                current_ids.add(base_id)

    existing.clear()
    existing.update(current_ids)
    _write_id_index(existing, resolved_config)


def _write_id_index(
    existing: set[str],
    config: DownloadConfig | None = None,
) -> None:
    """Persist the ID index to disk.

    Args:
        existing: Set of base IDs to write.
        config: Optional configuration override.
    Raises:
        OSError: If the index file cannot be written.
    Edge cases:
        Writes an empty file when existing is empty.
    """

    resolved_config = _resolve_config(config)
    with resolved_config.id_index.open("w", encoding="utf-8") as handle:
        for base_id in sorted(existing):
            handle.write(f"{base_id}\n")


def _base_id_from_versioned(versioned_id: str) -> str:
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


def _is_valid_base_id(base_id: str) -> bool:
    """Check whether a base arXiv ID matches known patterns.

    Args:
        base_id: Base arXiv ID to validate.
    Returns:
        True if the ID matches a valid format, otherwise False.
    """

    return bool(_NEW_ID_RE.match(base_id) or _OLD_ID_RE.match(base_id))


def _validate_base_ids(ids: Iterable[str]) -> list[str]:
    """Collect invalid base IDs from an iterable.

    Args:
        ids: Iterable of base IDs to validate.
    Returns:
        List of invalid base IDs.
    Edge cases:
        Returns an empty list when all IDs are valid.
    """

    return [base_id for base_id in ids if not _is_valid_base_id(base_id)]


def _search(
    query: str,
    max_results: int,
    sort: str,
    *,
    client: ArxivApiClient | None = None,
    config: DownloadConfig | None = None,
) -> int:
    """Search arXiv and print base IDs with titles.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        sort: Sort key matching SORT_CRITERIA.
        client: Optional arXiv client wrapper.
        config: Optional configuration override.
    Returns:
        Exit code (0 for success).
    Raises:
        ValueError: If sort is not supported by SORT_CRITERIA.
    """

    resolved_config = _resolve_config(config)
    resolved_client = _resolve_client(client, resolved_config)
    for result in resolved_client.search(query, max_results, sort):
        short_id = result.get_short_id()
        base_id = _base_id_from_versioned(short_id)
        print(f"{base_id}\t{result.title}")
    return 0


def _fetch_results(
    ids: Iterable[str],
    *,
    client: ArxivApiClient | None = None,
    config: DownloadConfig | None = None,
) -> dict[str, arxiv.Result]:
    """Fetch metadata for a list of arXiv IDs.

    Args:
        ids: Iterable of base arXiv IDs.
        client: Optional arXiv client wrapper.
        config: Optional configuration override.
    Returns:
        Mapping of base ID to arxiv.Result.
    Edge cases:
        Returns an empty dict when ids is empty.
    """

    resolved_config = _resolve_config(config)
    resolved_client = _resolve_client(client, resolved_config)
    return resolved_client.fetch_results(ids)


def _download_pdf(
    result: arxiv.Result,
    dest_path: Path,
    timeout: int,
    delay_seconds: float | None = None,
    *,
    user_agent: str | None = None,
    http_get: Callable[..., requests.Response] | None = None,
) -> float | None:
    """Download a PDF to the destination path.

    Args:
        result: arXiv result containing the PDF URL.
        dest_path: Destination path for the PDF.
        timeout: Request timeout in seconds.
        delay_seconds: Backoff delay to return when rate-limited.
        user_agent: Optional User-Agent header value.
        http_get: Optional HTTP GET function for dependency injection.
    Returns:
        Retry delay in seconds when rate-limited, otherwise None.
    Raises:
        requests.RequestException: If the request fails unexpectedly.
        OSError: If file writes fail.
    Edge cases:
        Returns None immediately when the result has no PDF URL.
    """

    if not result.pdf_url:
        return None
    headers = {"User-Agent": user_agent or _USER_AGENT}
    http_get = http_get or requests.get
    with http_get(
        result.pdf_url,
        stream=True,
        timeout=timeout,
        headers=headers,
    ) as response:
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                return float(retry_after)
            return float(delay_seconds or 1)
        response.raise_for_status()
        tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.part")
        try:
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            tmp_path.replace(dest_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return None


def _ensure_db(db_path: Path) -> None:
    """Create the SQLite database and papers table if missing.

    Args:
        db_path: SQLite database path.
    Raises:
        sqlite3.Error: If the database cannot be initialized.
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                doc_id TEXT,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                categories TEXT,
                published_date TEXT,
                pdf_path TEXT,
                total_pages INTEGER,
                indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                source_type TEXT DEFAULT 'arxiv',
                UNIQUE(doc_id)
            );
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(papers)")}
        if "doc_id" not in columns:
            conn.execute("ALTER TABLE papers ADD COLUMN doc_id TEXT")
        conn.commit()


def _metadata_from_result(result: arxiv.Result, pdf_path: Path) -> PaperMetadata:
    """Build a metadata model from an arXiv result.

    Args:
        result: arXiv result metadata.
        pdf_path: Path to the downloaded PDF.
    Returns:
        PaperMetadata populated from the result.
    Edge cases:
        Missing categories or published dates are stored as empty/None.
    """

    short_id = result.get_short_id()
    base_id = _base_id_from_versioned(short_id)
    authors = json.dumps([author.name for author in result.authors] or [])
    categories = json.dumps(list(result.categories) if result.categories else [])
    published = result.published.date().isoformat() if result.published else None

    return PaperMetadata(
        paper_id=base_id,
        title=result.title,
        authors=authors,
        abstract=result.summary,
        categories=categories,
        published_date=published,
        pdf_path=str(pdf_path),
        total_pages=None,
        source_type=SOURCE_TYPE_ARXIV,
    )


def _bulk_insert_metadata(
    db_path: Path,
    rows: Sequence[PaperMetadata],
) -> None:
    """Insert metadata rows into the database.

    Args:
        db_path: SQLite database path.
        rows: Sequence of PaperMetadata entries to insert.
    Raises:
        sqlite3.Error: If the database insert fails.
    Edge cases:
        No-op when rows is empty.
    """

    if not rows:
        return
    _ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO papers (
                paper_id,
                title,
                authors,
                abstract,
                categories,
                published_date,
                pdf_path,
                total_pages,
                source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                title = excluded.title,
                authors = excluded.authors,
                abstract = excluded.abstract,
                categories = excluded.categories,
                published_date = excluded.published_date,
                pdf_path = excluded.pdf_path,
                total_pages = excluded.total_pages,
                source_type = excluded.source_type,
                indexed_at = CURRENT_TIMESTAMP;
            """,
            [row.as_row() for row in rows],
        )
        conn.commit()


def _load_db_ids(db_path: Path) -> set[str]:
    """Load all paper IDs from the database.

    Args:
        db_path: SQLite database path.
    Returns:
        Set of paper IDs already stored in the database.
    Raises:
        sqlite3.Error: If the database query fails.
    Edge cases:
        Returns an empty set when the database does not exist.
    """

    if not db_path.exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT paper_id FROM papers").fetchall()
    return {row[0] for row in rows}


def _collect_pdf_paths(config: DownloadConfig | None = None) -> dict[str, Path]:
    """Collect the newest PDF path for each base ID.

    Args:
        config: Optional configuration override.
    Returns:
        Mapping of base ID to the most recently modified PDF path.
    Edge cases:
        Returns an empty dict when the PDF directory is missing.
    """

    resolved_config = _resolve_config(config)
    pdf_paths: dict[str, Path] = {}
    if not resolved_config.pdf_dir.exists():
        return pdf_paths
    for pdf_path in resolved_config.pdf_dir.glob("*.pdf"):
        base_id = _base_id_from_versioned(pdf_path.stem)
        if not _is_valid_base_id(base_id):
            continue
        if (
            base_id in pdf_paths
            and pdf_paths[base_id].stat().st_mtime >= pdf_path.stat().st_mtime
        ):
            continue
        pdf_paths[base_id] = pdf_path
    return pdf_paths


def _backfill_metadata(
    db_path: Path,
    *,
    client: ArxivApiClient | None = None,
    config: DownloadConfig | None = None,
) -> int:
    """Fetch metadata for PDFs already on disk.

    Args:
        db_path: SQLite database path.
        client: Optional arXiv client wrapper.
        config: Optional configuration override.
    Returns:
        Exit code (0 for success, 1 for missing metadata).
    """

    resolved_config = _resolve_config(config)
    resolved_client = _resolve_client(client, resolved_config)
    _ensure_db(db_path)
    pdf_paths = _collect_pdf_paths(resolved_config)
    if not pdf_paths:
        print("No PDFs found for metadata backfill.")
        return 0

    existing_ids = _load_db_ids(db_path)
    pending_ids = [base_id for base_id in pdf_paths if base_id not in existing_ids]
    if not pending_ids:
        print("All PDFs already have metadata.")
        return 0

    results = _fetch_results(pending_ids, client=resolved_client, config=resolved_config)
    metadata_rows: list[PaperMetadata] = []
    missing_ids: list[str] = []

    for base_id in pending_ids:
        if (result := results.get(base_id)) is None:
            missing_ids.append(base_id)
            continue
        metadata_rows.append(_metadata_from_result(result, pdf_paths[base_id]))

    if missing_ids:
        for base_id in missing_ids:
            print(f"Error: no metadata found for {base_id}.", file=sys.stderr)

    if metadata_rows:
        _bulk_insert_metadata(db_path, metadata_rows)
        print(f"Metadata fetched and stored for {len(metadata_rows)} papers.")
    return 1 if missing_ids else 0


def _should_sync_id_index(config: DownloadConfig | None = None) -> bool:
    """Determine whether the ID index should be rebuilt.

    Args:
        config: Optional configuration override.
    Returns:
        True if the index should be rebuilt, otherwise False.
    Edge cases:
        Returns True when the index is missing and PDFs exist.
    """

    resolved_config = _resolve_config(config)
    if not resolved_config.id_index.exists():
        return True
    if not resolved_config.pdf_dir.exists():
        return False
    if resolved_config.id_index.stat().st_size == 0:
        return any(resolved_config.pdf_dir.glob("*.pdf"))
    index_mtime = resolved_config.id_index.stat().st_mtime
    return any(
        pdf_path.stat().st_mtime >= index_mtime
        for pdf_path in resolved_config.pdf_dir.glob("*.pdf")
    )


def _deduped_ids(ids: Iterable[str]) -> list[str]:
    """Deduplicate IDs while preserving order.

    Args:
        ids: Iterable of base IDs.
    Returns:
        List of unique base IDs in their first-seen order.
    """

    unique_ids: list[str] = []
    seen: set[str] = set()
    for base_id in ids:
        if base_id in seen:
            continue
        seen.add(base_id)
        unique_ids.append(base_id)
    return unique_ids


def _report_invalid_ids(invalid_ids: Iterable[str]) -> None:
    """Report invalid arXiv IDs to stderr.

    Args:
        invalid_ids: Iterable of invalid base IDs.
    """

    for base_id in invalid_ids:
        print(f"Error: invalid arXiv ID format: {base_id}", file=sys.stderr)


def _attempt_download(
    base_id: str,
    result: arxiv.Result,
    dest_path: Path,
    retries: int,
    timeout: int,
    user_agent: str,
) -> str | None:
    """Attempt to download a PDF with retries and backoff.

    Args:
        base_id: Base arXiv ID being downloaded.
        result: arXiv result metadata.
        dest_path: Destination path for the PDF.
        retries: Maximum number of attempts.
        timeout: Request timeout in seconds.
        user_agent: User-Agent header value.
    Returns:
        Error message on final failure, otherwise None.
    Edge cases:
        Deletes partial files if a download attempt fails.
    """

    for attempt in range(retries):
        try:
            retry_after = _download_pdf(
                result,
                dest_path,
                timeout,
                delay_seconds=2**attempt,
                user_agent=user_agent,
            )
            if retry_after is not None:
                if attempt == retries - 1:
                    raise RuntimeError("rate limited by arXiv")
                time.sleep(retry_after)
                continue
            return None
        except Exception as exc:  # noqa: BLE001
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            if attempt == retries - 1:
                return f"failed to download {base_id} after {retries} attempts: {exc}"
            time.sleep(2**attempt)
    return f"failed to download {base_id} after {retries} attempts"


def _collect_pending_ids(unique_ids: list[str], existing_ids: set[str]) -> list[str]:
    """Filter IDs that are not yet downloaded.

    Args:
        unique_ids: Deduplicated list of base IDs.
        existing_ids: Set of IDs already downloaded.
    Returns:
        List of base IDs that still need downloads.
    Edge cases:
        Prints skipped messages when all IDs already exist.
    """

    if pending_ids := [
        base_id for base_id in unique_ids if base_id not in existing_ids
    ]:
        return pending_ids

    for base_id in unique_ids:
        print(f"Skipped {base_id}: already downloaded.")
    return []


def _handle_download_result(
    base_id: str,
    result: arxiv.Result | None,
    existing_ids: set[str],
    retries: int,
    timeout: int,
    config: DownloadConfig,
) -> tuple[Path | None, str | None]:
    """Process a single arXiv result and download its PDF.

    Args:
        base_id: Base arXiv ID to process.
        result: arXiv result metadata, if available.
        existing_ids: Set of IDs already downloaded (mutated in-place).
        retries: Maximum download attempts.
        timeout: Request timeout in seconds.
        config: Runtime configuration for paths and headers.
    Returns:
        Tuple of destination path (if downloaded) and error message (if any).
    Edge cases:
        Skips downloads when PDFs already exist on disk.
    """

    if base_id in existing_ids:
        print(f"Skipped {base_id}: already downloaded.")
        return None, None
    if result is None:
        return None, f"no metadata found for {base_id}."
    if not result.pdf_url:
        return None, f"no PDF available for {base_id}."

    short_id = result.get_short_id()
    dest_path = config.pdf_dir / f"{short_id}.pdf"
    if dest_path.exists():
        existing_ids.add(base_id)
        print(f"Skipped {base_id}: PDF exists.")
        return dest_path, None
    if error := _attempt_download(
        base_id,
        result,
        dest_path,
        retries,
        timeout,
        config.user_agent,
    ):
        return None, error
    existing_ids.add(base_id)
    print(f"Downloaded {base_id} -> {dest_path}")
    return dest_path, None


def _download_by_id(
    ids: list[str],
    retries: int,
    timeout: int,
    db_path: Path | None = None,
    *,
    client: ArxivApiClient | None = None,
    config: DownloadConfig | None = None,
) -> int:
    """Download PDFs and optionally ingest metadata into SQLite.

    Args:
        ids: List of base arXiv IDs.
        retries: Maximum download attempts per ID.
        timeout: Request timeout in seconds.
        db_path: Optional SQLite database path for metadata ingestion.
        client: Optional arXiv client wrapper.
        config: Optional configuration override.
    Returns:
        Exit code (0 for success, 1 for failures).
    """

    resolved_config = _resolve_config(config)
    resolved_client = _resolve_client(client, resolved_config)
    _ensure_paths(resolved_config)
    existing_ids = _load_id_index(resolved_config)
    if _should_sync_id_index(resolved_config):
        _sync_id_index(existing_ids, resolved_config)

    unique_ids = _deduped_ids(ids)
    if invalid_ids := _validate_base_ids(unique_ids):
        _report_invalid_ids(invalid_ids)
        return 1

    if not (pending_ids := _collect_pending_ids(unique_ids, existing_ids)):
        return 0

    results = _fetch_results(pending_ids, client=resolved_client, config=resolved_config)
    failures: list[str] = []
    metadata_rows: list[PaperMetadata] = []

    for base_id in unique_ids:
        result = results.get(base_id)
        dest_path, error = _handle_download_result(
            base_id,
            result,
            existing_ids,
            retries,
            timeout,
            resolved_config,
        )
        if error is None:
            if db_path and dest_path and result:
                metadata_rows.append(_metadata_from_result(result, dest_path))
            continue
        print(f"Error: {error}", file=sys.stderr)
        failures.append(base_id)

    _write_id_index(existing_ids, resolved_config)
    if db_path and metadata_rows:
        _bulk_insert_metadata(db_path, metadata_rows)
        print(f"Metadata fetched and stored for {len(metadata_rows)} papers.")
    return 1 if failures else 0


def _parse_args(config: DownloadConfig | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        config: Optional configuration override.
    Returns:
        Parsed argparse namespace.
    """

    resolved_config = _resolve_config(config)
    parser = argparse.ArgumentParser(description="Search and download arXiv PDFs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--query", type=str, help="Search query string.")
    mode.add_argument("--ids", nargs="+", help="One or more base arXiv IDs.")

    parser.add_argument(
        "--max-results",
        type=int,
        default=25,
        help="Maximum number of search results.",
    )
    parser.add_argument(
        "--sort",
        choices=list(SORT_CRITERIA.keys()),
        default="relevance",
        help="Sort order for search results.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Download retries per ID.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Download timeout in seconds.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=resolved_config.default_db_path,
        help="SQLite DB path for metadata ingestion.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable metadata ingestion into SQLite.",
    )
    parser.add_argument(
        "--backfill-db",
        action="store_true",
        help="Ingest metadata for existing PDFs in the data directory.",
    )

    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for arXiv search and download.

    Returns:
        Exit code following CLI conventions.
    """

    config = default_config()
    args = _parse_args(config)
    db_path = None if args.no_db else args.db
    if not (args.query or args.ids or args.backfill_db):
        print("Error: one of --query, --ids, or --backfill-db is required.", file=sys.stderr)
        return 2
    if db_path and (args.ids or args.backfill_db):
        _ensure_db(db_path)
    if args.backfill_db:
        return _backfill_metadata(db_path, config=config) if db_path else 2
    if args.query:
        return _search(args.query, args.max_results, args.sort, config=config)
    return (
        _download_by_id(args.ids, args.retries, args.timeout, db_path=db_path, config=config)
        if args.ids
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
