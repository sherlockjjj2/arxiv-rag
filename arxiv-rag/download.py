#!/usr/bin/env python3
"""Search and download arXiv PDFs via the arxiv package."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import arxiv
import requests

PDF_DIR = Path("data/arxiv-papers")
ID_INDEX = PDF_DIR / "arxiv_ids.txt"

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
_CLIENT: Optional[arxiv.Client] = None
DEFAULT_DB_PATH = Path("data/arxiv_rag.db")


def _ensure_paths() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    if not ID_INDEX.exists():
        ID_INDEX.touch()


def _load_id_index() -> Set[str]:
    if not ID_INDEX.exists():
        return set()
    with ID_INDEX.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _sync_id_index(existing: Set[str]) -> None:
    current_ids: Set[str] = set()
    if PDF_DIR.exists():
        for pdf_path in PDF_DIR.glob("*.pdf"):
            base_id = _base_id_from_versioned(pdf_path.stem)
            if _is_valid_base_id(base_id):
                current_ids.add(base_id)

    existing.clear()
    existing.update(current_ids)
    _write_id_index(existing)


def _write_id_index(existing: Set[str]) -> None:
    with ID_INDEX.open("w", encoding="utf-8") as handle:
        for base_id in sorted(existing):
            handle.write(f"{base_id}\n")


def _base_id_from_versioned(versioned_id: str) -> str:
    if match := _VERSIONED_NEW_ID_RE.match(versioned_id):
        return match.group("base")
    if match := _VERSIONED_OLD_ID_RE.match(versioned_id):
        return match.group("base")
    return versioned_id


def _is_valid_base_id(base_id: str) -> bool:
    return bool(_NEW_ID_RE.match(base_id) or _OLD_ID_RE.match(base_id))


def _validate_base_ids(ids: Iterable[str]) -> List[str]:
    return [base_id for base_id in ids if not _is_valid_base_id(base_id)]


def _get_client() -> arxiv.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = arxiv.Client(
            delay_seconds=_CLIENT_DELAY_SECONDS,
            num_retries=_CLIENT_NUM_RETRIES,
        )
    return _CLIENT


def _search(query: str, max_results: int, sort: str) -> int:
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_map[sort],
    )
    client = _get_client()
    for result in client.results(search):
        short_id = result.get_short_id()
        base_id = _base_id_from_versioned(short_id)
        print(f"{base_id}\t{result.title}")
    return 0


def _fetch_results(ids: Iterable[str]) -> Dict[str, arxiv.Result]:
    search = arxiv.Search(id_list=list(ids))
    client = _get_client()
    results: Dict[str, arxiv.Result] = {}
    for result in client.results(search):
        short_id = result.get_short_id()
        base_id = _base_id_from_versioned(short_id)
        results[base_id] = result
    return results


def _download_pdf(
    result: arxiv.Result,
    dest_path: Path,
    timeout: int,
    delay_seconds: Optional[int] = None,
) -> Optional[int]:
    if not result.pdf_url:
        return None
    headers = {"User-Agent": _USER_AGENT}
    with requests.get(
        result.pdf_url,
        stream=True,
        timeout=timeout,
        headers=headers,
    ) as response:
        if response.status_code == 429:
            return (
                int(retry_after)
                if (retry_after := response.headers.get("Retry-After"))
                and retry_after.isdigit()
                else (delay_seconds or 1)
            )
        response.raise_for_status()
        tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.part")
        try:
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            os.replace(tmp_path, dest_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return None


def _ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                categories TEXT,
                published_date TEXT,
                pdf_path TEXT,
                total_pages INTEGER,
                indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                source_type TEXT DEFAULT 'arxiv'
            );
            """
        )
        conn.commit()


def _metadata_row_from_result(result: arxiv.Result, pdf_path: Path) -> Tuple[object, ...]:
    short_id = result.get_short_id()
    base_id = _base_id_from_versioned(short_id)
    authors = json.dumps([author.name for author in result.authors])
    categories = json.dumps(list(result.categories) if result.categories else [])
    published = result.published.date().isoformat() if result.published else None

    return (
        base_id,
        result.title,
        authors,
        result.summary,
        categories,
        published,
        str(pdf_path),
        None,
        "arxiv",
    )


def _bulk_insert_metadata(db_path: Path, rows: Sequence[Tuple[object, ...]]) -> None:
    if not rows:
        return
    _ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO papers (
                paper_id,
                title,
                authors,
                abstract,
                categories,
                published_date,
                pdf_path,
                total_pages,
                source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )
        conn.commit()


def _load_db_ids(db_path: Path) -> Set[str]:
    if not db_path.exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT paper_id FROM papers").fetchall()
    return {row[0] for row in rows}


def _collect_pdf_paths() -> Dict[str, Path]:
    pdf_paths: Dict[str, Path] = {}
    if not PDF_DIR.exists():
        return pdf_paths
    for pdf_path in PDF_DIR.glob("*.pdf"):
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


def _backfill_metadata(db_path: Path) -> int:
    _ensure_db(db_path)
    pdf_paths = _collect_pdf_paths()
    if not pdf_paths:
        print("No PDFs found for metadata backfill.")
        return 0

    existing_ids = _load_db_ids(db_path)
    pending_ids = [base_id for base_id in pdf_paths if base_id not in existing_ids]
    if not pending_ids:
        print("All PDFs already have metadata.")
        return 0

    results = _fetch_results(pending_ids)
    metadata_rows: List[Tuple[object, ...]] = []
    missing_ids: List[str] = []

    for base_id in pending_ids:
        if (result := results.get(base_id)) is None:
            missing_ids.append(base_id)
            continue
        metadata_rows.append(_metadata_row_from_result(result, pdf_paths[base_id]))

    if missing_ids:
        for base_id in missing_ids:
            print(f"Error: no metadata found for {base_id}.", file=sys.stderr)

    if metadata_rows:
        _bulk_insert_metadata(db_path, metadata_rows)
        print(f"Metadata fetched and stored for {len(metadata_rows)} papers.")
    return 1 if missing_ids else 0


def _should_sync_id_index() -> bool:
    if not ID_INDEX.exists():
        return True
    if not PDF_DIR.exists():
        return False
    if ID_INDEX.stat().st_size == 0:
        return any(PDF_DIR.glob("*.pdf"))
    index_mtime = ID_INDEX.stat().st_mtime
    return any(
        pdf_path.stat().st_mtime >= index_mtime
        for pdf_path in PDF_DIR.glob("*.pdf")
    )


def _deduped_ids(ids: Iterable[str]) -> List[str]:
    unique_ids: List[str] = []
    seen: Set[str] = set()
    for base_id in ids:
        if base_id in seen:
            continue
        seen.add(base_id)
        unique_ids.append(base_id)
    return unique_ids


def _report_invalid_ids(invalid_ids: Iterable[str]) -> None:
    for base_id in invalid_ids:
        print(f"Error: invalid arXiv ID format: {base_id}", file=sys.stderr)


def _attempt_download(
    base_id: str,
    result: arxiv.Result,
    dest_path: Path,
    retries: int,
    timeout: int,
) -> Optional[str]:
    for attempt in range(retries):
        try:
            retry_after = _download_pdf(
                result,
                dest_path,
                timeout,
                delay_seconds=2 ** attempt,
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
            time.sleep(2 ** attempt)
    return f"failed to download {base_id} after {retries} attempts"


def _collect_pending_ids(unique_ids: List[str], existing_ids: Set[str]) -> List[str]:
    if pending_ids := [
        base_id for base_id in unique_ids if base_id not in existing_ids
    ]:
        return pending_ids

    for base_id in unique_ids:
        print(f"Skipped {base_id}: already downloaded.")
    return []


def _handle_download_result(
    base_id: str,
    result: Optional[arxiv.Result],
    existing_ids: Set[str],
    retries: int,
    timeout: int,
) -> Tuple[Optional[Path], Optional[str]]:
    if base_id in existing_ids:
        print(f"Skipped {base_id}: already downloaded.")
        return None, None
    if result is None:
        return None, f"no metadata found for {base_id}."
    if not result.pdf_url:
        return None, f"no PDF available for {base_id}."

    short_id = result.get_short_id()
    dest_path = PDF_DIR / f"{short_id}.pdf"
    if dest_path.exists():
        existing_ids.add(base_id)
        print(f"Skipped {base_id}: PDF exists.")
        return dest_path, None
    if error := _attempt_download(base_id, result, dest_path, retries, timeout):
        return None, error
    existing_ids.add(base_id)
    print(f"Downloaded {base_id} -> {dest_path}")
    return dest_path, None


def _download_by_id(
    ids: List[str],
    retries: int,
    timeout: int,
    db_path: Optional[Path] = None,
) -> int:
    _ensure_paths()
    existing_ids = _load_id_index()
    if _should_sync_id_index():
        _sync_id_index(existing_ids)

    unique_ids = _deduped_ids(ids)
    if invalid_ids := _validate_base_ids(unique_ids):
        _report_invalid_ids(invalid_ids)
        return 1

    if not (pending_ids := _collect_pending_ids(unique_ids, existing_ids)):
        return 0

    results = _fetch_results(pending_ids)
    failures: List[str] = []
    metadata_rows: List[Tuple[object, ...]] = []

    for base_id in unique_ids:
        result = results.get(base_id)
        dest_path, error = _handle_download_result(
            base_id,
            result,
            existing_ids,
            retries,
            timeout,
        )
        if error is None:
            if db_path and dest_path and result:
                metadata_rows.append(_metadata_row_from_result(result, dest_path))
            continue
        print(f"Error: {error}", file=sys.stderr)
        failures.append(base_id)

    _write_id_index(existing_ids)
    if db_path and metadata_rows:
        _bulk_insert_metadata(db_path, metadata_rows)
        print(f"Metadata fetched and stored for {len(metadata_rows)} papers.")
    return 1 if failures else 0


def _parse_args() -> argparse.Namespace:
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
        choices=["relevance", "submittedDate", "lastUpdatedDate"],
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
        default=DEFAULT_DB_PATH,
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
    args = _parse_args()
    db_path = None if args.no_db else args.db
    if not (args.query or args.ids or args.backfill_db):
        print("Error: one of --query, --ids, or --backfill-db is required.", file=sys.stderr)
        return 2
    if db_path and (args.ids or args.backfill_db):
        _ensure_db(db_path)
    if args.backfill_db:
        return _backfill_metadata(db_path) if db_path else 2
    if args.query:
        return _search(args.query, args.max_results, args.sort)
    return (
        _download_by_id(args.ids, args.retries, args.timeout, db_path=db_path)
        if args.ids
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
