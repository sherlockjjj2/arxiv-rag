#!/usr/bin/env python3
"""Search and download arXiv PDFs via the arxiv package."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

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
    match = _VERSIONED_NEW_ID_RE.match(versioned_id)
    if match:
        return match.group("base")
    match = _VERSIONED_OLD_ID_RE.match(versioned_id)
    if match:
        return match.group("base")
    return versioned_id


def _is_valid_base_id(base_id: str) -> bool:
    return bool(_NEW_ID_RE.match(base_id) or _OLD_ID_RE.match(base_id))


def _validate_base_ids(ids: Iterable[str]) -> List[str]:
    invalid = [base_id for base_id in ids if not _is_valid_base_id(base_id)]
    return invalid


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
    headers = {"User-Agent": _USER_AGENT}
    response = requests.get(
        result.pdf_url,
        stream=True,
        timeout=timeout,
        headers=headers,
    )
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            response.close()
            return int(retry_after)
        response.close()
        return delay_seconds
    response.raise_for_status()
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
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


def _download_by_id(ids: List[str], retries: int, timeout: int) -> int:
    _ensure_paths()
    existing_ids = _load_id_index()
    _sync_id_index(existing_ids)

    unique_ids: List[str] = []
    seen: Set[str] = set()
    for base_id in ids:
        if base_id not in seen:
            seen.add(base_id)
            unique_ids.append(base_id)

    invalid_ids = _validate_base_ids(unique_ids)
    if invalid_ids:
        for base_id in invalid_ids:
            print(f"Error: invalid arXiv ID format: {base_id}", file=sys.stderr)
        return 1

    pending_ids = [base_id for base_id in unique_ids if base_id not in existing_ids]
    if not pending_ids:
        for base_id in unique_ids:
            print(f"Skipped {base_id}: already downloaded.")
        return 0

    results = _fetch_results(pending_ids)
    failures: List[str] = []

    for base_id in unique_ids:
        if base_id in existing_ids:
            print(f"Skipped {base_id}: already downloaded.")
            continue

        result = results.get(base_id)
        if result is None:
            print(f"Error: no metadata found for {base_id}.", file=sys.stderr)
            failures.append(base_id)
            continue

        short_id = result.get_short_id()
        if not result.pdf_url:
            print(f"Error: no PDF available for {base_id}.", file=sys.stderr)
            failures.append(base_id)
            continue

        dest_path = PDF_DIR / f"{short_id}.pdf"
        if dest_path.exists():
            print(f"Skipped {base_id}: PDF exists.")
            existing_ids.add(base_id)
            continue

        for attempt in range(retries):
            try:
                backoff_seconds = 2 ** attempt
                retry_after = _download_pdf(
                    result,
                    dest_path,
                    timeout,
                    delay_seconds=backoff_seconds,
                )
                if retry_after is not None:
                    if attempt == retries - 1:
                        raise RuntimeError("rate limited by arXiv")
                    time.sleep(retry_after)
                    continue
                existing_ids.add(base_id)
                print(f"Downloaded {base_id} -> {dest_path}")
                break
            except Exception as exc:  # noqa: BLE001
                if dest_path.exists():
                    dest_path.unlink(missing_ok=True)
                if attempt == retries - 1:
                    print(
                        f"Error: failed to download {base_id} after {retries} attempts: {exc}",
                        file=sys.stderr,
                    )
                    failures.append(base_id)
                else:
                    time.sleep(2 ** attempt)

    _write_id_index(existing_ids)
    return 1 if failures else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search and download arXiv PDFs.")
    mode = parser.add_mutually_exclusive_group(required=True)
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

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.query:
        return _search(args.query, args.max_results, args.sort)
    if args.ids:
        return _download_by_id(args.ids, args.retries, args.timeout)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
