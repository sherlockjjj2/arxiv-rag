from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tiktoken import Encoding, get_encoding

LOGGER = logging.getLogger(__name__)

_NEW_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_OLD_ID_RE = re.compile(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}$", re.IGNORECASE)
_VERSIONED_NEW_ID_RE = re.compile(r"^(?P<base>\d{4}\.\d{4,5})v\d+$")
_VERSIONED_OLD_ID_RE = re.compile(
    r"^(?P<base>[a-z-]+(\.[A-Z]{2})?/\d{7})v\d+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkConfig:
    """Runtime configuration for chunking."""

    target_tokens: int
    overlap_tokens: int
    encoding_name: str = "cl100k_base"


@dataclass(frozen=True)
class ParsedPage:
    """Page-level parsed text."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    """Parsed PDF document with per-page text."""

    doc_id: str
    pdf_path: Path | None
    pages: list[ParsedPage]


@dataclass(frozen=True)
class ChunkRecord:
    """Chunk metadata for storage."""

    paper_id: str
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    token_count: int

    def as_row(self) -> tuple[object, ...]:
        """Return the chunk as a SQLite row tuple."""

        return (
            self.paper_id,
            self.doc_id,
            self.page_number,
            self.chunk_index,
            self.text,
            self.char_start,
            self.char_end,
            self.token_count,
        )


def _base_id_from_versioned(arxiv_id: str) -> str:
    """Return the base arXiv ID without a version suffix."""

    if match := _VERSIONED_NEW_ID_RE.match(arxiv_id):
        return match.group("base")
    if match := _VERSIONED_OLD_ID_RE.match(arxiv_id):
        return match.group("base")
    return arxiv_id


def _infer_paper_id(parsed_doc: ParsedDocument, parsed_path: Path) -> str | None:
    """Infer a base arXiv ID from the parsed JSON or PDF path."""

    candidates = []
    if parsed_doc.pdf_path is not None:
        candidates.append(parsed_doc.pdf_path.stem)
    candidates.append(parsed_path.stem)

    for candidate in candidates:
        base_id = _base_id_from_versioned(candidate)
        if _NEW_ID_RE.match(base_id) or _OLD_ID_RE.match(base_id):
            return base_id
    return None


def load_parsed_document(path: Path) -> ParsedDocument:
    """Load a parsed JSON file from disk.

    Args:
        path: Path to the parsed JSON.
    Returns:
        ParsedDocument with doc_id, pdf_path, and pages.
    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If required fields are missing.
    """

    if not path.exists():
        raise FileNotFoundError(f"Parsed JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    doc_id = payload.get("doc_id")
    if not doc_id:
        raise ValueError(f"Parsed JSON missing doc_id: {path}")

    pdf_path_raw = payload.get("pdf_path") or ""
    pdf_path = Path(pdf_path_raw) if pdf_path_raw else None

    pages = []
    for page in payload.get("pages", []):
        page_number = page.get("page")
        text = page.get("text", "")
        if page_number is None:
            continue
        pages.append(ParsedPage(page_number=page_number, text=text))

    return ParsedDocument(doc_id=doc_id, pdf_path=pdf_path, pages=pages)


def chunk_page(
    page_text: str,
    page_num: int,
    paper_id: str,
    doc_id: str,
    config: ChunkConfig,
    encoder: Encoding,
) -> list[ChunkRecord]:
    """Chunk text by tokens with overlap for a single page.

    Args:
        page_text: Cleaned page text.
        page_num: 1-based page number from the PDF.
        paper_id: Base arXiv ID.
        doc_id: SHA1 of PDF bytes.
        config: Chunking configuration.
        encoder: Token encoder for chunk sizing.
    Returns:
        List of ChunkRecord entries for the page.
    Raises:
        ValueError: When the chunk configuration is invalid.
    """

    if config.target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    if config.overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if config.overlap_tokens >= config.target_tokens:
        raise ValueError("overlap_tokens must be < target_tokens")

    if not page_text.strip():
        return []

    tokens = encoder.encode(page_text)
    if not tokens:
        return []

    chunks: list[ChunkRecord] = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + config.target_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)

        char_start = len(encoder.decode(tokens[:start]))
        char_end = char_start + len(chunk_text)

        chunks.append(
            ChunkRecord(
                paper_id=paper_id,
                doc_id=doc_id,
                page_number=page_num,
                chunk_index=chunk_idx,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                token_count=len(chunk_tokens),
            )
        )

        start = end - config.overlap_tokens if end < len(tokens) else end
        chunk_idx += 1

    return chunks


def chunk_document(
    parsed_doc: ParsedDocument,
    paper_id: str,
    config: ChunkConfig,
) -> list[ChunkRecord]:
    """Chunk all pages for a parsed document."""

    encoder = get_encoding(config.encoding_name)
    chunks: list[ChunkRecord] = []
    for page in parsed_doc.pages:
        chunks.extend(
            chunk_page(
                page_text=page.text,
                page_num=page.page_number,
                paper_id=paper_id,
                doc_id=parsed_doc.doc_id,
                config=config,
                encoder=encoder,
            )
        )
    return chunks


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_papers_doc_id(conn: sqlite3.Connection) -> None:
    """Ensure the papers table has a doc_id column."""

    if not _table_exists(conn, "papers"):
        raise ValueError("papers table not found; run download.py with --db first.")

    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()
    }
    if "doc_id" not in columns:
        conn.execute("ALTER TABLE papers ADD COLUMN doc_id TEXT")


def _ensure_chunks_schema(conn: sqlite3.Connection) -> None:
    """Ensure the chunks and FTS tables exist."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            token_count INTEGER,
            embedding BLOB
        );
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='chunk_id'
        );
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
        END;
        """
    )


def _delete_chunks_where(
    conn: sqlite3.Connection,
    where_sql: str,
    params: tuple[object, ...],
) -> int:
    """Delete chunks and matching FTS rows for a predicate."""

    deleted = 0
    if _table_exists(conn, "chunks_fts"):
        conn.execute(
            f"""
            DELETE FROM chunks_fts
            WHERE rowid IN (SELECT chunk_id FROM chunks WHERE {where_sql})
            """,
            params,
        )
    cursor = conn.execute(f"DELETE FROM chunks WHERE {where_sql}", params)
    deleted = cursor.rowcount if cursor.rowcount is not None else 0
    return deleted


def _resolve_paper_id(
    conn: sqlite3.Connection | None,
    parsed_doc: ParsedDocument,
    parsed_path: Path,
    explicit_paper_id: str | None,
) -> str:
    """Resolve paper_id from CLI, DB, or filename."""

    if explicit_paper_id:
        return explicit_paper_id

    if conn is not None and parsed_doc.pdf_path is not None:
        row = conn.execute(
            "SELECT paper_id FROM papers WHERE pdf_path = ?",
            (str(parsed_doc.pdf_path),),
        ).fetchone()
        if row:
            return row[0]

        rows = conn.execute(
            "SELECT paper_id, pdf_path FROM papers WHERE pdf_path LIKE ?",
            (f"%{parsed_doc.pdf_path.name}",),
        ).fetchall()
        if len(rows) == 1:
            return rows[0][0]

    inferred = _infer_paper_id(parsed_doc, parsed_path)
    if inferred:
        return inferred

    raise ValueError(
        f"Unable to resolve paper_id for {parsed_path}. "
        "Provide --paper-id or ensure papers.pdf_path matches."
    )


def _update_paper_doc_id(
    conn: sqlite3.Connection,
    paper_id: str,
    doc_id: str,
    pdf_path: Path | None,
) -> str | None:
    """Update the papers row with a doc_id, returning any prior doc_id."""

    row = conn.execute(
        "SELECT doc_id FROM papers WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"paper_id not found in papers table: {paper_id}")

    existing_doc_id = row[0] if row else None
    conn.execute(
        """
        UPDATE papers
        SET doc_id = ?, pdf_path = COALESCE(pdf_path, ?)
        WHERE paper_id = ?
        """,
        (doc_id, str(pdf_path) if pdf_path is not None else None, paper_id),
    )
    return existing_doc_id


def _ingest_chunks(
    conn: sqlite3.Connection,
    parsed_doc: ParsedDocument,
    paper_id: str,
    chunks: list[ChunkRecord],
) -> None:
    """Insert chunks into SQLite, replacing older versions."""

    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_papers_doc_id(conn)
    _ensure_chunks_schema(conn)

    existing_doc_id = _update_paper_doc_id(
        conn,
        paper_id=paper_id,
        doc_id=parsed_doc.doc_id,
        pdf_path=parsed_doc.pdf_path,
    )

    if existing_doc_id and existing_doc_id != parsed_doc.doc_id:
        removed = _delete_chunks_where(conn, "paper_id = ?", (paper_id,))
        LOGGER.info("Removed %s chunks for paper_id=%s", removed, paper_id)

    _delete_chunks_where(conn, "doc_id = ?", (parsed_doc.doc_id,))

    conn.executemany(
        """
        INSERT INTO chunks (
            paper_id,
            doc_id,
            page_number,
            chunk_index,
            text,
            char_start,
            char_end,
            token_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [chunk.as_row() for chunk in chunks],
    )


def _iter_parsed_paths(paths: list[Path]) -> list[Path]:
    """Expand parsed JSON paths from files or directories."""

    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.json")))
        else:
            expanded.append(path)
    return expanded


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for chunking."""

    parser = argparse.ArgumentParser(description="Chunk parsed PDFs into SQLite.")
    parser.add_argument(
        "--parsed",
        nargs="+",
        required=True,
        help="Parsed JSON file(s) or directory of parsed JSON files.",
    )
    parser.add_argument(
        "--db",
        default="data/arxiv_rag.db",
        help="SQLite DB path with papers metadata.",
    )
    parser.add_argument(
        "--paper-id",
        help="Override the paper_id for a single parsed JSON.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=512,
        help="Target tokens per chunk.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=100,
        help="Overlap tokens between chunks.",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="Tokenizer encoding name for chunk sizing.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the chunking CLI."""

    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    parsed_paths = _iter_parsed_paths([Path(p) for p in args.parsed])
    if not parsed_paths:
        print("No parsed JSON files found.", flush=True)
        return 1

    if args.paper_id and len(parsed_paths) > 1:
        print("--paper-id can only be used with a single parsed JSON.", flush=True)
        return 1

    config = ChunkConfig(
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        encoding_name=args.encoding,
    )

    db_path = Path(args.db)
    if not db_path.exists():
        print(
            f"Database not found: {db_path}. "
            "Run download.py with --db first.",
            flush=True,
        )
        return 1

    total_chunks = 0
    with sqlite3.connect(db_path) as conn:
        for parsed_path in parsed_paths:
            parsed_doc = load_parsed_document(parsed_path)
            paper_id = _resolve_paper_id(
                conn,
                parsed_doc,
                parsed_path,
                args.paper_id,
            )
            chunks = chunk_document(parsed_doc, paper_id, config)
            _ingest_chunks(conn, parsed_doc, paper_id, chunks)
            total_chunks += len(chunks)
            LOGGER.info(
                "Chunked %s -> %s chunks (paper_id=%s)",
                parsed_path.name,
                len(chunks),
                paper_id,
            )
        conn.commit()

    print(f"Inserted {total_chunks} chunks.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
