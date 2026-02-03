# arxiv-rag

CLI utilities for downloading arXiv PDFs, parsing text, chunking into SQLite FTS, and querying chunks.

## Contents

- Setup
- Quickstart
- CLI usage
- Data locations
- Development notes
- Testing

## Setup

Install dependencies (creates a local `.venv`):

```bash
uv sync
```

## Quickstart

1. Search arXiv for IDs (no downloads):

```bash
uv run python arxiv_rag/download.py --query "keyword phrase" --max-results 25 --sort relevance
```

2. Download PDFs by base ID:

```bash
uv run python arxiv_rag/download.py --ids 2301.12345 2310.00001
```

3. Parse a PDF into per-page JSON:

```bash
uv run python arxiv_rag/parse.py --pdf data/arxiv-papers/2301.12345v1.pdf
```

4. Chunk parsed JSON into SQLite:

```bash
uv run python arxiv_rag/chunk.py --parsed data/parsed/2301.12345v1.json --db data/arxiv_rag.db
```

5. Query the BM25 index:

```bash
uv run arxiv-rag query "dense retrieval" --top-k 5
```

## CLI usage

### Search only (no downloads)

```bash
uv run python arxiv_rag/download.py --query "keyword phrase" --max-results 25 --sort relevance
```

### Download PDFs by ID

```bash
uv run python arxiv_rag/download.py --ids 2301.12345 2310.00001
```

Notes:

- Use base IDs only (no version suffix).
- PDFs are saved under `data/arxiv-papers/` with versioned filenames.
- Base IDs are recorded in `data/arxiv-papers/arxiv_ids.txt`.

### Metadata ingestion (SQLite)

```bash
uv run python arxiv_rag/download.py --ids 2301.12345 --db data/arxiv_rag.db
```

```bash
uv run python arxiv_rag/download.py --ids 2301.12345 --no-db
```

```bash
uv run python arxiv_rag/download.py --backfill-db
```

### Parse PDFs

```bash
uv run python arxiv_rag/parse.py --pdf data/arxiv-papers/2505.09388v1.pdf
```

```bash
uv run python arxiv_rag/parse.py --pdf data/arxiv-papers/2505.09388v1.pdf --remove-headers-footers
```

### Chunk parsed JSON into SQLite

```bash
uv run python arxiv_rag/chunk.py --parsed data/parsed/2505.09388v1.json --db data/arxiv_rag.db
```

```bash
uv run python arxiv_rag/chunk.py --parsed data/parsed --db data/arxiv_rag.db
```

When a paper is re-parsed with a new PDF version, existing chunks for that `paper_id`
are deleted and replaced with the latest version.

### Query the BM25 index

```bash
uv run arxiv-rag query "dense retrieval" --top-k 5
```

Fallback entrypoints:

```bash
uv run python -m arxiv_rag.cli query "dense retrieval" --db data/arxiv_rag.db --top-k 5
```

```bash
uv run python main.py query "dense retrieval" --db data/arxiv_rag.db --top-k 5
```

Include BM25 scores:

```bash
uv run arxiv-rag query "dense retrieval" --show-score
```

## Data locations

- PDFs: `data/arxiv-papers/`
- ID index: `data/arxiv-papers/arxiv_ids.txt`
- Parsed JSON: `data/parsed/`
- SQLite DB: `data/arxiv_rag.db`

## Notes

### 2026-02-22

- Current implementation is CLI-only: download/search arXiv, parse PDFs, chunk into SQLite, and query via FTS5 (BM25).
- Embeddings are not generated or used; no vector DB or hybrid retrieval is implemented.
- No LLM answer generation or citation verification is implemented.
- Query logging is not implemented.
- Spec now defines a canonical `chunk_uid` for cross-index joins (SQLite ↔ vector DB).

## Development notes

- arXiv ID parsing helpers live in `arxiv_rag/arxiv_ids.py`.
- SQLite schema helpers live in `arxiv_rag/db.py`.

## Testing

```bash
uv run pytest
```
