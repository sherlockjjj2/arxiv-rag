# arxiv-rag

A small CLI for searching arXiv and downloading PDFs by explicit ID.

## Setup

Install dependencies (creates a local `.venv`):

```bash
uv sync
```

## Usage

Search only (no downloads):

```bash
uv run python arxiv-rag/download.py --query "keyword phrase" --max-results 25 --sort relevance
```

Download by ID (base IDs only, no version suffix):

```bash
uv run python arxiv-rag/download.py --ids 2301.12345 2310.00001
```

Downloads go to `data/arxiv-papers/` and are named with the versioned arXiv ID (e.g., `2301.12345v2.pdf`). The base ID is recorded in `data/arxiv-papers/arxiv_ids.txt`.

Metadata ingestion (SQLite):

```bash
uv run python arxiv-rag/download.py --ids 2301.12345 --db data/arxiv_rag.db
```

Skip metadata ingestion:

```bash
uv run python arxiv-rag/download.py --ids 2301.12345 --no-db
```

Backfill metadata for existing PDFs:

```bash
uv run python arxiv-rag/download.py --backfill-db
```

Parse a PDF (text layer only, defaults to `data/parsed/<pdf_stem>.json`):

```bash
uv run python arxiv-rag/parse.py --pdf data/arxiv-papers/2505.09388v1.pdf
```

Optional header/footer removal:

```bash
uv run python arxiv-rag/parse.py --pdf data/arxiv-papers/2505.09388v1.pdf --remove-headers-footers
```

## Testing

```bash
uv run pytest
```
