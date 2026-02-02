# arxiv-rag

A small CLI for searching arXiv and downloading PDFs by explicit ID.

## Setup

Install dependencies:

```bash
uv sync
```

## Usage

Search only (no downloads):

```bash
python3 arxiv-rag/download.py --query "keyword phrase" --max-results 25 --sort relevance
```

Download by ID (base IDs only, no version suffix):

```bash
python3 arxiv-rag/download.py --ids 2301.12345 2310.00001
```

Downloads go to `data/arxiv-papers/` and are named with the versioned arXiv ID (e.g., `2301.12345v2.pdf`). The base ID is recorded in `data/arxiv-papers/arxiv_ids.txt`.

Metadata ingestion (SQLite):

```bash
python3 arxiv-rag/download.py --ids 2301.12345 --db data/arxiv_rag.db
```

Skip metadata ingestion:

```bash
python3 arxiv-rag/download.py --ids 2301.12345 --no-db
```

Backfill metadata for existing PDFs:

```bash
python3 arxiv-rag/download.py --backfill-db
```

## Testing

```bash
pytest
```
