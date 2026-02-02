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
uv run python arxiv_rag/download.py --query "keyword phrase" --max-results 25 --sort relevance
```

Download by ID (base IDs only, no version suffix):

```bash
uv run python arxiv_rag/download.py --ids 2301.12345 2310.00001
```

Downloads go to `data/arxiv-papers/` and are named with the versioned arXiv ID (e.g., `2301.12345v2.pdf`). The base ID is recorded in `data/arxiv-papers/arxiv_ids.txt`.

Metadata ingestion (SQLite):

```bash
uv run python arxiv_rag/download.py --ids 2301.12345 --db data/arxiv_rag.db
```

Metadata ingestion uses an upsert, so re-downloading a paper refreshes title/authors/pdf_path.

Skip metadata ingestion:

```bash
uv run python arxiv_rag/download.py --ids 2301.12345 --no-db
```

Backfill metadata for existing PDFs:

```bash
uv run python arxiv_rag/download.py --backfill-db
```

Parse a PDF (text layer only, defaults to `data/parsed/<pdf_stem>.json`):

```bash
uv run python arxiv_rag/parse.py --pdf data/arxiv-papers/2505.09388v1.pdf
```

Optional header/footer removal:

```bash
uv run python arxiv_rag/parse.py --pdf data/arxiv-papers/2505.09388v1.pdf --remove-headers-footers
```

Chunk parsed JSON into SQLite (requires metadata in `papers` table):

```bash
uv run python arxiv_rag/chunk.py --parsed data/parsed/2505.09388v1.json --db data/arxiv_rag.db
```

Chunk an entire directory of parsed JSON files:

```bash
uv run python arxiv_rag/chunk.py --parsed data/parsed --db data/arxiv_rag.db
```

When a paper is re-parsed with a new PDF version, existing chunks for that `paper_id`
are deleted and replaced with the latest version.

Query the BM25 index (FTS5) for matching chunks:

```bash
uv run arxiv-rag query "dense retrieval" --top-k 5
```

If the console script is not available yet, run via module or main:

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

## Testing

```bash
uv run pytest
```
