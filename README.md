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

Optional vector dependencies (Chroma):

```bash
uv sync --extra vector
```

Note: Chroma depends on `onnxruntime` wheels; Python 3.13 is recommended (3.14 may not be supported yet).

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

5. Index embeddings into Chroma:

```bash
uv run arxiv-rag index --db data/arxiv_rag.db
```

6. Query the BM25 index:

```bash
uv run arxiv-rag query "dense retrieval" --top-k 5
```

7. Query vector embeddings (Chroma):

```bash
uv run arxiv-rag query "dense retrieval" --top-k 5 --mode vector
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

### Embed chunks with OpenAI

```bash
uv run arxiv-rag embed --db data/arxiv_rag.db
```

Note: this stores embeddings in SQLite for the legacy in-memory vector search path.

### Index embeddings into Chroma

```bash
uv run arxiv-rag index --db data/arxiv_rag.db
```

Notes:

- By default, indexing deletes existing vectors for the doc_ids being indexed.
- When `--limit` is set, deletes are skipped to avoid partial reindexing; use `--force-delete` to allow deletes.
- When `--doc-id` is provided, deletes occur even if no chunks exist for that doc_id.

### Query vector embeddings (Chroma)

```bash
uv run arxiv-rag query "dense retrieval" --mode vector --top-k 5
```

### Inspect Chroma counts

```bash
uv run arxiv-rag inspect --db data/arxiv_rag.db
```

## Data locations

- PDFs: `data/arxiv-papers/`
- ID index: `data/arxiv-papers/arxiv_ids.txt`
- Parsed JSON: `data/parsed/`
- SQLite DB: `data/arxiv_rag.db`
- Chroma persistence: `data/chroma/`

## Notes

### 2026-02-03

- Current implementation is CLI-only: download/search arXiv, parse PDFs, chunk into SQLite, and query via FTS5 (BM25).
- Embeddings are generated and indexed into Chroma for vector search; SQLite remains the source of chunk text.
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
