# arxiv-rag

CLI utilities for downloading arXiv PDFs, parsing text, chunking into SQLite FTS, and querying chunks.

## Contents

- Setup
- Quickstart
- CLI usage
- Data locations
- Current implementation
- Features
- Today's notes
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
uv run python -m arxiv_rag.cli index --db data/arxiv_rag.db
```

6. Query the BM25 index:

```bash
uv run python -m arxiv_rag.cli query "dense retrieval" --top-k 5
```

7. Query vector embeddings (Chroma):

```bash
uv run python -m arxiv_rag.cli query "dense retrieval" --top-k 5 --mode vector
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
uv run python -m arxiv_rag.cli query "dense retrieval" --top-k 5
```

Notes:

- FTS query building removes common stopwords.
- Terms length >= 6 are required (AND); shorter terms are optional (OR).

Fallback entrypoints:

```bash
uv run python -m arxiv_rag.cli query "dense retrieval" --db data/arxiv_rag.db --top-k 5
```

```bash
uv run python main.py query "dense retrieval" --db data/arxiv_rag.db --top-k 5
```

Include BM25 scores:

```bash
uv run python -m arxiv_rag.cli query "dense retrieval" --show-score
```

### Embed chunks with OpenAI

```bash
uv run python -m arxiv_rag.cli embed --db data/arxiv_rag.db
```

Note: this stores embeddings in SQLite for the legacy in-memory vector search path.

### Index embeddings into Chroma

```bash
uv run python -m arxiv_rag.cli index --db data/arxiv_rag.db
```

Notes:

- By default, indexing deletes existing vectors for the doc_ids being indexed.
- When `--limit` is set, deletes are skipped to avoid partial reindexing; use `--force-delete` to allow deletes.
- When `--doc-id` is provided, deletes occur even if no chunks exist for that doc_id.

### Add Papers By ID File (`--add-ids`)

Ingest additional papers end-to-end (download -> parse -> chunk -> index) from a file:

```bash
uv run python -m arxiv_rag.cli index --add-ids data/paper_ids.txt
```

```bash
uv run python -m arxiv_rag.cli index --add-ids data/paper_ids.json
```

Supported file formats:

- `.txt`: one ID per line, blank lines and `#` comments are ignored.
- `.json`: either `["2312.10997", "2401.00001"]` or `{"ids": ["2312.10997", "2401.00001"]}`.

Notes:

- IDs can include versions (`v2`); they are normalized to base IDs.
- The command writes parsed JSON files to `data/parsed/` by default.
- Failures are isolated per paper and reported at the end.

### Rebuild Local Corpus (`--rebuild`)

Re-parse local PDFs and rebuild chunk + vector indexes:

```bash
uv run python -m arxiv_rag.cli index --rebuild
```

Notes:

- Reads PDFs from `data/arxiv-papers/` by default.
- Useful after changing parse/chunk settings (for example `--remove-headers-footers`, `--target-tokens`).

### Query vector embeddings (Chroma)

```bash
uv run python -m arxiv_rag.cli query "dense retrieval" --mode vector --top-k 5
```

### Hybrid retrieval (RRF)

```bash
uv run python -m arxiv_rag.cli query "dense retrieval" --mode hybrid --top-k 5
```

### Generate answers with citations

```bash
uv run python -m arxiv_rag.cli query "How does dense retrieval work?" --mode hybrid --top-k 5 --generate
```

Verify citations and quotes in the generated answer:

```bash
uv run python -m arxiv_rag.cli query "How does dense retrieval work?" --mode hybrid --top-k 5 --generate --verify
```

Notes:

- `--generate` formats retrieved chunks into a citation prompt and returns a cited answer.
- Default generation model is `gpt-4o-mini`; override with `--generate-model`.
- `--verify` enforces paragraph-level citation coverage and quote matching against page text.

Verbose provenance (rank, raw score/distance, normalized rank score, RRF contribution):

```bash
uv run python -m arxiv_rag.cli query "dense retrieval" --mode hybrid --top-k 5 --verbose
```

Notes:

- Hybrid retrieval fuses FTS + vector results using RRF.
- If Chroma is unavailable or empty, it falls back to SQLite embeddings.

### Sources (hybrid + verbose alias)

```bash
uv run python -m arxiv_rag.cli sources "dense retrieval" --top-k 5
```

Notes:

- `sources` is an alias for `query --mode hybrid --verbose`.

### Evaluation

Generate a synthetic eval set (50 questions by default):

```bash
uv run python -m arxiv_rag.cli eval-generate --db data/arxiv_rag.db --output eval/eval_set.json
```

Run retrieval-only evaluation (Recall@5/10 + MRR):

```bash
uv run python -m arxiv_rag.cli eval-run --eval-set eval/eval_set.json --db data/arxiv_rag.db
```

Run evaluation with answer generation to compute chunk-level citation accuracy:

```bash
uv run python -m arxiv_rag.cli eval-run --eval-set eval/eval_set.json --db data/arxiv_rag.db --generate
```

Notes:

- Edit `eval/eval_set.json` manually to review or correct QA pairs before running eval.
- Citation accuracy is only computed when `--generate` is enabled.

### Inspect Chroma counts

```bash
uv run python -m arxiv_rag.cli inspect --db data/arxiv_rag.db
```

## Data locations

- PDFs: `data/arxiv-papers/`
- ID index: `data/arxiv-papers/arxiv_ids.txt`
- Parsed JSON: `data/parsed/`
- SQLite DB: `data/arxiv_rag.db`
- Chroma persistence: `data/chroma/`
- Eval set: `eval/eval_set.json`
- Eval reports: `eval/results/`

## Current implementation

- CLI-only pipeline: search/download arXiv PDFs, parse with PyMuPDF, chunk into SQLite, and retrieve via FTS/vector/hybrid modes.
- SQLite schema includes `papers`, `chunks`, and FTS5 triggers; `chunk_uid` is the stable join key across SQLite and Chroma.
- Embeddings use OpenAI (`text-embedding-3-small` default) with retries and token-aware batching; stored in SQLite for fallback retrieval.
- Vector search uses local Chroma collections; hybrid retrieval uses RRF fusion with optional fallback to SQLite embeddings.
- CLI supports answer generation via `--generate` and deterministic citation/quote verification via `--verify`.
- Citation prompt templates live under `arxiv_rag/prompts/` and are bundled with the package.

## Features

- Search arXiv or download by base ID; track downloaded IDs in `data/arxiv-papers/arxiv_ids.txt`.
- Metadata ingestion into SQLite during download, plus `--backfill-db` for existing PDFs.
- PDF parsing with cleaning and optional repeated header/footer removal.
- Token-based chunking with overlap, character offsets, and version-aware replacement when doc_id changes.
- Retrieval modes: `fts`, `vector` (Chroma), `hybrid` (RRF) with optional scores and verbose provenance.
- Chroma utilities: index embeddings, inspect per-doc_id counts, and delete-by-doc_id on re-index.

## Today's notes

### 2026-02-03

- Vector mode relies on Chroma; hybrid mode can fall back to SQLite embeddings if Chroma is missing or empty.
- `chunk_uid` is now the canonical cross-index join key; chunk backfills keep older rows consistent.
- Query output supports retrieval-only snippets or `--generate` for cited answers; `--verify` validates citations and quotes.
- Query logging is now stored in the SQLite `query_log` table.
- Eval utilities include `eval-generate` and `eval-run` for Recall@K/MRR and optional citation accuracy.
- Spec updates: Phase 3 acceptance criteria now define citation validation, `--verify` behavior, and optional LLM-judge schema.

## Development notes

- arXiv ID parsing helpers live in `arxiv_rag/arxiv_ids.py`.
- SQLite schema helpers live in `arxiv_rag/db.py`.

## Testing

```bash
uv run pytest
```
