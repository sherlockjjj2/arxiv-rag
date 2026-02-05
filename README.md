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

Set `OPENAI_API_KEY` (for example in `.env`) before running OpenAI-backed commands (`embed`, `index`, `query --mode vector|hybrid`, `query --generate`, `eval-generate`, `eval-run --generate`).

## Quickstart

1. Search arXiv for IDs (no downloads):

```bash
uv run python arxiv_rag/download.py --query "keyword phrase" --max-results 25 --sort relevance
```

If requests appear to hang, lower API retry/timeout:

```bash
uv run python arxiv_rag/download.py --query "keyword phrase" --max-results 25 --sort relevance --api-retries 1 --api-timeout 10
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

Requires `uv sync --extra vector` and `OPENAI_API_KEY`.

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

Troubleshooting:

- If the command is slow or seems stuck, use `--api-retries 1 --api-timeout 10` to fail fast.
- Check proxy env vars with `env | rg -i proxy`; stale proxy settings (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`) can block arXiv requests.

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

Notes:

- MuPDF parser diagnostics are muted during parsing to keep progress output clean.
- Process-level stderr is temporarily muted during parse extraction to suppress low-level MuPDF diagnostics that bypass Python logging.
- `parse.py` suppresses low-risk MuPDF warnings (font-metric / graphics-structure noise) and emits compact summaries only for higher-risk warnings.

### Chunk parsed JSON into SQLite

```bash
uv run python arxiv_rag/chunk.py --parsed data/parsed/2505.09388v1.json --db data/arxiv_rag.db
```

```bash
uv run python arxiv_rag/chunk.py --parsed data/parsed --db data/arxiv_rag.db
```

`--parsed` only accepts parsed `.json` files (or directories containing `.json` files).  
If you only have a PDF, run `parse.py` first.
- Chunk text preserves Unicode symbols (for example ligatures and math symbols)
  to avoid replacement characters (`�`) in retrieval output.

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
- arXiv client INFO logs are suppressed by default during `index --add-ids` to keep output concise; warnings/errors are still shown.
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
- Use `--generation-rerank lexical` to rerank retrieved chunks before generation (lexical overlap + original rank).
- Use `--generation-select-evidence` to run a lightweight evidence-selection step before answering; tune with `--generation-select-k` (default 3). This adds an extra model call.
- Use `--generation-quote-first` to force quote-first generation anchored to a single chunk (adds an extra model call).
- Use `--generation-cite-chunk-index` to emit `[chunk:N]` citations and map them to arXiv page citations post-generation.
- Use `--generation-repair-citations` to repair missing or malformed citations; tune with `--generation-repair-max-attempts` (default 1). This adds extra model calls when triggered.
- Generated answers apply a citation post-processing pass that remaps each citation to the best-supported retrieved `(paper_id, page_number)` using quote overlap.
- `--verify` requires `--generate` and validates the generated answer only.
- Default generation model is `gpt-4o-mini`; override with `--generate-model`.
- `--verify` enforces paragraph-level citation coverage and quote matching against page text.

Verbose provenance (rank, raw score/distance, normalized rank score, RRF contribution):

```bash
uv run python -m arxiv_rag.cli query "dense retrieval" --mode hybrid --top-k 5 --verbose
```

Notes:

- Hybrid retrieval fuses FTS + vector results using RRF.
- If Chroma is unavailable or empty, it falls back to SQLite embeddings.
- FTS queries require only a small set of long terms and fall back to an OR-only
  query when the strict query returns no hits.

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
uv run python -m arxiv_rag.cli eval-run --eval-set-path eval/eval_set.json --db data/arxiv_rag.db
```

Run evaluation with answer generation to compute chunk-level citation accuracy:

```bash
uv run python -m arxiv_rag.cli eval-run --eval-set-path eval/eval_set.json --db data/arxiv_rag.db --generate
```

Notes:

- Edit `eval/eval_set.json` manually to review or correct QA pairs before running eval.
- If you correct eval items manually, ensure ground-truth chunk IDs align with the cited page text (for example, q010/q015/q040 fixes on 2026-02-05).
- `eval-generate` now auto-filters low-quality QA pairs (for example, questions that reference "the excerpt" or answers weakly grounded in chunk text).
- `eval-generate` skips reference-like chunks (bibliography-heavy pages with citations/URLs) to avoid unanswerable QA pairs.
- If you want stricter selection with more headroom, increase `--questions-per-chunk` (for example `2` or `3`) and keep `--n-questions` fixed.
- Use `--temperature` and `--top-p` to adjust QA creativity (defaults `0.8` and `0.9`).
- `eval-generate` enforces a balanced list of question openings by default; override with `--openings-path` (JSON list or one opening per line).
- Custom eval QA prompt templates must include `$OPENINGS` when openings are enforced (default).
- Citation accuracy is only computed when `--generate` is enabled.
- Eval summary includes `citation_accuracy_when_recall5_hit` to separate retrieval misses from citation selection issues.
- Failure modes distinguish `citation_absent` (no parsed citations) from `citation_zero_score` (citations present but none matched ground truth).
- Eval reports include generation-context diagnostics for missing ground-truth chunks or citations outside the provided context.
- Use `--generation-rerank lexical` with `eval-run --generate` to rerank chunks before generation.
- Use `--generation-select-evidence` with `eval-run --generate` to force a smaller, model-selected evidence set; tune with `--generation-select-k`.
- Evidence selection always includes the top-ranked chunk as a guardrail (so `--generation-select-k 1` effectively keeps chunk 1).
- Use `--generation-quote-first` with `eval-run --generate` to anchor answers to a single quoted chunk.
- Use `--generation-cite-chunk-index` with `eval-run --generate` to map chunk-index citations to page-level citations.
- Use `--generation-repair-citations` to repair missing/malformed citations during eval generation.
- Generation applies a fallback remap that uses sentence-level overlap to realign citations when quotes are missing.
- Generation for `eval-run --generate` is concurrent by default (`--generation-concurrency 4`) to reduce wall-clock latency.
- `eval-run` now uses a persistent SQLite cache by default at `eval/cache/eval_cache.db` for query embeddings and generated answers.
- Use `--disable-cache` to force fresh API calls, or `--cache-db <path>` to override the cache location.

### Inspect Chroma counts

```bash
uv run python -m arxiv_rag.cli inspect --db data/arxiv_rag.db
```

### Quick corpus stats (SQLite)

There is no dedicated `stats` CLI command yet. Use SQLite directly:

```bash
sqlite3 data/arxiv_rag.db "SELECT COUNT(*) AS papers FROM papers; SELECT COUNT(*) AS chunks FROM chunks;"
```

**Learning 2026-02-05**
- Retrieval was not the bottleneck; most errors were citation grounding or eval-set quality.
- Eval-set quality matters: reference-like chunks produced unanswerable QA. Added reference-chunk filtering in `eval-generate` and fixed ground-truth mismatches (q010/q015/q040).
- Evidence selection guardrail (always include top-ranked chunk) helped, but `--generation-select-k 1` was too strict; `--generation-select-k 2` was a better tradeoff.
- Best config so far (with the refreshed eval set): `--generation-top-k 10` + `--generation-rerank lexical` + `--generation-select-evidence` + `--generation-select-k 2` + `--generation-repair-citations` reached citation accuracy `0.91` (Recall@5 `0.92`).
- Tightened citation grounding by forbidding the literal *"quote"* placeholder and adding sentence-overlap remapping when quotes are missing.
- `--generation-cite-chunk-index` regressed in this eval set; keep it off for now.

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

### 2026-02-04

- Query error handling was tightened: expected retrieval failures now print clean messages with no traceback.
- Exit code behavior is now consistent for query paths: `1` for user/input issues, `2` for system/runtime issues.
- CLI docs were synced to current flags and behavior (`--eval-set-path`, `--verify` requires `--generate`).
- Added a quick SQLite-based corpus stats command since there is no dedicated `stats` subcommand yet.

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
- Hybrid RRF fusion uses a dedicated accumulator dataclass for type-safe provenance aggregation.

## Testing

```bash
uv run pytest
```
