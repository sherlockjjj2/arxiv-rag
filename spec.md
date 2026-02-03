# arXiv RAG MVP — Technical Specification

**Version**: 1.0  
**Author**: Joseph Fang  
**Status**: Draft  
**Last Updated**: 2026-02-03

---

## 1. Project Overview

### Goal

Build a CLI tool that answers research questions using arXiv papers with verifiable, page-level citations.

### Success Criteria

- Query → answer in <10 seconds for 200-paper corpus
- Every claim includes citation: `[arXiv:2301.01234 p.7]`
- Every citation includes supporting quote snippet
- Retrieval quality: relevant chunks in top-5 for 80%+ of test queries

### Non-Goals (MVP)

- Web UI
- Incremental index updates
- Section-aware parsing
- Multi-modal (figures, tables)
- Private PDF support (design for it, don't build)

---

## 2. Data Source: arXiv

### API Options

| Method               | Use Case                         | Rate Limit        |
| -------------------- | -------------------------------- | ----------------- |
| arXiv API (OAI-PMH)  | Bulk metadata by category/date   | 1 req/3 sec       |
| arXiv Search API     | Keyword search, specific IDs     | 1 req/3 sec       |
| Semantic Scholar API | Citation graphs, abstracts       | 100 req/5 min     |
| Direct PDF download  | `https://arxiv.org/pdf/{id}.pdf` | Be polite (1/sec) |

### Recommended Approach

```python
# arxiv Python package (wrapper around arXiv API)
pip install arxiv

import arxiv

# Option 1: Search by keywords
search = arxiv.Search(
    query="RAG retrieval augmented generation",
    max_results=50,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

# Option 2: Fetch specific IDs
search = arxiv.Search(id_list=["2312.10997", "2305.14283"])

# Download PDFs
for paper in arxiv.Client().results(search):
    paper.download_pdf(dirpath="./papers", filename=f"{paper.get_short_id()}.pdf")
```

### Metadata to Capture

```json
{
  "arxiv_id": "2312.10997",
  "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
  "authors": ["Yunfan Gao", "..."],
  "abstract": "...",
  "categories": ["cs.CL", "cs.AI"],
  "published": "2023-12-18",
  "updated": "2024-01-04",
  "pdf_url": "https://arxiv.org/pdf/2312.10997.pdf",
  "primary_category": "cs.CL"
}
```

### Corpus Selection Strategy (for MVP)

Start with a focused topic to make evaluation easier:

1. **Seed papers** (5-10): Manually select landmark papers you know well
2. **Citation expansion**: Use Semantic Scholar to get papers that cite or are cited by seeds
3. **Keyword search**: `"retrieval augmented generation" OR "RAG" OR "dense retrieval"`
4. **Date filter**: 2023-01-01 to present (keeps corpus current)
5. **Cap at 200 papers** for MVP

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           CLI                                    │
│  query "How does RAG handle long contexts?" --top-k 5 --verify  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Hybrid Retriever                            │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │   SQLite FTS5   │              │   ChromaDB      │           │
│  │   (BM25-ish)    │              │   (Vectors)     │           │
│  └────────┬────────┘              └────────┬────────┘           │
│           │         Reciprocal Rank        │                    │
│           └──────────► Fusion ◄────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Answer Generator                            │
│  System: Use ONLY the provided chunks. Cite every claim.        │
│  Output: Answer + citations [arXiv:ID p.N] + quote snippets     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Database Schema

### SQLite Tables

```sql
-- Papers metadata
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,      -- arXiv ID: "2312.10997"
    doc_id TEXT,                    -- SHA1 of PDF bytes (set after parsing)
    title TEXT NOT NULL,
    authors TEXT,                   -- JSON array
    abstract TEXT,
    categories TEXT,                -- JSON array
    published_date TEXT,
    pdf_path TEXT,
    total_pages INTEGER,
    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    source_type TEXT DEFAULT 'arxiv', -- For future: 'arxiv', 'local', 'url'
    UNIQUE(doc_id)
);

-- Chunks with page-level granularity
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_uid TEXT NOT NULL UNIQUE, -- Stable ID for cross-index joins
    paper_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,   -- 0, 1, 2... within page
    text TEXT NOT NULL,
    char_start INTEGER,             -- Character offset in page
    char_end INTEGER,
    token_count INTEGER,
    embedding BLOB,                 -- Store embedding as numpy bytes (optional)
    UNIQUE(doc_id, page_number, chunk_index),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

-- Full-text search index
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='chunk_id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.chunk_id, old.text);
END;

CREATE TRIGGER chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.chunk_id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;

### Canonical Chunk Schema (Logical)

Use a stable `chunk_uid` as the **cross-index join key** (SQLite ↔ Chroma ↔ FAISS).
`chunk_id` remains a SQLite-internal row identifier only.

Recommended fields:

```json
{
  "chunk_uid": "sha1(doc_id:page_number:chunk_index:char_start:char_end)",
  "paper_id": "2312.10997",
  "doc_id": "b1946ac92492d2347c6235b4d2611184",
  "page_number": 5,
  "chunk_index": 2,
  "text": "chunk text...",
  "char_start": 1203,
  "char_end": 1875,
  "token_count": 512
}
```

Chunk UID options + trade-offs:

- **Readable key**: `"{doc_id}_p{page_number}_c{chunk_index}"`  
  Easy to debug, but long; collisions if `doc_id` is truncated.
- **Hashed key (recommended)**: `sha1("{doc_id}:{page_number}:{chunk_index}:{char_start}:{char_end}")`  
  Fixed length and stable; harder to eyeball but safe for joins across indexes.

-- Query log for evaluation
CREATE TABLE query_log (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    retrieved_chunks TEXT,          -- JSON array of chunk_uids
    answer TEXT,
    latency_ms INTEGER,
    model TEXT,
    feedback INTEGER                -- 1=good, 0=bad, NULL=not rated
);
```

### ChromaDB Collection

```python
# Separate from SQLite, just for vectors
collection = chroma_client.create_collection(
    name="arxiv_chunks",
    metadata={"hnsw:space": "cosine"}
)

# Each document stores:
{
    "id": "sha1(doc_id:page:chunk_index:char_start:char_end)",  # chunk_uid
    "embedding": [...],              # from OpenAI
    "metadata": {
        "chunk_uid": "sha1(doc_id:page:chunk_index:char_start:char_end)",
        "paper_id": "2312.10997",
        "doc_id": "b1946ac92492d2347c6235b4d2611184",
        "page": 5,
        "chunk_index": 2,
        "title": "RAG Survey"        # Denormalized for display
    },
    "document": "chunk text..."      # Optional, can fetch from SQLite
}
```

### MVP Embeddings (SQLite + In-Memory Cosine)

- Embeddings are generated with `text-embedding-3-small`.
- Vectors are stored in `chunks.embedding` as float32 bytes (normalized).
- Vector retrieval loads embeddings from SQLite and computes cosine similarity in memory.
- This keeps the MVP dependency footprint small while remaining compatible with future vector DBs.

---

## 5. Processing Pipeline

### 5.1 PDF Parsing

```python
# Use PyMuPDF (fitz) - fast, good text extraction
import fitz  # pip install pymupdf

def extract_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        # Skip if too little text (likely figure-only page)
        if len(text.strip()) < 100:
            continue
        pages.append({
            "page": page_num,
            "text": text,
            "char_count": len(text)
        })
    return pages
```

#### Current parse flow (implemented)

```
PDF path
  │
  ├─ validate exists/is_file
  ├─ compute doc_id = sha1(file bytes)
  ├─ open PDF with PyMuPDF
  ├─ for each page:
  │    ├─ page.get_text("text")
  │    ├─ clean text (join hyphen breaks, normalize whitespace, drop empty lines)
  │    └─ on error: warn + skip page
  ├─ optional: remove repeated headers/footers
  │    ├─ collect first/last line per page (post-clean)
  │    └─ remove lines repeating on >=60% of pages and <=120 chars
  └─ emit JSON: doc_id, pdf_path, num_pages, pages[{page, text}]
```

#### Ingestion/versioning flow (planned)

```
Parse JSON (doc_id, pages)
  │
  ├─ fetch arXiv metadata -> paper_id, title, authors, etc.
  ├─ upsert papers(paper_id, doc_id, ...)
  │    ├─ update pdf_path/metadata on re-download
  │    └─ if paper_id exists with different doc_id: delete old chunks/vectors
  ├─ chunk pages using (paper_id, doc_id)
  ├─ compute chunk_uid for each chunk
  ├─ insert chunks (unique on chunk_uid; also unique on doc_id + page_number + chunk_index)
  └─ insert vectors with id = chunk_uid
```

## Design decisions\*\*

- Store both `paper_id` (human-facing arXiv ID) and `doc_id` (SHA1 of PDF bytes).
- Use `doc_id` for versioning/dedup; if a `paper_id` is re-downloaded with a new
  `doc_id`, replace the prior record (delete old chunks/vectors) unless versioning
  is added later.
- Metadata ingestion is an upsert; new downloads refresh title/authors/pdf_path.
- Per-page error isolation; parsing continues even if a page fails.
- Text-layer only; no OCR, no chunking.
- Header/footer removal is optional to avoid dropping real section headers.

### 5.2 Chunking Strategy

```python
from tiktoken import get_encoding
import hashlib

def chunk_page(
    page_text: str,
    page_num: int,
    paper_id: str,
    doc_id: str,
    target_tokens: int = 512,
    overlap_tokens: int = 100
) -> list[dict]:
    """
    Chunk by tokens with overlap. Page-aware.
    Baseline: 512 tokens with 100 overlap (MVP default).
    """
    enc = get_encoding("cl100k_base")
    tokens = enc.encode(page_text)

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        # Calculate character offsets (approximate)
        char_start = len(enc.decode(tokens[:start]))
        char_end = char_start + len(chunk_text)
        chunk_uid_source = f"{doc_id}:{page_num}:{chunk_idx}:{char_start}:{char_end}"
        chunk_uid = hashlib.sha1(chunk_uid_source.encode("utf-8")).hexdigest()

        chunks.append({
            "chunk_uid": chunk_uid,
            "paper_id": paper_id,
            "doc_id": doc_id,
            "page_number": page_num,
            "chunk_index": chunk_idx,
            "text": chunk_text,
            "char_start": char_start,
            "char_end": char_end,
            "token_count": len(chunk_tokens)
        })

        # Move forward with overlap
        start = end - overlap_tokens if end < len(tokens) else end
        chunk_idx += 1

    return chunks
```

### 5.3 Embedding

```python
from openai import OpenAI

client = OpenAI()

def embed_chunks(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Batch embed. Max 2048 inputs per call."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]

# Cost estimate for 200 papers:
# ~50 pages/paper × 200 papers = 10,000 pages
# ~2 chunks/page × 512 tokens = ~10M tokens
# text-embedding-3-small: $0.02/1M tokens = ~$0.20 total
```

---

## 6. Retrieval

### 6.1 Hybrid Search with Reciprocal Rank Fusion

```python
def hybrid_search(
    query: str,
    top_k: int = 10,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5
) -> list[dict]:
    """
    Combine BM25 and vector search using RRF.
    """
    # BM25 via SQLite FTS5
    bm25_results = search_fts(query, top_k=top_k * 2)

    # Vector search via ChromaDB
    query_embedding = embed_chunks([query])[0]
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2
    )

    # Reciprocal Rank Fusion
    # Score = sum(1 / (k + rank)) across methods
    k = 60  # RRF constant
    scores = {}

    for rank, chunk_uid in enumerate(bm25_results):
        scores[chunk_uid] = scores.get(chunk_uid, 0) + bm25_weight / (k + rank + 1)

    for rank, chunk_uid in enumerate(vector_results['ids'][0]):
        scores[chunk_uid] = scores.get(chunk_uid, 0) + vector_weight / (k + rank + 1)

    # Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Fetch full chunk data
    return [get_chunk_with_metadata(chunk_uid) for chunk_uid, _ in ranked]
```

### 6.2 FTS5 Search

```python
def search_fts(query: str, top_k: int = 20) -> list[str]:
    """
    BM25-style search using SQLite FTS5.
    Returns chunk_uids ordered by relevance.
    """
    # FTS5 uses BM25 by default
    sql = """
        SELECT chunks.chunk_uid, bm25(chunks_fts) as score
        FROM chunks_fts
        JOIN chunks ON chunks_fts.rowid = chunks.chunk_id
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """
    # FTS5 query syntax: use AND between terms
    fts_query = ' AND '.join(query.split())

    cursor.execute(sql, (fts_query, top_k))
    return [row[0] for row in cursor.fetchall()]
```

---

## 7. Answer Generation

### System Prompt

```python
SYSTEM_PROMPT = """You are a research assistant answering questions using ONLY the provided paper excerpts.

STRICT RULES:
1. Use ONLY information from the provided chunks. Do not use prior knowledge.
2. EVERY factual claim must have a citation in format: [arXiv:PAPER_ID p.PAGE]
3. After each citation, include a brief supporting quote in italics
4. If the chunks don't contain enough information, say so explicitly
5. If chunks conflict, note the disagreement and cite both sources

CITATION FORMAT EXAMPLE:
Dense retrieval models encode queries and documents into dense vectors [arXiv:2312.10997 p.3] *"BERT-based encoders map text to 768-dimensional vectors"*

CHUNKS:
{chunks}

Answer the following question using only the above chunks:"""
```

### Generation Function

```python
def generate_answer(query: str, chunks: list[dict], model: str = "gpt-4o-mini") -> str:
    """
    Generate answer with citations from retrieved chunks.
    """
    # Format chunks for context
    chunks_text = "\n\n".join([
        f"[arXiv:{c['paper_id']} p.{c['page_number']}] {c['title']}\n{c['text']}"
        for c in chunks
    ])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(chunks=chunks_text)},
            {"role": "user", "content": query}
        ],
        temperature=0.3,  # Lower for factual accuracy
        max_tokens=1000
    )

    return response.choices[0].message.content
```

---

## 8. CLI Interface

```bash
# Basic query
arxiv-rag query "What are the main approaches to dense retrieval?"

# With options
arxiv-rag query "How does RAG handle hallucination?" \
    --top-k 10 \
    --model gpt-4o \
    --verbose

# Sources only (no generation)
arxiv-rag sources "contrastive learning for embeddings" --top-k 5

# Verify mode: show chunks + open PDF
arxiv-rag query "What is the FAISS index?" --verify

# Index management
arxiv-rag index --papers ./papers/ --rebuild
arxiv-rag index --add-ids 2312.10997 2401.00001

# Stats
arxiv-rag stats
# Output: 200 papers, 15,234 chunks, index size: 45MB
```

### CLI Implementation Sketch

```python
# cli.py
import typer
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer()
console = Console()

@app.command()
def query(
    question: str,
    top_k: int = typer.Option(5, help="Number of chunks to retrieve"),
    model: str = typer.Option("gpt-4o-mini", help="Model for generation"),
    verify: bool = typer.Option(False, help="Show sources and open PDF"),
    verbose: bool = typer.Option(False, help="Show retrieval details")
):
    """Ask a question about the indexed papers."""

    # Retrieve
    chunks = hybrid_search(question, top_k=top_k)

    if verbose:
        console.print("[bold]Retrieved chunks:[/bold]")
        for c in chunks:
            console.print(f"  • [arXiv:{c['paper_id']} p.{c['page_number']}] {c['title'][:50]}...")

    # Generate
    answer = generate_answer(question, chunks, model=model)
    console.print(Markdown(answer))

    # Verify mode
    if verify:
        console.print("\n[bold]Source passages:[/bold]")
        for c in chunks:
            console.print(f"\n[arXiv:{c['paper_id']} p.{c['page_number']}]")
            console.print(c['text'][:500] + "...")

        # Open first PDF at relevant page
        if chunks:
            open_pdf_at_page(chunks[0]['pdf_path'], chunks[0]['page_number'])

@app.command()
def sources(question: str, top_k: int = 5):
    """Find relevant passages without generating an answer."""
    chunks = hybrid_search(question, top_k=top_k)
    for c in chunks:
        console.print(f"\n[bold][arXiv:{c['paper_id']} p.{c['page_number']}][/bold] {c['title']}")
        console.print(c['text'][:300] + "...")
```

---

## 9. Evaluation Dataset

### Creating Ground Truth

You need query-answer pairs with known relevant passages. Three approaches:

#### Approach 1: Synthetic from Papers (Recommended for MVP)

Use an LLM to generate questions from paper sections:

```python
def generate_qa_pairs(chunk: dict, n_questions: int = 2) -> list[dict]:
    """Generate questions that this chunk should answer."""

    prompt = f"""Given this excerpt from a research paper, generate {n_questions} questions that:
1. Can be answered using ONLY this text
2. Are specific enough that this chunk is clearly the best answer
3. Vary in complexity (1 factual, 1 requiring synthesis)

Paper: {chunk['title']}
Excerpt:
{chunk['text']}

Output as JSON: [{{"question": "...", "expected_answer": "...", "difficulty": "factual|synthesis"}}]"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    qa_pairs = json.loads(response.choices[0].message.content)

    # Add ground truth chunk reference
    for qa in qa_pairs:
        qa['ground_truth_chunk_uid'] = chunk['chunk_uid']
        qa['ground_truth_paper'] = chunk['paper_id']
        qa['ground_truth_page'] = chunk['page_number']

    return qa_pairs
```

#### Approach 2: Manual Curation (Highest Quality)

1. Read 10-20 papers thoroughly
2. Write questions you actually want answered
3. Note which papers/pages contain the answer
4. Time-consuming but catches edge cases

#### Approach 3: Real Query Logs (Post-Launch)

Use `--feedback` flag to mark good/bad answers, then review.

### Eval Dataset Schema

```json
{
  "eval_set": [
    {
      "query_id": "q001",
      "query": "What is the difference between sparse and dense retrieval?",
      "difficulty": "factual",
      "ground_truth": {
        "chunk_uids": ["...", "...", "..."],
        "papers": ["2312.10997", "2305.14283"],
        "pages": [[3, 4], [7]],
        "expected_topics": ["BM25", "dense vectors", "BERT encoders"]
      },
      "reference_answer": "Sparse retrieval uses term frequency (BM25), dense uses learned embeddings..."
    }
  ],
  "metadata": {
    "created": "2026-02-01",
    "corpus_version": "v1_200papers",
    "n_queries": 50
  }
}
```

### Evaluation Metrics

```python
def evaluate(eval_set: list[dict]) -> dict:
    """Run eval and compute metrics."""

    results = {
        "retrieval": {"recall@5": [], "recall@10": [], "mrr": []},
        "generation": {"has_citation": [], "citation_accuracy": [], "hallucination": []}
    }

    for item in eval_set:
        # Retrieval metrics
        retrieved = hybrid_search(item['query'], top_k=10)
        retrieved_ids = [c['chunk_uid'] for c in retrieved]
        ground_truth_ids = set(item['ground_truth']['chunk_uids'])

        # Recall@K: what fraction of ground truth was retrieved?
        recall_5 = len(set(retrieved_ids[:5]) & ground_truth_ids) / len(ground_truth_ids)
        recall_10 = len(set(retrieved_ids[:10]) & ground_truth_ids) / len(ground_truth_ids)

        # MRR: reciprocal rank of first correct result
        mrr = 0
        for rank, c in enumerate(retrieved, 1):
            if c['chunk_uid'] in ground_truth_ids:
                mrr = 1.0 / rank
                break

        results['retrieval']['recall@5'].append(recall_5)
        results['retrieval']['recall@10'].append(recall_10)
        results['retrieval']['mrr'].append(mrr)

        # Generation metrics
        answer = generate_answer(item['query'], retrieved[:5])

        # Check citation presence
        has_citation = bool(re.search(r'\[arXiv:\d+\.\d+ p\.\d+\]', answer))
        results['generation']['has_citation'].append(has_citation)

        # Citation accuracy: do cited papers exist and are they in ground truth?
        cited_papers = re.findall(r'arXiv:(\d+\.\d+)', answer)
        gt_papers = set(item['ground_truth']['papers'])
        citation_acc = len(set(cited_papers) & gt_papers) / max(len(cited_papers), 1)
        results['generation']['citation_accuracy'].append(citation_acc)

    # Aggregate
    return {
        "retrieval": {k: sum(v)/len(v) for k, v in results['retrieval'].items()},
        "generation": {k: sum(v)/len(v) for k, v in results['generation'].items()},
        "n_queries": len(eval_set)
    }
```

### Target Metrics (MVP)

| Metric            | Target | Notes                                   |
| ----------------- | ------ | --------------------------------------- |
| Recall@5          | >0.60  | At least 60% of ground truth in top 5   |
| Recall@10         | >0.80  | 80% in top 10                           |
| MRR               | >0.50  | Correct result usually in top 2         |
| Citation rate     | 100%   | Every answer has citations              |
| Citation accuracy | >0.80  | 80% of citations point to correct paper |

---

## 10. Project Phases

### Phase 1: Data Pipeline (Days 1)

**Goal**: 50 papers indexed and queryable via BM25

- [x] Set up project structure
- [x] Implement arXiv download script (IDs + keyword search)
- [x] PDF extraction with PyMuPDF
- [x] Chunking with page tracking
- [x] SQLite schema + FTS5 index
- [x] Basic CLI: `query` command (BM25 only)

**Deliverable**: Can run `arxiv-rag query "dense retrieval"` and get text chunks back

### Phase 2: Vector Search (Days 2)

**Goal**: Hybrid retrieval working

- [ ] OpenAI embeddings integration
- [ ] ChromaDB setup and indexing
- [ ] Hybrid search with RRF
- [ ] CLI: `--verbose` flag to show retrieval

**Deliverable**: `arxiv-rag sources "How does FAISS work?"` returns relevant chunks from both indexes

### Phase 3: Generation + Citations (Days 3)

**Goal**: End-to-end query → cited answer

- [ ] Answer generation with citation prompt
- [ ] Citation parsing and validation
- [ ] CLI: `--verify` mode
- [ ] Quote extraction

**Deliverable**: Full answers with `[arXiv:ID p.N]` citations and supporting quotes

### Phase 4: Evaluation (Days 4)

**Goal**: Know how good/bad your system is

- [ ] Generate synthetic QA pairs (50 questions)
- [ ] Manual review and correction of QA pairs
- [ ] Implement eval metrics
- [ ] Run eval, identify failure modes
- [ ] Query logging to SQLite

**Deliverable**: Eval report showing Recall@5, MRR, citation accuracy

### Phase 5: Polish + Scale (Days 5)

**Goal**: 200 papers, robust CLI

- [ ] Scale to 200 papers
- [ ] Error handling (bad PDFs, API failures)
- [ ] Progress bars and better UX
- [ ] `--add-ids` and `--rebuild` commands
- [ ] README and usage docs

**Deliverable**: Shareable tool you can demo

---

## 11. File Structure

```
arxiv-rag/
├── arxiv_rag/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI
│   ├── config.py           # Settings, API keys
│   ├── download.py         # arXiv fetching
│   ├── parse.py            # PDF extraction
│   ├── chunk.py            # Chunking logic
│   ├── index.py            # SQLite + ChromaDB
│   ├── retrieve.py         # Hybrid search
│   ├── generate.py         # Answer generation
│   └── evaluate.py         # Eval metrics
├── data/
│   ├── papers/             # Downloaded PDFs
│   ├── arxiv_rag.db        # SQLite database
│   └── chroma/             # ChromaDB persistence
├── eval/
│   ├── eval_set.json       # Ground truth QA pairs
│   └── results/            # Eval run outputs
├── tests/
│   └── ...
├── pyproject.toml
└── README.md
```

---

## 12. Dependencies

```toml
# pyproject.toml
[project]
name = "arxiv-rag"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "arxiv>=2.0",           # arXiv API
    "pymupdf>=1.23",        # PDF extraction
    "openai>=1.0",          # Embeddings + generation
    "chromadb>=0.4",        # Vector store
    "tiktoken>=0.5",        # Token counting
    "typer>=0.9",           # CLI
    "rich>=13.0",           # Pretty output
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
```

---

## 13. Cost Estimate

| Component                | Calculation                                                 | Cost    |
| ------------------------ | ----------------------------------------------------------- | ------- |
| Embeddings (index)       | 200 papers × 50 pages × 2 chunks × 512 tokens = ~10M tokens | $0.20   |
| Embeddings (queries)     | 100 queries × 512 tokens = 51K tokens                       | $0.01   |
| Generation (GPT-4o-mini) | 100 queries × 3K tokens out = 300K tokens                   | $0.45   |
| Generation (GPT-4o)      | Optional, for eval                                          | ~$3.00  |
| **Total MVP**            |                                                             | **<$5** |

---

## 14. Open Questions / Risks

| Risk                              | Mitigation                                                  |
| --------------------------------- | ----------------------------------------------------------- |
| PDF extraction quality varies     | Skip papers with <50% text extraction. Log failures.        |
| arXiv rate limits                 | Cache aggressively. Download once, store locally.           |
| Chunking splits important context | Overlap helps. Can increase to 150 tokens if needed.        |
| Citation hallucination            | Strict prompt + post-processing to validate citations exist |
| Eval set too synthetic            | Mix in 10-20 manually written questions                     |

---

## Appendix: Useful arXiv Categories

For RAG/LLM papers, focus on:

- `cs.CL` - Computation and Language (NLP)
- `cs.IR` - Information Retrieval
- `cs.LG` - Machine Learning
- `cs.AI` - Artificial Intelligence

Sample keyword searches:

- `"retrieval augmented generation"`
- `"dense retrieval" AND "language model"`
- `"RAG" AND "hallucination"`
- `"embedding" AND "similarity search"`
