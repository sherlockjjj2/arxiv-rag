from pathlib import Path
import time

import arxiv_rag.evaluate as evaluate_module
from arxiv_rag.evaluate import (
    CachedEmbeddingsClient,
    EvalCache,
    EvalFailureSummary,
    EvalItemResult,
    EvalMetadata,
    EvalSet,
    EvalGroundTruth,
    EvalItem,
    EvalReport,
    EvalSummary,
    _summarize_failures,
    compute_citation_accuracy,
    compute_mrr,
    compute_recall_at_k,
    render_report_markdown,
    run_eval,
)
from arxiv_rag.retrieve import ChunkResult
from arxiv_rag.verify import CitationRecord


def test_compute_recall_at_k() -> None:
    retrieved = ["c1", "c2", "c3"]
    ground_truth = ["c2", "c5"]

    assert compute_recall_at_k(retrieved, ground_truth, k=2) == 0.5
    assert compute_recall_at_k(retrieved, ground_truth, k=1) == 0.0


def test_compute_mrr() -> None:
    retrieved = ["c1", "c2", "c3"]
    ground_truth = ["c3", "c9"]

    assert compute_mrr(retrieved, ground_truth) == 1.0 / 3.0


def test_compute_citation_accuracy() -> None:
    citations = [
        CitationRecord(citation_id="c1", paper_id="1234.5678", page_number=1),
        CitationRecord(citation_id="c2", paper_id="9999.0000", page_number=5),
    ]
    mapping = {
        ("1234.5678", 1): ["u1"],
        ("9999.0000", 5): ["u3"],
    }
    ground_truth = ["u1", "u2"]

    assert (
        compute_citation_accuracy(
            citations,
            ground_truth_chunk_uids=ground_truth,
            chunk_uids_by_page=mapping,
        )
        == 0.5
    )


def test_summarize_failures_splits_absent_and_zero_score() -> None:
    results = [
        EvalItemResult(
            query_id="q1",
            query="q1",
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            first_correct_rank=None,
            retrieved_chunk_uids=["u1"],
            ground_truth_chunk_uids=["g1"],
            citation_count=0,
            citation_accuracy=0.0,
        ),
        EvalItemResult(
            query_id="q2",
            query="q2",
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr=1.0,
            first_correct_rank=1,
            retrieved_chunk_uids=["u2"],
            ground_truth_chunk_uids=["g2"],
            citation_count=2,
            citation_accuracy=0.0,
        ),
        EvalItemResult(
            query_id="q3",
            query="q3",
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr=1.0,
            first_correct_rank=1,
            retrieved_chunk_uids=["u3"],
            ground_truth_chunk_uids=["g3"],
            citation_count=1,
            citation_accuracy=1.0,
        ),
    ]

    failures = _summarize_failures(results)

    assert failures.citation_absent == 1
    assert failures.citation_zero_score == 1
    assert failures.citation_inaccurate == 2


def test_render_report_markdown_includes_new_citation_fields() -> None:
    report = EvalReport(
        created="2026-02-04T20:19:53",
        summary=EvalSummary(
            recall_at_5=0.7,
            recall_at_10=0.82,
            mrr=0.58,
            citation_accuracy=0.63,
            citation_accuracy_when_recall5_hit=0.89,
            n_queries=50,
        ),
        failures=EvalFailureSummary(
            retrieval_empty=0,
            recall_at_5_zero=15,
            mrr_zero=9,
            citation_absent=4,
            citation_zero_score=12,
            citation_parse_error=0,
            citation_inaccurate=19,
        ),
        items=[],
    )

    rendered = render_report_markdown(report)

    assert "Citation accuracy (Recall@5 > 0): 0.890" in rendered
    assert "Citation absent: 4" in rendered
    assert "Citation zero score: 12" in rendered


def test_eval_cache_round_trip(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "eval_cache.db")
    cache.set_query_embedding(
        query="what is rrf",
        embedding_model="text-embedding-3-small",
        embedding=[0.1, 0.2, 0.3],
    )
    cache.set_generated_answer(
        query="what is rrf",
        generation_model="gpt-4o-mini",
        prompt_version="prompt-v1",
        chunk_uids_hash="hash-1",
        answer="RRF answer",
    )

    assert cache.get_query_embedding(
        query="what is rrf",
        embedding_model="text-embedding-3-small",
    ) == [0.1, 0.2, 0.3]
    assert (
        cache.get_generated_answer(
            query="what is rrf",
            generation_model="gpt-4o-mini",
            prompt_version="prompt-v1",
            chunk_uids_hash="hash-1",
        )
        == "RRF answer"
    )


def test_cached_embeddings_client_reuses_cache(tmp_path: Path) -> None:
    class _FakeBaseClient:
        def __init__(self) -> None:
            from arxiv_rag.embeddings_client import EmbeddingsConfig

            self.config = EmbeddingsConfig(model="text-embedding-3-small")
            self.calls = 0

        def embed(self, inputs):
            from arxiv_rag.embeddings_client import EmbeddingBatchResult

            self.calls += 1
            return EmbeddingBatchResult(
                embeddings=[[float(len(text)), 1.0] for text in inputs],
                total_tokens=5,
            )

    base_client = _FakeBaseClient()
    cache = EvalCache(tmp_path / "emb_cache.db")
    client = CachedEmbeddingsClient(
        base_client=base_client,  # type: ignore[arg-type]
        cache=cache,
    )

    first = client.embed(["q1", "q2", "q1"])
    second = client.embed(["q1"])

    assert base_client.calls == 1
    assert first.embeddings[0] == first.embeddings[2]
    assert second.embeddings[0] == first.embeddings[0]


def test_run_eval_uses_generated_answer_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "eval.db"
    db_path.touch()
    cache_db_path = tmp_path / "cache.db"
    generation_calls = {"count": 0}

    eval_set = EvalSet(
        eval_set=[
            EvalItem(
                query_id="q1",
                query="Explain dense retrieval",
                difficulty="factual",
                ground_truth=EvalGroundTruth(
                    chunk_uids=["uid-1"],
                    papers=["1234.5678"],
                    pages=[[1]],
                ),
                reference_answer="Ref",
            )
        ],
        metadata=EvalMetadata(
            created="2026-02-04",
            corpus_version="v1",
            n_queries=1,
        ),
    )

    def _fake_run_retrieval(*, query, db_path, retrieval_config, **kwargs):
        del query, db_path, retrieval_config, kwargs
        return (
            [
                ChunkResult(
                    chunk_uid="uid-1",
                    chunk_id=1,
                    paper_id="1234.5678",
                    page_number=1,
                    text="chunk text",
                    score=1.0,
                )
            ],
            [],
        )

    def _fake_generate_answer(query, chunks, model="gpt-4o-mini"):
        del query, chunks, model
        generation_calls["count"] += 1
        return "cached answer"

    monkeypatch.setattr(evaluate_module, "_run_retrieval", _fake_run_retrieval)
    monkeypatch.setattr(evaluate_module, "generate_answer", _fake_generate_answer)
    monkeypatch.setattr(evaluate_module, "parse_citations", lambda answer: [])
    monkeypatch.setattr(evaluate_module, "load_prompt_template", lambda path=None: "P")

    run_eval(
        eval_set=eval_set,
        db_path=db_path,
        retrieval_config=evaluate_module.RetrievalConfig(mode="fts", top_k=5),
        generate=True,
        generate_model="gpt-4o-mini",
        generation_top_k=1,
        cache_db_path=cache_db_path,
    )
    run_eval(
        eval_set=eval_set,
        db_path=db_path,
        retrieval_config=evaluate_module.RetrievalConfig(mode="fts", top_k=5),
        generate=True,
        generate_model="gpt-4o-mini",
        generation_top_k=1,
        cache_db_path=cache_db_path,
    )

    assert generation_calls["count"] == 1


def test_run_eval_avoids_duplicate_generation_with_concurrency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "eval.db"
    db_path.touch()
    cache_db_path = tmp_path / "cache.db"
    generation_calls = {"count": 0}

    eval_set = EvalSet(
        eval_set=[
            EvalItem(
                query_id="q1",
                query="Explain dense retrieval",
                difficulty="factual",
                ground_truth=EvalGroundTruth(
                    chunk_uids=["uid-1"],
                    papers=["1234.5678"],
                    pages=[[1]],
                ),
                reference_answer="Ref 1",
            ),
            EvalItem(
                query_id="q2",
                query="Explain dense retrieval",
                difficulty="factual",
                ground_truth=EvalGroundTruth(
                    chunk_uids=["uid-1"],
                    papers=["1234.5678"],
                    pages=[[1]],
                ),
                reference_answer="Ref 2",
            ),
        ],
        metadata=EvalMetadata(
            created="2026-02-04",
            corpus_version="v1",
            n_queries=2,
        ),
    )

    def _fake_run_retrieval(*, query, db_path, retrieval_config, **kwargs):
        del query, db_path, retrieval_config, kwargs
        return (
            [
                ChunkResult(
                    chunk_uid="uid-1",
                    chunk_id=1,
                    paper_id="1234.5678",
                    page_number=1,
                    text="chunk text",
                    score=1.0,
                )
            ],
            [],
        )

    def _fake_generate_answer(query, chunks, model="gpt-4o-mini"):
        del query, chunks, model
        generation_calls["count"] += 1
        time.sleep(0.05)
        return "cached answer"

    monkeypatch.setattr(evaluate_module, "_run_retrieval", _fake_run_retrieval)
    monkeypatch.setattr(evaluate_module, "generate_answer", _fake_generate_answer)
    monkeypatch.setattr(evaluate_module, "parse_citations", lambda answer: [])
    monkeypatch.setattr(evaluate_module, "load_prompt_template", lambda path=None: "P")

    run_eval(
        eval_set=eval_set,
        db_path=db_path,
        retrieval_config=evaluate_module.RetrievalConfig(mode="fts", top_k=5),
        generate=True,
        generate_model="gpt-4o-mini",
        generation_top_k=1,
        generation_concurrency=2,
        cache_db_path=cache_db_path,
    )

    assert generation_calls["count"] == 1
