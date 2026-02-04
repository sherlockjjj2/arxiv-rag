from arxiv_rag.evaluate import (
    EvalFailureSummary,
    EvalItemResult,
    EvalReport,
    EvalSummary,
    _summarize_failures,
    compute_citation_accuracy,
    compute_mrr,
    compute_recall_at_k,
    render_report_markdown,
)
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
