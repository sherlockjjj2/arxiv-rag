from arxiv_rag.evaluate import (
    compute_citation_accuracy,
    compute_mrr,
    compute_recall_at_k,
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

    assert compute_citation_accuracy(
        citations,
        ground_truth_chunk_uids=ground_truth,
        chunk_uids_by_page=mapping,
    ) == 0.5
