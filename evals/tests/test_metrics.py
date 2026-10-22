from evals.metrics.generation import citation_precision, judge_agreement, refusal_correct
from evals.metrics.retrieval import hit_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics_support_chunks_with_multiple_document_ids():
    retrieved = [["doc-a", "doc-b"], ["doc-c"], []]
    assert hit_at_k(retrieved, ["doc-c"], 1) == 0.0
    assert hit_at_k(retrieved, ["doc-c"], 3) == 1.0
    assert reciprocal_rank(retrieved, ["doc-c"]) == 0.5
    assert recall_at_k(retrieved, ["doc-b", "doc-c"], 1) == 0.5
    assert recall_at_k(retrieved, ["doc-b", "doc-c"], 3) == 1.0


def test_generation_metrics():
    assert refusal_correct("I don't have enough information in the provided sources.", True) == 1.0
    assert citation_precision("See [bloomington_00001] and [bloomington_99999].", ["bloomington_00001"]) == 0.5
    assert judge_agreement([1.0, 0.5], [0.9, 0.7]) == 1.0
