import json
from pathlib import Path


EVALS_ROOT = Path(__file__).resolve().parents[1]


def test_v1_dataset_schema_and_counts():
    cases = [json.loads(line) for line in (EVALS_ROOT / "datasets/v1.jsonl").read_text().splitlines() if line]
    assert len(cases) >= 60
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(not case["answerable"] for case in cases) >= 10
    for case in cases:
        assert case["question"]
        assert case["case_type"]
        assert isinstance(case["gold_document_ids"], list)
        assert bool(case["gold_document_ids"]) == bool(case["answerable"])
