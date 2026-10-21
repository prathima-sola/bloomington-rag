"""Backward-compatible entry point for the versioned retrieval evaluation.

The old file duplicated production retrieval, allowing CI and the API to drift.
This module now delegates to the shared harness.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.check_regression import main as check_regression
from evals.runners.run_eval import run


def run_retrieval_quality_test() -> bool:
    output = ROOT / "evals/results/legacy_entrypoint.json"
    run(
        ROOT / "evals/configs/baseline.yaml",
        ROOT / "evals/datasets/v1.jsonl",
        output,
        with_generation=False,
    )
    original = sys.argv
    try:
        sys.argv = [
            "check_regression",
            "--baseline",
            str(ROOT / "evals/baselines/baseline_linear.json"),
            "--candidate",
            str(output),
        ]
        return check_regression() == 0
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(0 if run_retrieval_quality_test() else 1)
