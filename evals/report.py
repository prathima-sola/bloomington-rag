#!/usr/bin/env python3
"""Build a Markdown comparison table from versioned result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row(path: Path) -> list[str]:
    result = load(path)
    summary = result["summary"]["overall"]
    latencies = [case["retrieval_ms"] for case in result["cases"]]
    generation = [case.get("generation") for case in result["cases"] if case.get("generation")]
    faithfulness = [item["judge"]["faithfulness"] for item in generation if item.get("judge")]
    refusal = [item["refusal_accuracy"] for item in generation]
    return [
        result["config_name"],
        result["pipeline"],
        str(summary["count"]),
        f"{summary['hit_at_3']:.3f}",
        f"{summary['mrr']:.3f}",
        f"{summary['recall_at_5']:.3f}",
        f"{sum(faithfulness) / len(faithfulness):.3f}" if faithfulness else "—",
        f"{sum(refusal) / len(refusal):.3f}" if refusal else "—",
        f"{median(latencies):.0f}" if latencies else "—",
        path.name,
    ]


def render(paths: list[Path]) -> str:
    headers = [
        "Config", "Pipeline", "N", "Hit@3", "MRR", "Recall@5",
        "Faithfulness", "Refusal acc.", "p50 retrieval ms", "Result",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row(path)) + " |" for path in paths)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = render(args.results)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
