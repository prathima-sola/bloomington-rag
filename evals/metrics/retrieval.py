"""Deterministic retrieval metrics."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable


def _relevant_at_k(retrieved: list[list[str]], gold: set[str], k: int) -> list[set[str]]:
    return [set(ids) & gold for ids in retrieved[:k]]


def hit_at_k(retrieved: list[list[str]], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return float(any(_relevant_at_k(retrieved, gold_set, k)))


def reciprocal_rank(retrieved: list[list[str]], gold: Iterable[str]) -> float:
    gold_set = set(gold)
    for rank, ids in enumerate(retrieved, 1):
        if set(ids) & gold_set:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[list[str]], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    found = set().union(*_relevant_at_k(retrieved, gold_set, k))
    return len(found) / len(gold_set)


def score_case(retrieved: list[list[str]], gold: Iterable[str]) -> dict[str, float]:
    return {
        "hit_at_1": hit_at_k(retrieved, gold, 1),
        "hit_at_3": hit_at_k(retrieved, gold, 3),
        "hit_at_5": hit_at_k(retrieved, gold, 5),
        "mrr": reciprocal_rank(retrieved, gold),
        "recall_at_1": recall_at_k(retrieved, gold, 1),
        "recall_at_3": recall_at_k(retrieved, gold, 3),
        "recall_at_5": recall_at_k(retrieved, gold, 5),
    }


def aggregate(case_results: list[dict]) -> dict:
    answerable = [item for item in case_results if item.get("gold_document_ids")]
    metric_names = list(answerable[0]["retrieval_metrics"]) if answerable else []

    def summarize(items: list[dict]) -> dict[str, float | int]:
        summary: dict[str, float | int] = {"count": len(items)}
        for name in metric_names:
            summary[name] = mean(item["retrieval_metrics"][name] for item in items) if items else 0.0
        return summary

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in answerable:
        grouped[item.get("case_type", "unknown")].append(item)

    return {
        "overall": summarize(answerable),
        "by_case_type": {name: summarize(items) for name, items in sorted(grouped.items())},
    }
