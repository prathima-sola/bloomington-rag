"""Free deterministic metrics plus agreement helpers for local LLM judging."""

from __future__ import annotations

import re
from statistics import mean


REFUSAL_PATTERNS = (
    "don't have enough information",
    "do not have enough information",
    "not available in the provided",
    "cannot answer from",
    "can't answer from",
)


def keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    lowered = answer.casefold()
    return sum(keyword.casefold() in lowered for keyword in keywords) / len(keywords)


def refused(answer: str) -> bool:
    lowered = answer.casefold()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def refusal_correct(answer: str, should_refuse: bool) -> float:
    return float(refused(answer) == should_refuse)


def cited_document_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[(bloomington_\d{5})\]", answer))


def citation_precision(answer: str, allowed_ids: list[str]) -> float | None:
    citations = cited_document_ids(answer)
    if not citations:
        return None
    return len(citations & set(allowed_ids)) / len(citations)


def judge_agreement(human_scores: list[float], judge_scores: list[float], tolerance: float = 0.25) -> float:
    if len(human_scores) != len(judge_scores) or not human_scores:
        raise ValueError("human and judge score lists must have the same non-zero length")
    return mean(abs(human - judge) <= tolerance for human, judge in zip(human_scores, judge_scores))
