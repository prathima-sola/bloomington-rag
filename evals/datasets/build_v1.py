#!/usr/bin/env python3
"""Build the reproducible v1 seed dataset from stable corpus record IDs.

The output deliberately records that answerable questions were selected from
the existing corpus. They are retrieval regression cases, not independent
real-world factual validation.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/full_dataset.json"
OUTPUT = Path(__file__).with_name("v1.jsonl")

TARGETS = {
    "Hotels and Accommodation": 13,
    "General Bloomington Info": 9,
    "Restaurants and Dining": 7,
    "Events and University": 6,
    "Attractions and Entertainment": 5,
    "Shopping": 2,
    "Transportation": 2,
    "Weather and Seasons": 1,
}

MULTI_HOP = [
    {
        "question": "What does a room at Graduate Bloomington cost, where is the hotel, and is it easy to get around from there?",
        "gold_document_ids": ["bloomington_00063", "bloomington_00463"],
        "sub_questions": [
            "What's the approximate price range for a room at the Graduate Bloomington hotel?",
            "Where is The Graduate Bloomington located, and how easy is it to get around from there?",
        ],
    },
    {
        "question": "Suggest a budget hotel near IU and explain how close the Super 8 is to both campus and downtown.",
        "gold_document_ids": ["bloomington_00119", "bloomington_00168"],
        "sub_questions": [
            "I'm looking for a budget-friendly hotel in Bloomington near Indiana University. What are some options and their approximate price ranges?",
            "How close is the Super 8 by Wyndham Bloomington University Area to Indiana University and downtown Bloomington?",
        ],
    },
    {
        "question": "Compare cabin or campground options outside downtown with family resorts near Bloomington, including likely prices and amenities.",
        "gold_document_ids": ["bloomington_00006", "bloomington_00448"],
        "sub_questions": [
            "My family is looking for a cabin or campground rental outside of downtown Bloomington. Where can we find options and what should we expect to pay?",
            "What's the best resort near Bloomington for families with young children, and what amenities do they offer?",
        ],
    },
    {
        "question": "What do Bloomington bed and breakfasts usually cost, and can any accommodate a group of six adults?",
        "gold_document_ids": ["bloomington_00500", "bloomington_00513"],
        "sub_questions": [
            "What's the typical price range for bed and breakfasts in Bloomington, Indiana, and what factors influence the cost?",
            "I'm traveling with a large group (6 adults). Can any of these historic bed and breakfasts accommodate us?",
        ],
    },
    {
        "question": "What should I consider when choosing off-campus IU housing, and how can I find a long-term rental for several months?",
        "gold_document_ids": ["bloomington_00921", "bloomington_00925"],
        "sub_questions": [
            "What factors should I consider when choosing off-campus housing near Indiana University?",
            "How can I find long-term rental options near IU Bloomington if I'm visiting for several months?",
        ],
    },
]

REJECT = re.compile(
    r"(?i)\[(?:insert|hotel name|website|address)|source text|provided text|"
    r"december 2024|\b2025\b|today|tonight"
)

UNANSWERABLE = [
    ("unanswerable_001", "What concerts are happening in Bloomington tonight?", "time_sensitive"),
    ("unanswerable_002", "What is the current wait time at FARMbloomington?", "time_sensitive"),
    ("unanswerable_003", "Is the next Bloomington Transit bus running late right now?", "time_sensitive"),
    ("unanswerable_004", "What will the exact weather be at 3 PM tomorrow?", "time_sensitive"),
    ("unanswerable_005", "Can you book a room at the Graduate Bloomington for me?", "out_of_scope"),
    ("unanswerable_006", "What is my Indiana University student account balance?", "out_of_scope"),
    ("unanswerable_007", "Who is currently sitting at the mayor's desk?", "out_of_scope"),
    ("unanswerable_008", "What is the Wi-Fi password at my hotel?", "out_of_scope"),
    ("unanswerable_009", "Can you refund my Bloomington parking ticket?", "out_of_scope"),
    ("unanswerable_010", "Which restaurant has an open table for six at 7:15 PM today?", "time_sensitive"),
]


def case_type(question: str) -> str:
    lowered = question.casefold()
    if any(term in lowered for term in ("compare", " versus ", "both ")):
        return "multi_hop"
    if any(term in lowered for term in ("what are some", "which hotels", "what options")):
        return "list"
    if any(term in lowered for term in ("price", "cost", "available", "hours")):
        return "time_sensitive"
    return "factual"


def useful(record: dict) -> bool:
    question, answer = record["question"].strip(), record["answer"].strip()
    return (
        30 <= len(question) <= 180
        and 80 <= len(answer) <= 700
        and not REJECT.search(question)
        and not REJECT.search(answer)
    )


def evenly_spaced(records: list[dict], count: int) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"Need {count} records but only found {len(records)}")
    if count == 1:
        return [records[len(records) // 2]]
    return [records[round(index * (len(records) - 1) / (count - 1))] for index in range(count)]


def main() -> None:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    all_by_category: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        all_by_category[record["category"]].append(record)
        if useful(record):
            grouped[record["category"]].append(record)

    cases = []
    for category, target in TARGETS.items():
        pool = list(grouped[category])
        if len(pool) < target:
            selected_ids = {record["id"] for record in pool}
            pool.extend(record for record in all_by_category[category] if record["id"] not in selected_ids)
        for record in evenly_spaced(pool, target):
            cases.append(
                {
                    "id": f"v1_{len(cases) + 1:03d}",
                    "question": record["question"].strip(),
                    "case_type": case_type(record["question"]),
                    "answerable": True,
                    "gold_document_ids": [record["id"]],
                    "reference_answer": record["answer"].strip(),
                    "keywords": [],
                    "category": category,
                    "provenance": "corpus_exact_seed",
                    "known_data_quality_issue": not useful(record),
                }
            )

    records_by_id = {record["id"]: record for record in records}
    for value in MULTI_HOP:
        cases.append(
            {
                "id": f"v1_{len(cases) + 1:03d}",
                "question": value["question"],
                "case_type": "multi_hop",
                "answerable": True,
                "gold_document_ids": value["gold_document_ids"],
                "reference_answer": "\n\n".join(
                    records_by_id[doc_id]["answer"] for doc_id in value["gold_document_ids"]
                ),
                "keywords": [],
                "category": "Cross-category",
                "provenance": "human_composed_from_corpus_records",
                "sub_questions": value["sub_questions"],
                "known_data_quality_issue": False,
            }
        )

    for case_id, question, kind in UNANSWERABLE:
        cases.append(
            {
                "id": case_id,
                "question": question,
                "case_type": kind,
                "answerable": False,
                "gold_document_ids": [],
                "reference_answer": "The assistant should refuse because the static corpus cannot answer this.",
                "keywords": [],
                "category": "Unanswerable",
                "provenance": "human_authored_seed",
            }
        )

    OUTPUT.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
