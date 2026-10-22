#!/usr/bin/env python3
"""Run a versioned dataset against one retrieval/generation configuration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.metrics.generation import citation_precision, keyword_coverage, refusal_correct
from evals.metrics.retrieval import aggregate, score_case
from evals.ollama import OllamaUnavailable, generate, judge
from src.retrieval import RetrievalConfig, RetrievalEngine, RetrievedChunk


def load_jsonl(path: Path) -> list[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        required = {"id", "question", "case_type", "answerable", "gold_document_ids"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
        cases.append(value)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError(f"{path}: case IDs must be unique")
    return cases


def load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("pipeline") not in {"linear", "langgraph"}:
        raise ValueError("pipeline must be 'linear' or 'langgraph'")
    return value


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def retrieve_case(engine: RetrievalEngine, case: dict, pipeline: str) -> list[RetrievedChunk]:
    if pipeline == "linear":
        return engine.retrieve(case["question"])

    queries = case.get("sub_questions") or [case["question"]]
    combined: list[RetrievedChunk] = []
    seen: set[str] = set()
    for query in queries:
        for item in engine.retrieve(query, top_k=3):
            if item.chunk_id not in seen:
                seen.add(item.chunk_id)
                combined.append(item)
    for rank, item in enumerate(combined, 1):
        item.rank = rank
    return combined[: engine.config.top_k]


def context_for(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for chunk in chunks:
        ids = ", ".join(f"[{doc_id}]" for doc_id in chunk.document_ids) or f"[{chunk.chunk_id}]"
        parts.append(f"SOURCE {ids}\n{chunk.text}")
    return "\n\n".join(parts)


def generate_answer(case: dict, chunks: list[RetrievedChunk], settings: dict) -> str:
    context = context_for(chunks)
    prompt = f"""You are a Bloomington, Indiana tourism assistant.
Answer only from the context. Cite supporting record IDs in square brackets.
If the context does not answer the question, say: "I don't have enough information in the provided sources."

CONTEXT:
{context}

QUESTION:
{case['question']}

ANSWER:
"""
    return generate(
        prompt,
        model=settings["model"],
        base_url=settings.get("base_url", "http://localhost:11434"),
    )


def run(config_path: Path, dataset_path: Path, output_path: Path | None, with_generation: bool) -> dict:
    config = load_config(config_path)
    cases = load_jsonl(dataset_path)
    retrieval_config = RetrievalConfig(**config["retrieval"])
    engine = RetrievalEngine(retrieval_config)
    generation_settings = config.get("generation", {})

    results = []
    for index, case in enumerate(cases, 1):
        started = perf_counter()
        chunks = retrieve_case(engine, case, config["pipeline"])
        retrieval_ms = (perf_counter() - started) * 1000
        retrieved_ids = [chunk.document_ids for chunk in chunks]
        item = {
            "id": case["id"],
            "question": case["question"],
            "case_type": case["case_type"],
            "answerable": case["answerable"],
            "gold_document_ids": case["gold_document_ids"],
            "retrieval_ms": round(retrieval_ms, 2),
            "retrieved": [chunk.to_dict() for chunk in chunks],
            "retrieval_metrics": score_case(retrieved_ids, case["gold_document_ids"]),
        }
        if with_generation:
            answer_started = perf_counter()
            answer = generate_answer(case, chunks, generation_settings)
            generation_ms = (perf_counter() - answer_started) * 1000
            allowed_ids = sorted({doc_id for chunk in chunks for doc_id in chunk.document_ids})
            item["generation"] = {
                "answer": answer,
                "generation_ms": round(generation_ms, 2),
                "keyword_coverage": keyword_coverage(answer, case.get("keywords", [])),
                "refusal_accuracy": refusal_correct(answer, not case["answerable"]),
                "citation_precision": citation_precision(answer, allowed_ids),
            }
            judge_settings = config.get("judge")
            if judge_settings:
                item["generation"]["judge"] = judge(
                    case["question"],
                    context_for(chunks),
                    answer,
                    model=judge_settings["model"],
                    base_url=judge_settings.get("base_url", "http://localhost:11434"),
                )
        results.append(item)
        print(f"[{index:02d}/{len(cases)}] {case['id']} {retrieval_ms:.0f}ms")

    summary = aggregate(results)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "working_tree_dirty": git_dirty(),
        "config_name": config["name"],
        "pipeline": config["pipeline"],
        "config": config,
        "dataset": str(dataset_path.relative_to(ROOT)),
        "summary": summary,
        "cases": results,
    }

    if output_path is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = ROOT / "evals" / "results" / f"{stamp}_{config['name']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Result: {output_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/datasets/v1.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-generation", action="store_true")
    args = parser.parse_args()
    try:
        run(args.config.resolve(), args.dataset.resolve(), args.output, args.with_generation)
    except OllamaUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
