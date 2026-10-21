"""Minimal zero-cost Ollama client for generation and LLM-as-judge runs."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


class OllamaUnavailable(RuntimeError):
    pass


def generate(prompt: str, model: str, base_url: str = "http://localhost:11434", timeout: int = 180) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaUnavailable(
            "Ollama is unavailable. Start it locally and pull the configured models; "
            "the harness will never fall back to a paid API."
        ) from exc
    return str(value["response"]).strip()


def judge(question: str, context: str, answer: str, model: str, base_url: str) -> dict:
    prompt = f"""You are a strict RAG evaluator. Use only the supplied context.
Return JSON only with numeric scores from 0 to 1:
{{"faithfulness": 0.0, "relevance": 0.0, "reason": "short explanation"}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""
    raw = generate(prompt, model=model, base_url=base_url)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Judge did not return JSON: {raw[:200]}")
    value = json.loads(raw[start : end + 1])
    return {
        "faithfulness": float(value["faithfulness"]),
        "relevance": float(value["relevance"]),
        "reason": str(value.get("reason", "")),
    }
