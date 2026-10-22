# RAG evaluation harness

This harness evaluates both Bloomington pipelines without using paid services:

- `linear`: one retrieval pass, matching the FastAPI application.
- `langgraph`: multi-query retrieval for cases with curated `sub_questions`; otherwise one retrieval pass. Full routing and generation use local Ollama.

Retrieval metrics run offline and are the only metrics used as a pull-request gate. Generation and judge runs are opt-in because local Ollama models are too large for ordinary GitHub-hosted runners.

## Dataset

`datasets/v1.jsonl` contains 60 versioned cases: 50 answerable cases anchored to stable `bloomington_*` corpus IDs and 10 unanswerable cases. Every record declares its provenance.

Important limitation: the original CSV has no source URLs. The initial answerable cases are marked `corpus_exact_seed`; they measure retrieval regression against the current corpus, not independent real-world factual correctness. Cases with known source-data defects are explicitly flagged. Do not present this dataset as externally validated until its questions and answers have been checked against authoritative Bloomington sources.

Never edit a released dataset in place. Create `v2.jsonl`, then re-run every baseline configuration.

## Commands

Create a Python 3.11 environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e '.[eval]'
```

Run retrieval evaluation:

```bash
python -m evals.runners.run_eval --config evals/configs/baseline.yaml
python -m evals.runners.run_eval --config evals/configs/baseline_langgraph.yaml
python -m evals.runners.run_eval --config evals/configs/dense_only.yaml
python -m evals.runners.run_eval --config evals/configs/hybrid.yaml
```

Run free local generation and judging:

```bash
ollama pull llama3.1:8b
ollama pull qwen3:4b
python -m evals.runners.run_eval \
  --config evals/configs/baseline.yaml \
  --with-generation
```

The harness never falls back to a hosted API. If Ollama is unavailable, it exits with an error.

## First experiment results

The measured October 21, 2026 runs are committed in `results/`. On the initial corpus-derived seed set, hybrid retrieval without SBERT reranking outperformed the current production configuration while being substantially faster. Keep the production configuration as the CI baseline until the seed cases have been manually reviewed; then decide whether to promote the no-reranker configuration and regenerate both baselines.

Compare saved runs:

```bash
python -m evals.report evals/results/*.json
```

Check a candidate against a committed baseline:

```bash
python -m evals.check_regression \
  --baseline evals/baselines/baseline_linear.json \
  --candidate evals/results/ci_baseline_linear.json
```

## Metric definitions

- `hit@k`: at least one gold record appears in the first `k` retrieved chunks.
- `MRR`: reciprocal rank of the first chunk containing a gold record.
- `recall@k`: fraction of all gold record IDs found in the first `k` chunks.
- `faithfulness`: local LLM judge score for support from retrieved context.
- `relevance`: local LLM judge score for answering the question.
- `refusal accuracy`: whether answerable cases are answered and unanswerable cases are refused.
- `citation precision`: fraction of cited corpus IDs that actually occur in retrieved context.

## Human judge calibration

Before publishing LLM-judge numbers, hand-label at least 20 generated answers. Store the labels in a new versioned JSONL file and report agreement using `evals.metrics.generation.judge_agreement`. A judge score without this calibration is exploratory, not a portfolio proof point.
