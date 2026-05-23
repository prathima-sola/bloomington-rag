# Bloomington Tourist Assistant — Production RAG System

A production-grade Retrieval Augmented Generation (RAG) system 
for Bloomington, Indiana tourism, built with 6 layers of 
retrieval and evaluation infrastructure.

**Live Demo:** https://bloomington-rag-473936713859.us-central1.run.app

---

## Architecture
User Question
↓
LangGraph Router (classify: simple vs complex)
↓                        ↓
Direct Retrieval          Query Decomposition
↓                        ↓
Layer 1: Vector Search (Qdrant + all-MiniLM-L6-v2)
Layer 2: BM25 Keyword Search
Layer 3: Reciprocal Rank Fusion (RRF)
Layer 4: Cross-Encoder Reranking (SBERT)
↓
Groq LLM (llama-3.1-8b-instant)
↓
Cited Answer
## Stack

- **Orchestration:** LangChain, LangGraph
- **Vector Store:** Qdrant (in-memory)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (local, free)
- **Hybrid Search:** BM25 + Vector + Reciprocal Rank Fusion
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2 (SBERT)
- **LLM:** Groq (llama-3.1-8b-instant)
- **Backend:** FastAPI + uvicorn
- **Deployment:** GCP Cloud Run
- **CI/CD:** GitHub Actions

---

## Evaluation Results

| Metric | Score | Method |
|--------|-------|--------|
| Retrieval Accuracy | 86.67% (13/15) | Automated CI — runs on every commit |
| Retrieval Threshold | 73% | Build fails if accuracy drops below this |
| Faithfulness (overall) | 0.73 | LLM-as-judge evaluation |
| Faithfulness (hotel queries) | 0.89 | LLM-as-judge evaluation |
| Faithfulness (restaurant queries) | 0.33 | LLM-as-judge evaluation |

---

## Known Limitations

**Data imbalance:** Dataset contains 829 hotel Q&A pairs vs 
53 restaurant pairs vs 26 attraction pairs. Faithfulness drops 
to 0.33 on restaurant queries due to weak retrieval context 
causing LLM hallucination. Root cause identified — expanding 
dataset is the documented next step.

**In-memory vector store:** Qdrant runs in memory. Each cold 
start rebuilds the index (~30 seconds). A persistent Qdrant 
instance would eliminate this.

**Single model tier:** Classification and synthesis use the 
same model (llama-3.1-8b-instant). Model tiering — small 
classifier, large synthesizer — is the planned improvement 
for cost efficiency at scale.

**Static dataset:** Data collected December 2024. Hotel prices, 
restaurant hours, and events change. Answers may be outdated.

---

## Project Structure
bloomington-rag/
├── src/
│   ├── api.py              # FastAPI backend
│   ├── rag_pipeline.py     # 3-layer retrieval pipeline
│   ├── agentic_rag.py      # LangGraph routing
│   ├── prepare_data.py     # ETL pipeline
│   └── static/
│       └── index.html      # Chat UI
├── data/
│   └── processed/          # 1,000 Q&A pairs, 8 topic files
├── tests/
│   ├── test_retrieval.py   # CI quality gate
│   ├── faithfulness_eval.py # Faithfulness evaluation
│   └── eval_dataset.py     # 15 evaluation questions
├── .github/
│   └── workflows/
│       └── quality_gate.yml # GitHub Actions CI
└── Procfile                # GCP Cloud Run entrypoint
---

## CI Pipeline

GitHub Actions runs on every push to main:
- Loads all 1,281 document chunks
- Runs 15 test questions through full retrieval pipeline
- Zero API calls — deterministic, free
- Fails build if accuracy drops below 73% threshold
- Runs in under 3 minutes

---

## What Each Layer Adds

| Layer | What it does | Why it exists |
|-------|-------------|---------------|
| Vector search | Semantic similarity | Finds conceptually related chunks |
| BM25 | Keyword matching | Catches exact term matches vector misses |
| RRF | Combines both lists | Neither search alone is sufficient |
| Cross-encoder | Reads query+chunk together | Eliminates wrong-document contamination |
| LangGraph routing | Simple vs complex path | Complex questions need decomposition |

---

## Running Locally

```bash
git clone https://github.com/prathima-sola/bloomington-rag
cd bloomington-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py
uvicorn src.api:app --reload --port 8000
```

Open http://localhost:8000

---

## Running Evaluation

```bash
# Retrieval quality test (no API needed, ~3 minutes)
python tests/test_retrieval.py

# Faithfulness evaluation (uses Groq API, ~4 minutes)
python tests/faithfulness_eval.py
```