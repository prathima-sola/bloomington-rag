import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Pipeline state
_pipeline = {}
_ready = False
_loading = False
_lock = threading.Lock()


def load_pipeline():
    global _pipeline, _ready, _loading
    _loading = True

    try:
        print("Starting pipeline load...")

        from langchain_community.document_loaders import (
            TextLoader, DirectoryLoader
        )
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter
        )
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder

        print("Imports done.")

        base = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base, "..", "data", "processed")

        loader = DirectoryLoader(
            data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["---", "\n\n", "\n", ". "]
        )
        chunks = splitter.split_documents(documents)
        print(f"Loaded {len(chunks)} chunks")

        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="bloomington",
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="bloomington",
            embedding=embedding_model
        )
        vector_store.add_documents(chunks)
        print("Vector store ready.")

        tokenized = [
            doc.page_content.lower().split()
            for doc in chunks
        ]
        bm25 = BM25Okapi(tokenized)

        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512
        )

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template("""
You are a friendly Bloomington Indiana tourist assistant.
Answer using ONLY the information below. Be helpful and concise.
If you don't have enough information say so honestly.

Information:
{context}

Question: {question}

Answer:""")

        chain = (
            RunnableLambda(lambda x: {
                "context": x["context"],
                "question": x["question"]
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        _pipeline["vector_store"] = vector_store
        _pipeline["bm25"] = bm25
        _pipeline["chunks"] = chunks
        _pipeline["reranker"] = reranker
        _pipeline["chain"] = chain
        _ready = True
        print("Pipeline ready!")

    except Exception as e:
        print(f"Pipeline error: {e}")

    finally:
        _loading = False


def reciprocal_rank_fusion(results_list, k=60):
    scores = {}
    for results in results_list:
        for rank, doc in enumerate(results, 1):
            doc_id = doc.page_content[:100]
            if doc_id not in scores:
                scores[doc_id] = {"score": 0, "doc": doc}
            scores[doc_id]["score"] += 1 / (k + rank)
    return [
        item["doc"]
        for item in sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
    ]


def retrieve(query):
    vector_store = _pipeline["vector_store"]
    bm25 = _pipeline["bm25"]
    chunks = _pipeline["chunks"]
    reranker = _pipeline["reranker"]

    vector_results = vector_store.similarity_search(query, k=8)
    bm25_scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:8]
    bm25_results = [chunks[i] for i in top_indices]
    combined = reciprocal_rank_fusion(
        [vector_results, bm25_results]
    )

    pairs = [(query, doc.page_content) for doc in combined]
    scores = reranker.predict(pairs)
    scored = sorted(
        zip(scores, combined),
        key=lambda x: x[0],
        reverse=True
    )

    confident = [(s, d) for s, d in scored if s > 0]
    if len(confident) < 2:
        confident = scored[:2]
    return confident[:4]


def format_context(docs):
    parts = []
    for i, (score, doc) in enumerate(docs, 1):
        text = doc.page_content
        if "ANSWER:" in text:
            parts.append(
                f"[Info {i}]\n"
                f"{text.split('ANSWER:')[-1].strip()}"
            )
        else:
            parts.append(f"[Info {i}]\n{text[:400]}")
    return "\n\n".join(parts)


# FastAPI app
app = FastAPI(title="Bloomington Tourist Assistant")


@app.on_event("startup")
def startup_event():
    load_pipeline()
    print("Server ready.")


@app.get("/health")
def health():
    return {"status": "ok", "ready": _ready, "loading": _loading}


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Question cannot be empty"}
        )

    if not _ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Still loading, please wait 60 seconds"}
        )

    try:
        docs = retrieve(question)
        context = format_context(docs)
        answer = _pipeline["chain"].invoke({
            "question": question,
            "context": context
        })
        return {
            "question": question,
            "answer": answer,
            "sources": len(docs),
            "top_score": float(docs[0][0]) if docs else 0
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


static_dir = os.path.join(
    os.path.dirname(__file__), "static"
)
os.makedirs(static_dir, exist_ok=True)
app.mount(
    "/static",
    StaticFiles(directory=static_dir),
    name="static"
)


@app.get("/", response_class=HTMLResponse)
def frontend():
    html_path = os.path.join(static_dir, "index.html")
    with open(html_path, "r") as f:
        return f.read()