import os
from contextlib import asynccontextmanager
from threading import Lock, Thread

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
_load_error = None
_load_lock = Lock()


def load_pipeline():
    global _pipeline, _ready, _loading, _load_error
    if not _load_lock.acquire(blocking=False):
        return
    _ready = False
    _loading = True
    _load_error = None

    try:
        print("Starting pipeline load...")

        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda
        from src.retrieval import RetrievalConfig, RetrievalEngine

        print("Imports done.")

        base = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base, "..", "data", "processed")

        engine = RetrievalEngine(
            RetrievalConfig(strategy="hybrid", rerank=True, top_k=4),
            data_dir=data_dir,
        )
        print(f"Loaded {len(engine.chunks)} chunks")

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

        _pipeline["retrieval_engine"] = engine
        _pipeline["chain"] = chain
        _ready = True
        print("Pipeline ready!")

    except Exception as e:
        print(f"Pipeline error: {e}")
        _load_error = str(e)

    finally:
        _loading = False
        _load_lock.release()


def retrieve(query):
    return _pipeline["retrieval_engine"].retrieve_legacy(query, top_k=4)


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


@asynccontextmanager
async def lifespan(_app):
    # Do not block the web server port while models download/load. This lets
    # Cloud Run start the container and serve the UI/health endpoint during a
    # cold start. /ask remains unavailable until /ready reports 200.
    Thread(target=load_pipeline, name="retrieval-loader", daemon=True).start()
    print("Server started; retrieval pipeline is loading.")
    yield


# FastAPI app
app = FastAPI(title="Bloomington Tourist Assistant", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ready" if _ready else "loading" if _loading else "error",
        "ready": _ready,
        "loading": _loading,
        "error": _load_error,
    }


@app.get("/ready")
def readiness():
    if _ready:
        return {"status": "ready"}
    return JSONResponse(
        status_code=503,
        content={
            "status": "loading" if _loading else "error",
            "error": _load_error,
        },
    )


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
