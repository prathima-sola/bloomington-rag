import os
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader, DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
from langgraph.graph import StateGraph, END
import operator

load_dotenv()


# ============================================================
# STATE — what flows through the graph
# ============================================================

class RAGState(TypedDict):
    """
    The state object that passes between nodes in the graph.
    Every node reads from this and writes back to it.

    Think of it as a shared whiteboard that all nodes
    can read and write to as the question flows through
    the pipeline.
    """
    question: str                    # original user question
    question_type: str               # "simple" or "complex"
    sub_questions: List[str]         # for complex questions
    retrieved_docs: List            # retrieved chunks
    context: str                     # formatted context for LLM
    answer: str                      # final answer
    reasoning: str                   # why we routed this way


# ============================================================
# SETUP (same as before)
# ============================================================

def setup_retrievers(data_dir="data/processed"):
    print("Setting up retrievers...")

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

    bm25 = BM25OkapiRetriever(chunks)

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=512
    )

    print("All retrievers ready.")
    return vector_store, bm25, reranker, chunks


class BM25OkapiRetriever:
    def __init__(self, documents):
        self.documents = documents
        tokenized = [
            doc.page_content.lower().split()
            for doc in documents
        ]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, k=8):
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        return [self.documents[i] for i in top_indices]


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


def retrieve_for_query(query, vector_store, bm25, reranker, k=4):
    """Full 3-layer retrieval for a single query."""
    vector_results = vector_store.similarity_search(query, k=8)
    bm25_results = bm25.retrieve(query, k=8)
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

    return confident[:k]


# ============================================================
# LLM SETUP
# ============================================================

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )


# ============================================================
# GRAPH NODES
# ============================================================

def classify_question(state: RAGState) -> RAGState:
    """
    NODE 1: Classify question as simple or complex.

    Simple = one clear question with one answer
    Complex = multiple parts, comparisons, or follow-ups

    We use a SMALL model here to keep costs low.
    This node runs on EVERY request so efficiency matters.

    Examples:
    Simple:  "What hotels are near IU?"
    Complex: "Compare budget vs mid-range hotels near IU
              for a family of four visiting in summer"
    """
    question = state["question"]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a question classifier for a Bloomington, Indiana tourist assistant.

Classify this question as either SIMPLE or COMPLEX.

SIMPLE: Single question with one clear answer.
Examples:
- "What hotels are near Indiana University?"
- "Are there good restaurants downtown?"
- "What is the weather like in summer?"

COMPLEX: Multiple parts, comparisons, or requires
combining information from different topics.
Examples:
- "Compare budget and mid-range hotels and tell me
   which is better for a family"
- "What should I do on my first day and where should
   I eat after?"
- "What are the best hotels near campus and what
   restaurants are nearby?"

Question: {question}

Respond with ONLY one word: SIMPLE or COMPLEX""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().upper()

    # Default to SIMPLE if unclear
    question_type = "complex" if "COMPLEX" in result else "simple"

    reasoning = (
        f"Classified as {question_type.upper()} because: "
        f"{'multiple parts or comparison detected' if question_type == 'complex' else 'single focused question'}"
    )

    print(f"\n[NODE 1 - CLASSIFY] '{question[:50]}...' "
          f"→ {question_type.upper()}")

    return {
        **state,
        "question_type": question_type,
        "reasoning": reasoning
    }


def decompose_question(state: RAGState) -> RAGState:
    """
    NODE 2a: For COMPLEX questions only.
    Breaks the question into 2-4 simpler sub-questions.

    WHY THIS HELPS:
    A complex question like "Compare budget and luxury hotels
    near IU for a family" needs two separate retrievals:
    1. "budget hotels near Indiana University"
    2. "luxury hotels near Indiana University family amenities"

    Each sub-question gets its own retrieval pass so we find
    the best chunks for each part independently.
    """
    question = state["question"]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
Break this complex question about Bloomington, Indiana into
2-4 simple sub-questions. Each sub-question should be
independently answerable.

Original question: {question}

Return ONLY the sub-questions, one per line.
No numbering, no explanation, just the questions.""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    # Parse sub-questions
    sub_questions = [
        q.strip()
        for q in result.strip().split('\n')
        if q.strip() and len(q.strip()) > 10
    ]

    # Limit to 4 sub-questions
    sub_questions = sub_questions[:4]

    print(f"\n[NODE 2a - DECOMPOSE] Split into "
          f"{len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"  {i}. {q}")

    return {**state, "sub_questions": sub_questions}


def retrieve_simple(state: RAGState,
                    vector_store, bm25, reranker) -> RAGState:
    """
    NODE 2b: For SIMPLE questions.
    Direct retrieval — one query, one retrieval pass.
    Fast and efficient.
    """
    question = state["question"]
    docs = retrieve_for_query(
        question, vector_store, bm25, reranker
    )

    print(f"\n[NODE 2b - RETRIEVE SIMPLE] "
          f"Found {len(docs)} chunks")
    for i, (score, doc) in enumerate(docs, 1):
        src = doc.metadata.get(
            "source", "?"
        ).split("/")[-1].replace(".txt", "")
        print(f"  [{i}] score:{score:.2f} [{src}]")

    return {**state, "retrieved_docs": docs}


def retrieve_complex(state: RAGState,
                     vector_store, bm25, reranker) -> RAGState:
    """
    NODE 3a: For COMPLEX questions.
    Retrieves separately for each sub-question,
    then combines all results.

    This gives us the best chunks for EACH part of
    the complex question independently.
    """
    sub_questions = state.get("sub_questions", [])
    all_docs = []
    seen_content = set()

    print(f"\n[NODE 3a - RETRIEVE COMPLEX] "
          f"Retrieving for {len(sub_questions)} sub-questions")

    for i, sub_q in enumerate(sub_questions, 1):
        docs = retrieve_for_query(
            sub_q, vector_store, bm25, reranker, k=3
        )
        print(f"\n  Sub-question {i}: '{sub_q[:50]}'")

        for score, doc in docs:
            # Deduplicate — avoid same chunk twice
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                all_docs.append((score, doc))
                src = doc.metadata.get(
                    "source", "?"
                ).split("/")[-1].replace(".txt", "")
                print(f"    score:{score:.2f} [{src}]")

    print(f"\n  Total unique chunks: {len(all_docs)}")
    return {**state, "retrieved_docs": all_docs}


def format_context(state: RAGState) -> RAGState:
    """
    NODE 3b / 4: Formats retrieved docs into clean context.
    Extracts just the ANSWER portion from each Q&A chunk.
    """
    docs = state["retrieved_docs"]
    parts = []

    for i, (score, doc) in enumerate(docs, 1):
        text = doc.page_content
        if "ANSWER:" in text:
            answer_part = text.split("ANSWER:")[-1].strip()
            parts.append(f"[Info {i}]\n{answer_part}")
        else:
            parts.append(f"[Info {i}]\n{text[:400]}")

    context = "\n\n".join(parts)
    return {**state, "context": context}


def generate_answer(state: RAGState) -> RAGState:
    """
    FINAL NODE: Generates the answer using retrieved context.
    Uses a LARGE model here since this is the quality-critical step.
    """
    question = state["question"]
    context = state["context"]
    question_type = state["question_type"]
    llm = get_llm()

    if question_type == "complex":
        prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable Bloomington, Indiana tourist assistant.
This was a complex question requiring multiple pieces of information.
Provide a comprehensive, well-organized answer.

Information gathered:
{context}

Question: {question}

Provide a thorough answer that addresses all parts of the question.
Use the information above and organize your response clearly:""")
    else:
        prompt = ChatPromptTemplate.from_template("""
You are a friendly Bloomington, Indiana tourist assistant.
Answer the visitor's question using the information below.
Be helpful, specific, and concise.

Information:
{context}

Question: {question}

Answer:""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "context": context
    })

    print(f"\n[FINAL NODE - GENERATE] Answer generated")
    return {**state, "answer": answer}


# ============================================================
# ROUTING FUNCTION
# ============================================================

def route_question(state: RAGState) -> str:
    """
    Decides which path to take after classification.
    Returns the name of the next node.

    simple   → retrieve_simple → format → generate
    complex  → decompose → retrieve_complex → format → generate
    """
    if state["question_type"] == "complex":
        return "decompose"
    return "retrieve_simple"


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_graph(vector_store, bm25, reranker):
    """
    Assembles the LangGraph state machine.

    The graph looks like this:

    START
      |
      v
    classify_question
      |
      |-- simple --> retrieve_simple --> format_context --> generate --> END
      |
      |-- complex --> decompose --> retrieve_complex --> format_context --> generate --> END
    """

    # Wrap retrieval functions with their dependencies
    def retrieve_simple_node(state):
        return retrieve_simple(state, vector_store, bm25, reranker)

    def retrieve_complex_node(state):
        return retrieve_complex(state, vector_store, bm25, reranker)

    # Create the graph
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("classify", classify_question)
    graph.add_node("decompose", decompose_question)
    graph.add_node("retrieve_simple", retrieve_simple_node)
    graph.add_node("retrieve_complex", retrieve_complex_node)
    graph.add_node("format_context", format_context)
    graph.add_node("generate", generate_answer)

    # Set entry point
    graph.set_entry_point("classify")

    # Add conditional routing after classification
    graph.add_conditional_edges(
        "classify",
        route_question
    )

    # Simple path
    graph.add_edge("retrieve_simple", "format_context")

    # Complex path
    graph.add_edge("decompose", "retrieve_complex")
    graph.add_edge("retrieve_complex", "format_context")

    # Both paths converge here
    graph.add_edge("format_context", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ============================================================
# RUN
# ============================================================

def ask(graph, question):
    """Ask the graph a question and display the result."""
    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    initial_state = {
        "question": question,
        "question_type": "",
        "sub_questions": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "reasoning": ""
    }

    result = graph.invoke(initial_state)

    print(f"\nROUTING: {result['reasoning']}")
    print(f"\nANSWER:\n{result['answer']}")
    print("=" * 60)
    return result


def main():
    # Setup
    vector_store, bm25, reranker, chunks = setup_retrievers()
    graph = build_graph(vector_store, bm25, reranker)

    print("\n" + "=" * 60)
    print("LAYER 4: LANGGRAPH AGENTIC ROUTING")
    print("=" * 60)
    print("Watch how simple and complex questions")
    print("take different paths through the graph.")

    # Simple questions — should route directly to retrieval
    simple_questions = [
        "What budget hotels are near Indiana University?",
        "Where can I find good pizza in Bloomington?"
    ]

    # Complex questions — should decompose first
    complex_questions = [
        "Compare budget and mid-range hotels near IU and tell me which is better value for a family of four visiting in summer",
        "What should I do on my first day in Bloomington and where should I eat lunch and dinner?"
    ]

    print("\n--- SIMPLE QUESTIONS (direct retrieval) ---")
    for q in simple_questions:
        ask(graph, q)

    print("\n--- COMPLEX QUESTIONS (decompose first) ---")
    for q in complex_questions:
        ask(graph, q)


if __name__ == "__main__":
    main()