import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
try:
    from src.retrieval import RetrievalConfig, RetrievalEngine
except ModuleNotFoundError:  # direct execution with src/ on sys.path
    from retrieval import RetrievalConfig, RetrievalEngine

load_dotenv()


# ============================================================
# SETUP PIPELINE
# ============================================================

def setup_pipeline(data_dir=None):
    """
    Loads processed Bloomington Q&A documents,
    splits them into chunks, and builds all retrievers.
    """
    print("\n--- Loading shared Bloomington retrieval engine ---")
    engine = RetrievalEngine(
        RetrievalConfig(strategy="hybrid", rerank=True, top_k=4),
        data_dir=data_dir,
    )
    print(f"Created {len(engine.chunks)} chunks")
    return engine, None, None, engine.chunks


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(query, vector_store, bm25, reranker, k=4):
    """
    Full 3-layer retrieval:
    1. Vector search (8 candidates)
    2. BM25 keyword search (8 candidates)
    3. RRF combines both lists
    4. Cross-encoder reranks, returns top 4
    """
    if not isinstance(vector_store, RetrievalEngine):
        raise TypeError("setup_pipeline() must supply the shared RetrievalEngine")
    return vector_store.retrieve_legacy(query, top_k=k)


# ============================================================
# GENERATE ANSWER
# ============================================================

def create_chain():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are a friendly and knowledgeable Bloomington, Indiana tourist assistant.
Answer the visitor's question using ONLY the information provided below.

Rules:
- Be helpful, specific, and friendly
- If exact information is not available, say so honestly
- Suggest practical next steps when relevant
- Keep answers concise and useful for a visitor

Information about Bloomington:
{context}

Visitor's question: {question}

Answer:""")

    chain = (
        RunnableLambda(lambda x: {
            "context": format_context(x["docs"]),
            "question": x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def format_context(scored_docs):
    """Formats retrieved Q&A pairs as clean context."""
    parts = []
    for i, (score, doc) in enumerate(scored_docs, 1):
        # Extract just the ANSWER part from each Q&A chunk
        text = doc.page_content
        if "ANSWER:" in text:
            answer_part = text.split("ANSWER:")[-1].strip()
            parts.append(f"[Info {i}]\n{answer_part}")
        else:
            parts.append(f"[Info {i}]\n{text[:400]}")
    return "\n\n".join(parts)


# ============================================================
# INTERACTIVE CHATBOT
# ============================================================

def run_chatbot(vector_store, bm25, reranker):
    """
    Simple interactive loop.
    Type a question, get an answer.
    Type 'quit' to exit.
    """
    chain = create_chain()

    print("\n" + "=" * 55)
    print("BLOOMINGTON TOURIST ASSISTANT")
    print("=" * 55)
    print("Ask me anything about Bloomington, Indiana!")
    print("Type 'quit' to exit")
    print("=" * 55)

    while True:
        print()
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! Enjoy Bloomington!")
            break

        # Retrieve relevant context
        docs = retrieve(question, vector_store, bm25, reranker)

        if not docs:
            print("Assistant: I don't have enough information "
                  "to answer that question.")
            continue

        # Generate answer
        answer = chain.invoke({
            "question": question,
            "docs": docs
        })

        print(f"\nAssistant: {answer}")


# ============================================================
# TEST WITH SAMPLE QUESTIONS
# ============================================================

def run_test(vector_store, bm25, reranker):
    """
    Tests the pipeline with sample questions
    covering different topics in your dataset.
    """
    chain = create_chain()

    questions = [
        "What are some good budget hotels near Indiana University?",
        "Where can I find good restaurants in Bloomington?",
        "What attractions should I visit in Bloomington?",
        "How can I get around Bloomington without a car?",
        "What events happen at Indiana University?"
    ]

    print("\n" + "=" * 55)
    print("BLOOMINGTON RAG - TEST RUN")
    print("=" * 55)

    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 55)

        docs = retrieve(question, vector_store, bm25, reranker)

        # Show what was retrieved
        print(f"Retrieved {len(docs)} chunks:")
        for i, (score, doc) in enumerate(docs, 1):
            source = doc.metadata.get(
                "source", "?"
            ).split("/")[-1].replace(".txt", "")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  [{i}] score:{score:.2f} [{source}]"
                  f" {preview}...")

        # Generate answer
        answer = chain.invoke({
            "question": question,
            "docs": docs
        })
        print(f"\nA: {answer}")
        print("=" * 55)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    # Setup
    vector_store, bm25, reranker, chunks = setup_pipeline()

    # Run test or interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        run_chatbot(vector_store, bm25, reranker)
    else:
        run_test(vector_store, bm25, reranker)
