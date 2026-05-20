import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
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

load_dotenv()


# ============================================================
# RRF + BM25 (same as before)
# ============================================================

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


class BM25Retriever:
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


# ============================================================
# SETUP PIPELINE
# ============================================================

def setup_pipeline(data_dir="data/processed"):
    """
    Loads processed Bloomington Q&A documents,
    splits them into chunks, and builds all retrievers.
    """
    print("\n--- Loading Bloomington documents ---")

    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} topic files")

    # Split into chunks
    # Each chunk will contain 1-2 Q&A pairs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["---", "\n\n", "\n", ". "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Vector store
    print("\n--- Building vector store ---")
    print("(Embedding model downloads ~90MB on first run)")
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

    # BM25
    bm25 = BM25Retriever(chunks)
    print("BM25 retriever ready.")

    # Cross-encoder
    print("\n--- Loading cross-encoder ---")
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=512
    )
    print("Cross-encoder ready.")

    return vector_store, bm25, reranker, chunks


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
    vector_results = vector_store.similarity_search(query, k=8)
    bm25_results = bm25.retrieve(query, k=8)
    combined = reciprocal_rank_fusion([vector_results, bm25_results])

    # Cross-encoder reranking
    pairs = [(query, doc.page_content) for doc in combined]
    scores = reranker.predict(pairs)
    scored = sorted(
        zip(scores, combined),
        key=lambda x: x[0],
        reverse=True
    )

    # Return only confident results (score > 0)
    # If nothing is confident return top 2 anyway
    confident = [(s, d) for s, d in scored if s > 0]
    if len(confident) < 2:
        confident = scored[:2]

    return confident[:k]


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