"""
Full RAGAS evaluation — run this manually, not in CI.
Measures faithfulness and answer relevancy using LLM-as-judge.
Takes several minutes and uses Groq API.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from eval_dataset import EVAL_QUESTIONS
from rag_pipeline import setup_pipeline, retrieve, create_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def format_context(scored_docs):
    parts = []
    for i, (score, doc) in enumerate(scored_docs, 1):
        text = doc.page_content
        if "ANSWER:" in text:
            answer_part = text.split("ANSWER:")[-1].strip()
            parts.append(f"[Info {i}]\n{answer_part}")
        else:
            parts.append(f"[Info {i}]\n{text[:400]}")
    return "\n\n".join(parts)


def run_ragas_evaluation():
    print("=" * 60)
    print("RAGAS FULL EVALUATION")
    print("=" * 60)
    print("Setting up pipeline...")

    vector_store, bm25, reranker, chunks = setup_pipeline()

    # Use only first 10 questions to avoid rate limits
    eval_items = EVAL_QUESTIONS[:10]
    print(f"Evaluating {len(eval_items)} questions...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are a friendly Bloomington, Indiana tourist assistant.
Answer using ONLY the information provided below.

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

    for i, item in enumerate(eval_items, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i}/{len(eval_items)}] {question[:50]}...")

        # Retrieve
        docs = retrieve(question, vector_store, bm25, reranker)
        context = format_context(docs)

        # Generate
        answer = chain.invoke({
            "question": question,
            "context": context
        })

        # Collect context texts for RAGAS
        context_texts = [
            doc.page_content
            for _, doc in docs
        ]

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

    # Build dataset for RAGAS
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Setup RAGAS with Groq
    ragas_llm = LangchainLLMWrapper(ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    ))

    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    )

    # Configure metrics
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    print("\nRunning RAGAS evaluation...")
    print("(This takes a few minutes)")

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy]
    )

    print("\n" + "=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)
    print(f"Faithfulness:     {results['faithfulness']:.4f}")
    print(f"Answer Relevancy: {results['answer_relevancy']:.4f}")
    print()
    print("Score interpretation:")
    print("  0.8+ = Good    0.6-0.8 = Acceptable    <0.6 = Needs work")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_ragas_evaluation()