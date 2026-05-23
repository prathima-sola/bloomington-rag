"""
Full RAGAS evaluation — run manually, not in CI.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

print("Starting RAGAS evaluation...")

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from eval_dataset import EVAL_QUESTIONS
from rag_pipeline import setup_pipeline, retrieve
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def format_context(scored_docs):
    parts = []
    for i, (score, doc) in enumerate(scored_docs, 1):
        text = doc.page_content
        if "ANSWER:" in text:
            parts.append(f"[Info {i}]\n{text.split('ANSWER:')[-1].strip()}")
        else:
            parts.append(f"[Info {i}]\n{text[:400]}")
    return "\n\n".join(parts)


def run_ragas_evaluation():
    print("Setting up pipeline...")
    vector_store, bm25, reranker, chunks = setup_pipeline()

    eval_items = EVAL_QUESTIONS[:10]
    print(f"Evaluating {len(eval_items)} questions...")

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

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(eval_items, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"  [{i}/{len(eval_items)}] {question[:50]}...")

        try:
            docs = retrieve(question, vector_store, bm25, reranker)
            context = format_context(docs)
            answer = chain.invoke({
                "question": question,
                "context": context
            })
            context_texts = [doc.page_content for _, doc in docs]

            questions.append(question)
            answers.append(answer)
            contexts.append(context_texts)
            ground_truths.append(ground_truth)

        except Exception as e:
            print(f"    Error: {e}")
            continue

    print(f"\nSuccessfully processed {len(questions)} questions")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Setup RAGAS
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

    faith_metric = Faithfulness(llm=ragas_llm)
    relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    print("\nRunning RAGAS evaluation...")

    results = evaluate(
        dataset=dataset,
        metrics=[faith_metric, relevancy_metric],
        raise_exceptions=False
    )

    print("\n" + "=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)

    try:
        df = results.to_pandas()
        print(df[['question', 'faithfulness', 'answer_relevancy']].to_string())
        print()
        print(f"Average Faithfulness:     {df['faithfulness'].mean():.4f}")
        print(f"Average Answer Relevancy: {df['answer_relevancy'].mean():.4f}")
        print()
        print("Score interpretation:")
        print("  0.8+ = Good    0.6-0.8 = Acceptable    <0.6 = Needs work")
    except Exception as e:
        print(f"Error parsing results: {e}")
        print(f"Raw: {results}")

    print("=" * 60)


if __name__ == "__main__":
    run_ragas_evaluation()