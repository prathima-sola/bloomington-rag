"""
Manual faithfulness evaluation using LLM-as-judge.
Measures whether answers are grounded in retrieved context.
Does not use RAGAS — works directly with Groq.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from eval_dataset import EVAL_QUESTIONS
from rag_pipeline import setup_pipeline, retrieve


def format_context(scored_docs):
    parts = []
    for i, (score, doc) in enumerate(scored_docs, 1):
        text = doc.page_content
        if "ANSWER:" in text:
            parts.append(f"[Info {i}]\n{text.split('ANSWER:')[-1].strip()}")
        else:
            parts.append(f"[Info {i}]\n{text[:400]}")
    return "\n\n".join(parts)


def evaluate_faithfulness():
    print("=" * 60)
    print("FAITHFULNESS EVALUATION")
    print("=" * 60)
    print("Measures: does the answer contain only information")
    print("from the retrieved context, or is it hallucinating?")
    print("=" * 60)

    # Setup pipeline
    vector_store, bm25, reranker, chunks = setup_pipeline()

    # LLM for generation
    gen_llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # LLM for evaluation (judge)
    judge_llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    # Generation prompt
    gen_prompt = ChatPromptTemplate.from_template("""
You are a friendly Bloomington, Indiana tourist assistant.
Answer using ONLY the information provided below.
If the information doesn't contain the answer, say so.

Information:
{context}

Question: {question}

Answer:""")

    # Faithfulness judge prompt
    judge_prompt = ChatPromptTemplate.from_template("""
You are evaluating whether an AI answer is faithful to its source context.

CONTEXT (what the AI was given):
{context}

QUESTION: {question}

ANSWER (what the AI said):
{answer}

Task: Check if every claim in the ANSWER is supported by the CONTEXT.
An answer is faithful if it only uses information from the context.
An answer is NOT faithful if it adds facts not in the context.

Respond with ONLY a JSON object like this:
{{"score": 0.9, "reason": "brief explanation"}}

Score 1.0 = completely faithful, all claims in context
Score 0.5 = partially faithful, some claims not in context  
Score 0.0 = not faithful, answer ignores context entirely

JSON response:""")

    gen_chain = gen_prompt | gen_llm | StrOutputParser()
    judge_chain = judge_prompt | judge_llm | StrOutputParser()

    results = []
    eval_items = EVAL_QUESTIONS[:10]

    print(f"\nEvaluating {len(eval_items)} questions...\n")

    for i, item in enumerate(eval_items, 1):
        question = item["question"]
        print(f"[{i}/{len(eval_items)}] {question[:55]}...")

        try:
            # Retrieve
            docs = retrieve(question, vector_store, bm25, reranker)
            context = format_context(docs)

            # Generate answer
            answer = gen_chain.invoke({
                "question": question,
                "context": context
            })

            # Judge faithfulness
            judgment = judge_chain.invoke({
                "question": question,
                "context": context,
                "answer": answer
            })

            # Parse score
            import json
            import re
            json_match = re.search(r'\{.*\}', judgment, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                score = float(parsed.get("score", 0.5))
                reason = parsed.get("reason", "")
            else:
                score = 0.5
                reason = "Could not parse judgment"

            results.append({
                "question": question[:50],
                "score": score,
                "reason": reason,
                "answer_preview": answer[:100]
            })

            print(f"  Score: {score:.2f} — {reason[:60]}")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "question": question[:50],
                "score": 0.0,
                "reason": f"Error: {e}",
                "answer_preview": ""
            })

    # Summary
    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0

    print("\n" + "=" * 60)
    print("FAITHFULNESS RESULTS")
    print("=" * 60)
    print(f"\n{'Question':<52} {'Score'}")
    print("-" * 60)
    for r in results:
        bar = "█" * int(r["score"] * 10)
        print(f"{r['question']:<52} {r['score']:.2f} {bar}")

    print("-" * 60)
    print(f"\nAverage Faithfulness Score: {avg:.4f}")
    print()
    if avg >= 0.8:
        print("✓ GOOD — answers are grounded in retrieved context")
    elif avg >= 0.6:
        print("⚠ ACCEPTABLE — some answers may include external knowledge")
    else:
        print("✗ NEEDS WORK — answers frequently go beyond context")

    print()
    print("Score interpretation:")
    print("  1.0 = all claims supported by context")
    print("  0.5 = some claims not in context")
    print("  0.0 = answer ignores context entirely")
    print("=" * 60)

    return avg


if __name__ == "__main__":
    score = evaluate_faithfulness()