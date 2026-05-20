import pandas as pd
import ast
import os
import json


def load_and_convert_csv(
    csv_path="data/raw/bloomington_dataset_20241212_012231.csv"
):
    """
    Reads your original instruction-tuning CSV and converts
    each Q&A pair into a plain text document for RAG indexing.

    Your CSV format:
        id: bloomington_00000
        messages: [{'role': 'user', 'content': '...'},
                   {'role': 'assistant', 'content': '...'}]

    We convert each pair to:
        TOPIC: ...
        QUESTION: ...
        ANSWER: ...

    Each pair becomes one retrievable document in our RAG system.
    """

    print(f"Loading CSV...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} rows")

    documents = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            messages = ast.literal_eval(row['messages'])

            question = None
            answer = None

            for msg in messages:
                if msg['role'] == 'user':
                    question = msg['content'].strip()
                elif msg['role'] == 'assistant':
                    answer = msg['content'].strip()

            if not question or not answer:
                skipped += 1
                continue
            if len(answer) < 50:
                skipped += 1
                continue

            category = categorize(question)

            doc_text = f"""TOPIC: {category}
QUESTION: {question}
ANSWER: {answer}"""

            documents.append({
                "id": row['id'],
                "category": category,
                "question": question,
                "answer": answer,
                "text": doc_text
            })

        except Exception:
            skipped += 1
            continue

    print(f"Converted {len(documents)} documents ({skipped} skipped)")
    return documents


def categorize(question):
    q = question.lower()

    if any(w in q for w in ['hotel', 'motel', 'stay', 'accommodation',
                              'room', 'inn', 'suite', 'b&b', 'airbnb']):
        return "Hotels and Accommodation"
    elif any(w in q for w in ['restaurant', 'food', 'eat', 'dining',
                                'cuisine', 'lunch', 'dinner', 'breakfast',
                                'bar', 'cafe', 'coffee']):
        return "Restaurants and Dining"
    elif any(w in q for w in ['attraction', 'museum', 'park', 'lake',
                                'hiking', 'art', 'gallery', 'theater',
                                'entertainment', 'nightlife']):
        return "Attractions and Entertainment"
    elif any(w in q for w in ['transport', 'bus', 'drive', 'parking',
                                'airport', 'uber', 'lyft', 'taxi', 'bike']):
        return "Transportation"
    elif any(w in q for w in ['shop', 'store', 'mall', 'market',
                                'buy', 'purchase']):
        return "Shopping"
    elif any(w in q for w in ['event', 'festival', 'concert', 'game',
                                'iu', 'indiana university', 'hoosiers']):
        return "Events and University"
    elif any(w in q for w in ['weather', 'season', 'spring', 'summer',
                                'fall', 'winter', 'snow', 'rain']):
        return "Weather and Seasons"
    else:
        return "General Bloomington Info"


def save_documents(documents, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    # Count by category
    categories = {}
    for doc in documents:
        cat = doc['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("\nDocuments by category:")
    for cat, count in sorted(
        categories.items(), key=lambda x: -x[1]
    ):
        print(f"  {cat}: {count}")

    # Group Q&A pairs by category into one file each
    # This avoids creating 1000 individual files while
    # keeping topics separate for better retrieval
    category_docs = {}
    for doc in documents:
        cat = doc['category']
        if cat not in category_docs:
            category_docs[cat] = []
        category_docs[cat].append(doc['text'])

    saved_files = []
    for cat, texts in category_docs.items():
        filename = (
            cat.lower()
               .replace(' ', '_')
               .replace('&', 'and')
        )
        filepath = f"{output_dir}/{filename}.txt"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {cat} - Bloomington, Indiana\n\n")
            f.write("\n\n---\n\n".join(texts))

        saved_files.append(filepath)
        print(f"  Saved: {filepath} ({len(texts)} Q&A pairs)")

    # Save full dataset as JSON for reference
    json_path = f"{output_dir}/full_dataset.json"
    with open(json_path, 'w') as f:
        json.dump(documents, f, indent=2)
    print(f"\nFull dataset saved to {json_path}")

    return saved_files


def main():
    print("=" * 55)
    print("BLOOMINGTON DATA PREPARATION")
    print("=" * 55)

    documents = load_and_convert_csv()

    if not documents:
        print("No documents found. Check CSV path.")
        return

    saved_files = save_documents(documents)

    print(f"\nReady for RAG indexing")
    print(f"  {len(documents)} Q&A pairs converted")
    print(f"  {len(saved_files)} topic files created")
    print(f"  Files saved in: data/processed/")


if __name__ == "__main__":
    main()