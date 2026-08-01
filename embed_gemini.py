import json
import os
from google import genai
import pandas as pd
import numpy as np
from google.genai.errors import ClientError
import time

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODEL_NAME = "gemini-embedding-001"

JSONS_FOLDER = "jsons"
OUTPUT_FILE = "embeddings_gemini.pkl"   # separate file so your working Ollama version stays intact

BATCH_SIZE = 10


def create_embeddings_batch(texts):
    while True:
        try:
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=texts,
            )
            return [embedding.values for embedding in response.embeddings]

        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print("Quota exceeded. Waiting 35 seconds...")
                time.sleep(35)
            else:
                raise


def load_chunks_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "chunks" in data:
        title = data.get("title", "")
        number = data.get("number", "")
        chunks = data["chunks"]
        for c in chunks:
            c.setdefault("title", title)
            c.setdefault("number", number)
        return chunks

    return data


def load_all_chunks(folder: str):
    all_chunks = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(folder, filename)
        chunks = load_chunks_from_file(filepath)
        print(f"  {filename}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


def build_dataframe(chunks):
    rows = []
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        print(f"Embedding chunks {start + 1} to {end} of {total}")

        batch = chunks[start:end]
        texts = [chunk["text"] for chunk in batch]
        embeddings = create_embeddings_batch(texts)

        for chunk, embedding in zip(batch, embeddings):
            rows.append({
                "chunk_id": len(rows),
                "number": chunk.get("number"),
                "title": chunk.get("title"),
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "text": chunk["text"],
                "embedding": embedding,
            })

    return pd.DataFrame(rows)


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def ask_question(df: pd.DataFrame, top_k: int = 3):
    incoming_query = input("\nAsk a Question: ")
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=[incoming_query],
    )
    question_embedding = response.embeddings[0].values

    similarities = df["embedding"].apply(
        lambda emb: cosine_similarity(question_embedding, emb)
    )

    results = df.copy()
    results["similarity"] = similarities
    top_results = results.sort_values("similarity", ascending=False).head(top_k)

    print("\nTop matching chunks:\n")
    for idx, row in top_results.iterrows():
        print(f"[{row['similarity']:.4f}] {row['title']} ({row['start']}s - {row['end']}s)")
        print(f"   {row['text'][:200]}...\n")


def main():
    print(f"Scanning '{JSONS_FOLDER}' for chunk files...")
    chunks = load_all_chunks(JSONS_FOLDER)
    print(f"\nTotal chunks loaded across all files: {len(chunks)}\n")

    if not chunks:
        print("No chunks found. Check JSONS_FOLDER path.")
        return

    df = build_dataframe(chunks)

    df.to_pickle(OUTPUT_FILE)
    print(f"\nSaved {len(df)} rows with embeddings to {OUTPUT_FILE}")

    pd.set_option('display.max_rows', 10)
    print(f"\nShape: {df.shape}\n")
    print(df)

    ask_question(df)


if __name__ == "__main__":
    main()
