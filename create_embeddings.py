"""
All-in-one RAG embedding pipeline:
1. Load all transcript chunk JSON files from a folder
2. Create embeddings for each chunk using a local Ollama server (bge-m3)
3. Build a combined pandas DataFrame
4. Save it to disk
5. Display it in a clean, readable format
"""

import json
import os
import requests
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "bge-m3"
JSONS_FOLDER = "jsons"          # folder containing your per-video JSON files
OUTPUT_FILE = "embeddings.pkl"  # where the final DataFrame is saved


def create_embedding(text: str):
    r = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


def load_chunks_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # create_all_chunks.py format: {"number", "title", "full_text", "chunks": [...]}
    if isinstance(data, dict) and "chunks" in data:
        title = data.get("title", "")
        number = data.get("number", "")
        chunks = data["chunks"]
        for c in chunks:
            c.setdefault("title", title)
            c.setdefault("number", number)
        return chunks

    # stt.py format: plain list of chunks
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
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i + 1}/{len(chunks)}...")
        embedding = create_embedding(chunk["text"])

        rows.append({
            "chunk_id": i,
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
    """Embed a typed question and return the most relevant chunks."""
    incoming_query = input("\nAsk a Question: ")
    question_embedding = create_embedding(incoming_query)

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

    # --- Clean display (pandas defaults: head+tail with "..." divider) ---
    pd.set_option('display.max_rows', 10)
    print(f"\nShape: {df.shape}\n")
    print(df)

    # --- Test retrieval right away with a typed question ---
    ask_question(df)


if __name__ == "__main__":
    main()