"""
Generation step (the "G" in RAG):
1. Retrieve the most relevant chunks for a question (same as before)
2. Build a prompt that includes those chunks as context
3. Send the prompt to a local LLM via Ollama
4. Print the model's answer, grounded in your transcript data
"""

import requests
import pandas as pd
import numpy as np

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3"          # change this to whichever chat model you've pulled in Ollama
EMBEDDINGS_FILE = "embeddings.pkl"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.55   # below this, treat the topic as not covered


def create_embedding(text: str):
    r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def retrieve(query: str, df: pd.DataFrame, top_k: int = TOP_K):
    query_embedding = create_embedding(query)
    similarities = df["embedding"].apply(lambda emb: cosine_similarity(query_embedding, emb))

    results = df.copy()
    results["similarity"] = similarities
    return results.sort_values("similarity", ascending=False).head(top_k)


def build_prompt(query: str, top_chunks: pd.DataFrame) -> str:
    context_blocks = []
    for _, row in top_chunks.iterrows():
        context_blocks.append(f"[From: {row['title']}]\n{row['text']}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful teaching assistant. Answer the student's question
using ONLY the context provided below.

Strict rules:
- If the context does not clearly and directly answer the question, respond
  with EXACTLY: "This isn't covered in the course material I have access to."
- Do NOT guess, speculate, or say things like "might be related to" or
  "educated guess" — either the context answers it, or it doesn't.
- Do NOT reason out loud about what could be inferred. Just answer directly
  or give the exact fallback line above.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def generate_answer(prompt: str) -> str:
    r = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    df = pd.read_pickle(EMBEDDINGS_FILE)
    print(f"Loaded {len(df)} chunks from {EMBEDDINGS_FILE}")

    query = input("\nAsk a Question: ")

    top_chunks = retrieve(query, df, TOP_K)
    print(f"\nRetrieved {len(top_chunks)} relevant chunks:")
    for _, row in top_chunks.iterrows():
        print(f"  [{row['similarity']:.4f}] {row['title']} ({row['start']}s-{row['end']}s)")

    best_score = top_chunks["similarity"].max()
    if best_score < SIMILARITY_THRESHOLD:
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("This isn't covered in the course material I have access to.")
        print("=" * 60)
        return

    prompt = build_prompt(query, top_chunks)

    print("\nGenerating answer...\n")
    answer = generate_answer(prompt)

    print("=" * 60)
    print("ANSWER:")
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()