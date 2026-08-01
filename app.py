"""
Streamlit UI for the course-video RAG assistant.

Run with:
    streamlit run app.py
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3"
EMBEDDINGS_FILE = "embeddings.pkl"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.55


# ---------------------------------------------------------------
# CORE RAG FUNCTIONS (same logic as rag_generate.py)
# ---------------------------------------------------------------
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

    return f"""You are a helpful teaching assistant. Answer the student's question
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


# ---------------------------------------------------------------
# DATA LOADING (cached so it doesn't reload on every question)
# ---------------------------------------------------------------
@st.cache_data
def load_embeddings():
    return pd.read_pickle(EMBEDDINGS_FILE)


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Course RAG Assistant", page_icon="🎓", layout="centered")

st.title("🎓 Course RAG Assistant")
st.caption("Ask questions about your course videos — answers are generated only from the transcript content.")

try:
    df = load_embeddings()
    st.success(f"Loaded {len(df)} chunks from {len(df['title'].unique())} videos.", icon="✅")
except FileNotFoundError:
    st.error(f"Could not find '{EMBEDDINGS_FILE}'. Run your embedding pipeline first.")
    st.stop()

# Keep chat history for the session
if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a question about your course...")

if query:
    with st.spinner("Retrieving relevant chunks..."):
        top_chunks = retrieve(query, df, TOP_K)
        best_score = top_chunks["similarity"].max()

    if best_score < SIMILARITY_THRESHOLD:
        answer = "This isn't covered in the course material I have access to."
    else:
        with st.spinner("Generating answer..."):
            prompt = build_prompt(query, top_chunks)
            answer = generate_answer(prompt)

    st.session_state.history.append({
        "query": query,
        "answer": answer,
        "chunks": top_chunks,
    })

# Render chat history (most recent last, like a normal chat)
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])

    with st.chat_message("assistant"):
        st.write(turn["answer"])

        with st.expander("Sources used"):
            for _, row in turn["chunks"].iterrows():
                st.markdown(
                    f"**{row['title']}** ({row['start']}s–{row['end']}s) "
                    f"— similarity: `{row['similarity']:.4f}`"
                )
                st.caption(row["text"][:250] + "...")