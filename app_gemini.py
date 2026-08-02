"""
Streamlit UI for the course-video RAG assistant (Gemini version).

Run with:
    streamlit run app_gemini.py
"""

import os
from google import genai
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets else os.environ.get("GEMINI_API_KEY")
client = genai.Client(
    api_key=api_key
)

EMBED_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.6-flash"
EMBEDDINGS_FILE = "embeddings_gemini.pkl"   # must match the file embed_gemini.py saved
TOP_K = 3
SIMILARITY_THRESHOLD = 0.55


# ---------------------------------------------------------------
# CORE RAG FUNCTIONS
# ---------------------------------------------------------------
def create_embedding(text: str):
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
    )
    return response.embeddings[0].values


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
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text


# ---------------------------------------------------------------
# DATA LOADING
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
    st.error(f"Could not find '{EMBEDDINGS_FILE}'. Run embed_gemini.py first.")
    st.stop()

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
