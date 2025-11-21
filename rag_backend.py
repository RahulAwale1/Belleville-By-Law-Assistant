"""
Shared RAG helper functions for Belleville By-Law Bot.
Used by streamlit_app.py and fastapi_app.py
"""
import json
import os
from pathlib import Path

import faiss
import numpy as np
import ollama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Paths and model names
DATA_DIR = Path(__file__).parent / "data"
INDEX_PATH = DATA_DIR / "bylaw_faiss.index"
METADATA_PATH = DATA_DIR / "bylaw_metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3")


def load_index_and_metadata():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Missing FAISS index or metadata; build chunks/embeddings first.")
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def load_embed_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def retrieve_top_k(query: str, k: int, embed_model, index, metadata):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        info = metadata[idx]
        results.append(
            {
                "score": float(score),
                "id": info["id"],
                "file_name": info["file_name"],
                "bylaw_name": info.get("bylaw_name"),
                "page_number": info.get("page_number"),
                "text": info["text"],
            }
        )
    return results


def build_context_from_results(results):
    parts = []
    for r in results:
        header = f"[{r.get('bylaw_name')} | page {r.get('page_number')} | {r.get('file_name')}]"
        parts.append(header + "\n" + r["text"])
    return "\n\n---\n\n".join(parts)


def call_llm_with_context(question: str, context: str, max_new_tokens: int = 300) -> str:
    system_prompt = (
        "You are an assistant that answers questions about City of Belleville by-laws.\n"
        "Use ONLY the information in the provided CONTEXT.\n"
        "If the answer is not clearly in the context, say you are not sure and suggest "
        "contacting the City of Belleville for confirmation.\n\n"
        "Format your answer as:\n"
        "- First, a short 1–2 sentence summary in plain language.\n"
        "- Then, a bullet list focusing ONLY on fees or dollar amounts relevant to the question.\n"
        "- Each bullet must be of the form: '<item or waste type>: $<amount> (short description if needed)'.\n"
        "- Do NOT include operational rules (like number of lifts, tag requirements, schedule rules) in the bullet list;\n"
        "  those can be mentioned in the summary if important.\n"
        "- Do NOT leave any bullet unfinished. Do NOT end the answer with a hanging '-' or incomplete sentence.\n"
        "Do NOT invent information that is not supported by the context.\n"
        "Do NOT ask or answer extra questions. Just answer the user's question.\n"
    )

    user_message = (
        f"Here is the context from Belleville by-laws:\n\n"
        f"{context}\n\n"
        f"Based ONLY on the context above, answer this question clearly and concisely:\n"
        f"{question}"
    )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"num_predict": max_new_tokens, "temperature": 0.2},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"LLM call failed: {e}"


def answer_question_with_rag(question: str, k: int, embed_model=None, index=None, metadata=None):
    load_dotenv()

    if embed_model is None or index is None or metadata is None:
        index, metadata = load_index_and_metadata()
        embed_model = load_embed_model()

    q_lower = question.lower().strip()
    greeting_phrases = ["hello", "hi", "hey", "what can you do", "who are you", "help", "how can you help me"]
    if any(p in q_lower for p in greeting_phrases):
        intro_answer = (
            "Hi! I’m the Belleville By-Law Assistant.\n\n"
            "- I can help you understand City of Belleville by-laws in plain language.\n"
            "- You can ask about things like parking rules, noise restrictions, property standards,\n"
            "  fees and charges, animals/pets, streets and driveways, and other municipal rules.\n"
            "- I’ll look up the relevant by-law sections and summarize them for you, and I’ll tell you\n"
            "  which by-law and page the information comes from.\n\n"
            "Try asking something like:\n"
            "- \"Can I park on the street overnight in Belleville?\"\n"
            "- \"What are the fees for a change of ownership?\"\n"
            "- \"Are there any noise restrictions at night?\""
        )
        return {"answer": intro_answer, "sources": []}

    results = retrieve_top_k(question, k=k, embed_model=embed_model, index=index, metadata=metadata)
    if not results:
        return {
            "answer": (
                "I couldn’t find any relevant sections in the by-laws for that question. "
                "Try asking specifically about parking, noise, fees, property standards, "
                "streets/driveways, or other Belleville by-laws."
            ),
            "sources": [],
        }

    if results[0]["score"] < 0.35:
        return {
            "answer": (
                "I checked the by-laws I have, but nothing seems clearly related to that question.\n\n"
                "I’m best at answering questions specifically about City of Belleville by-laws — "
                "for example parking rules, noise by-laws, fees and charges, or property-related rules."
            ),
            "sources": [],
        }

    context = build_context_from_results(results)
    main_answer = call_llm_with_context(question, context)

    sources = [
        {
            "file_name": r["file_name"],
            "bylaw_name": r["bylaw_name"],
            "page_number": r["page_number"],
            "score": r["score"],
        }
        for r in results
    ]

    seen = set()
    source_bits = []
    for s in sources:
        key = (s["bylaw_name"], s["page_number"])
        if key in seen:
            continue
        seen.add(key)
        if s["bylaw_name"] is not None and s["page_number"] is not None:
            source_bits.append(f"{s['bylaw_name']} (page {s['page_number']})")

    full_answer = main_answer
    if source_bits:
        full_answer += "\n\n" + "_Source by-laws: " + ", ".join(source_bits) + "._"

    return {"answer": full_answer, "sources": sources}
