#!/usr/bin/env python
"""
Streamlit UI for the Belleville By-Law Bot (RAG + Ollama).
Prereqs:
  - Built artifacts: data/bylaw_faiss.index and data/bylaw_metadata.json
  - Ollama daemon running locally with the desired model (default: llama3)
Run:
  streamlit run streamlit_app.py
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

# FastAPI endpoint
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/ask")


def main():
    load_dotenv()
    st.title("Belleville By-Law Assistant")
    st.caption("Ask questions about City of Belleville by-laws. Uses local embeddings + FAISS + Ollama.")

    # simple controls
    cols = st.columns([1, 1, 1])
    with cols[2]:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    # simple chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display existing messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # chat input
    user_input = st.chat_input("Ask about Belleville by-laws...")

    if user_input:
        # add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # assistant placeholder while generating
        assistant_msg = st.chat_message("assistant")
        placeholder = assistant_msg.empty()
        placeholder.markdown("_Assistant is typing..._")

        # call FastAPI backend
        try:
            resp = requests.post(
                FASTAPI_URL,
                json={"question": user_input.strip(), "k": 5},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            error_text = f"Backend call failed: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_text})
            placeholder.markdown(error_text)
            return

        answer = result["answer"]

        # Build a single combined assistant message including sources
        sources_md = ""
        if result.get("sources"):
            lines = []
            for s in result["sources"]:
                lines.append(
                    f"- {s['file_name']} | {s.get('bylaw_name')} | page {s.get('page_number')} | score={s['score']:.3f}"
                )
            sources_md = "\n\n**Sources:**\n" + "\n".join(lines)

        full_response = answer + sources_md

        # persist combined message
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # render combined message
        placeholder.markdown(full_response)


if __name__ == "__main__":
    main()
