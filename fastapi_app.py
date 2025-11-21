"""
FastAPI service for the Belleville By-Law Bot.
Start with:
    uvicorn fastapi_app:app --reload
Prereqs: build data/bylaw_faiss.index and data/bylaw_metadata.json, run Ollama daemon.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from rag_backend import (
    answer_question_with_rag,
    load_embed_model,
    load_index_and_metadata,
)

app = FastAPI(title="Belleville By-Law Bot API")


class Query(BaseModel):
    question: str
    k: int = 5


# Load heavy resources once
index, metadata = load_index_and_metadata()
embed_model = load_embed_model()


@app.post("/ask")
def ask(query: Query):
    result = answer_question_with_rag(
        query.question, k=query.k, embed_model=embed_model, index=index, metadata=metadata
    )
    return result
