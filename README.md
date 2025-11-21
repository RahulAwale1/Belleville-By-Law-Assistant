# ⚖️ Belleville By-Law Assistant
### *A Retrieval-Augmented Generation (RAG) system with OCR, FAISS, FastAPI backend, and Streamlit/Gradio chat UIs*

This project transforms the City of Belleville’s public by-law documents—many stored as scanned PDFs—into a fully interactive AI assistant that answers municipal questions with accuracy, citations, and legal grounding.

## 🚀 Features

### 🔍 1. OCR + Text Processing Pipeline
- Extracts text from scanned PDFs using **Tesseract OCR** and `pdf2image`
- Cleans, normalizes, and segments by-laws into meaningful chunks
- Handles multi-page noise, broken formatting, and OCR errors

### 🧠 2. Semantic Embeddings + Vector Search
- Embeddings via **Sentence-Transformers (MiniLM-L6-v2)**
- High-speed semantic retrieval using **FAISS**
- Retrieves most relevant by-law sections for every query

### 🤖 3. Dual LLM Support
- **Local Llama3 via Ollama** — fast, offline, accurate
- **Zephyr-7B (Hugging Face)** — used for benchmarking and comparison
- Evaluated for grounding, latency, and hallucination rate

### 🧩 4. RAG Pipeline
```
OCR → Cleaning → Chunking → Embeddings → FAISS Search →
LLM (Ollama/Zephyr) → Structured Legal Answer
```

### 💬 5. Multiple User Interfaces
- **Streamlit Web App** frontend
- **FastAPI backend** for LLM + RAG inference
- **Gradio Chat UI** for local testing
- Structured answers: summary, bullet points, citations

## 🛠 Tech Stack

**Core NLP / RAG:**  
Python, Sentence-Transformers, FAISS, Tesseract OCR, Ollama (Llama3), Zephyr-7B

**Backend:**  
FastAPI, Pydantic, Uvicorn

**Frontend:**  
Streamlit, Gradio

## Quick start
Prereqs: Python 3.12+, Tesseract, Poppler, Ollama running with `ollama pull llama3`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Build data once (from the notebook):
1) Drop PDFs in `data/raw_pdfs/`
2) In `belleville_bylaw_bot.ipynb`, run cells to:
   - OCR → `data/ocr_json/*.json`
   - Chunk → `data/bylaw_chunks.json`
   - Embed + index → `data/bylaw_faiss.index`, `data/bylaw_metadata.json`

## ▶️ Running the Project

### 1. Start the FastAPI Backend
```
uvicorn backend.api:app --reload
```

### 2. Run the Streamlit UI
```
streamlit run ui/streamlit_app.py
```

### 2. Running the CLI (optional):
```bash
python rag_cli.py "What fees and charges does the city collect?"
```

## Env vars
- `OLLAMA_MODEL_NAME` (default `llama3`)
- `FASTAPI_URL` (Streamlit backend, default `http://localhost:8000/ask`)

## Code map
- `belleville_bylaw_bot.ipynb` — build pipeline (OCR, chunk, embed, index)
- `rag_backend.py` — shared RAG helpers
- `fastapi_app.py` — FastAPI `/ask`
- `streamlit_app.py` — chat UI calling FastAPI
- `rag_cli.py` — CLI client
- `data/` — PDFs (`raw_pdfs`), OCR JSON (`ocr_json`), chunks/index/metadata

## 🧠 Example Output
```
Summary: The city collects fees for change of ownership, account transfers,
waste items, water usage, and corporate searches.

• White goods with freon: $35  
• Large goods: $25  
• Bulky items: $150  
• Water rate (first 455m³): $1.99  
• Corporate search: $20  

_Source: By-Law 2024-201, pages 1 & 10_
```

## Notes
- Tesseract + Poppler must be installed and on PATH for OCR.
- Ollama daemon must be running for answers. Rebuild chunks/index if PDFs change.
