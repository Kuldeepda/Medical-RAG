# Medical RAG — AI-Powered Medical Question Answering

A production-oriented **Retrieval-Augmented Generation (RAG)** system for doctors. Questions are answered **only** from uploaded medical PDFs. If context is missing or similarity is too low, the system responds exactly:

> **I don't know based on the provided medical data.**

## Architecture

```
Frontend (Streamlit)
        ↓
FastAPI Backend
        ↓
sentence-transformers (all-MiniLM-L6-v2)
        ↓
ChromaDB (cosine similarity, top-5 chunks)
        ↓
Similarity threshold (default 0.75)
        ↓
LLM (OpenAI GPT / Ollama Llama 3) with strict system prompt
        ↓
Grounded answer + source citation
```

## Features

- PDF ingestion with chunking (500 / 100 overlap)
- Vector search with similarity threshold
- Strict anti-hallucination prompts
- Source filename, page, and confidence score
- `POST /ask`, `POST /upload`, `GET /health`
- Streamlit dashboard (dark mode, chat history, PDF upload)
- Docker & deployment guides (Render / Railway)

## Project structure

```
medical-rag/
├── backend/
│   ├── main.py           # FastAPI routes
│   ├── rag_pipeline.py   # Retrieve + generate
│   ├── ingest.py         # PDF → ChromaDB
│   ├── prompt.py         # Strict prompts
│   ├── config.py         # Environment config
│   └── utils.py          # Logging, similarity helpers
├── frontend/
│   └── app.py            # Streamlit UI
├── data/medical_pdfs/    # Source PDFs
├── chroma_db/            # Persistent vector store
├── scripts/
│   └── create_sample_pdf.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Installation

### 1. Clone and enter project

```bash
cd medical-rag
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` and set `OPENAI_API_KEY` (or switch to `LLM_PROVIDER=ollama`).

### 4. Add sample medical PDFs

```bash
python scripts/create_sample_pdf.py
python -m backend.ingest
```

Or place your own PDFs in `data/medical_pdfs/` and run ingest.

### 5. Start backend

From the `medical-rag` folder (so `PYTHONPATH` includes the project root):

```bash
set PYTHONPATH=.          # Windows CMD
$env:PYTHONPATH="."       # Windows PowerShell
export PYTHONPATH=.       # macOS/Linux

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start frontend

```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501**

## API usage

### Health

```bash
curl http://localhost:8000/health
```

### Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What causes asthma?\"}"
```

Example response:

```json
{
  "question": "What causes asthma?",
  "answer": "Asthma is caused by airway inflammation...",
  "source": "sample_medical_reference.pdf",
  "page": 1,
  "score": 0.87,
  "sources": [...],
  "grounded": true
}
```

### Upload PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/medical_pdfs/sample_medical_reference.pdf"
```

Interactive docs: **http://localhost:8000/docs**

## Hallucination prevention

1. **Similarity threshold** — best chunk must score ≥ `0.75` (configurable via `SIMILARITY_THRESHOLD`)
2. **Strict system prompt** — LLM instructed to use context only
3. **Refusal detection** — model answers containing the refusal phrase are normalized
4. **No retrieval → no answer** — empty or weak context returns the mandatory unknown message

## Docker

```bash
docker compose up --build
```

- API: http://localhost:8000  
- UI: http://localhost:8501  

Ensure `.env` exists before `docker compose up`.

## Deployment

### Render

1. Create a **Web Service** from this repo.
2. **Build command:** `pip install -r requirements.txt`
3. **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add env vars: `OPENAI_API_KEY`, `PYTHONPATH=/opt/render/project/src`
5. Use a **persistent disk** mounted at `/app/chroma_db` and `/app/data` for vectors and PDFs.
6. Deploy Streamlit as a second service with `streamlit run frontend/app.py --server.port=$PORT` and `API_URL` pointing to the API service.

### Railway

1. New project → Deploy from GitHub.
2. Set root directory to `medical-rag`.
3. Add variables from `.env.example`.
4. API service start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add a volume for `chroma_db` and `data/medical_pdfs`.
6. Optional second service for Streamlit frontend.

## Screenshots

After starting Streamlit, you will see:

- Question input and chat interface
- Answer panel with source document and page
- Confidence progress bar (similarity score)
- Sidebar PDF upload and dark mode toggle

*(Add screenshots to `docs/screenshots/` after your first run.)*

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Chat model |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `SIMILARITY_THRESHOLD` | `0.75` | Min cosine similarity to answer |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |

## Future improvements

- Hybrid search (BM25 + vector) and re-ranking
- Medical term highlighting in answers
- Conversation memory across turns
- Doctor authentication (OAuth2 / JWT)
- Citation highlighting in PDF viewer
- Multi-tenant document collections

## Disclaimer

This software is for **educational and decision-support prototyping** only. It is not a substitute for clinical judgment, licensed medical advice, or regulated medical devices. Always verify information against authoritative sources.

## License

MIT (adjust as needed for your organization).
