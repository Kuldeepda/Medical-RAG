"""FastAPI backend for medical RAG Q&A."""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import (
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
    DATA_DIR,
    UNKNOWN_RESPONSE,
)
from backend.ingest import ingest_pdf
from backend.rag_pipeline import get_pipeline
from backend.utils import logger

app = FastAPI(
    title="Medical RAG API",
    description="AI-powered medical Q&A grounded in uploaded documents only.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    question: Optional[str] = None
    answer: str
    source: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None
    sources: Optional[list] = None
    grounded: bool = True
    mode: Optional[str] = None  # "llm" | "extractive"
    notice: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool = False
    llm_provider: str = "openai"


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


@app.get("/health", response_model=HealthResponse)
def health():
    from backend.config import LLM_PROVIDER, OPENAI_API_KEY

    llm_ok = LLM_PROVIDER == "ollama" or bool(
        OPENAI_API_KEY and OPENAI_API_KEY.strip()
    )
    return {
        "status": "running",
        "llm_configured": llm_ok,
        "llm_provider": LLM_PROVIDER,
    }


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    """Answer a medical question using RAG over ingested documents."""
    try:
        pipeline = get_pipeline()
        result = pipeline.generate_answer(body.question)
        return AskResponse(**result)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ask endpoint failed")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question.",
        ) from exc


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and ingest it into the vector database."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = DATA_DIR / Path(file.filename).name

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        shutil.move(str(tmp_path), str(dest_path))
        chunks_added = ingest_pdf(dest_path)

        if chunks_added == 0:
            return UploadResponse(
                filename=file.filename,
                chunks_added=0,
                message="PDF saved but no text could be extracted.",
            )

        # Reset pipeline so it picks up new collection data
        from backend import rag_pipeline as rag_module

        rag_module._pipeline = None

        return UploadResponse(
            filename=file.filename,
            chunks_added=chunks_added,
            message=f"Successfully ingested {chunks_added} chunks.",
        )
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
