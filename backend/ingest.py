"""PDF ingestion: extract, chunk, embed, and store in ChromaDB."""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
)
from backend.utils import logger


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: Optional[chromadb.PersistentClient] = None):
    if client is None:
        client = get_chroma_client()
    embed_fn = get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def load_pdf_documents(pdf_path: Path) -> list:
    """Load a single PDF with page-level metadata."""
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def chunk_documents(documents: list) -> list:
    """Split documents into overlapping chunks preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def ingest_pdf(pdf_path: Path, collection=None) -> int:
    """
    Ingest one PDF into ChromaDB. Returns number of chunks added.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if collection is None:
        collection = get_or_create_collection()

    # Replace prior chunks for the same file to avoid duplicate IDs on re-ingest
    try:
        existing = collection.get(where={"source": pdf_path.name})
        if existing and existing.get("ids"):
            collection.delete(ids=existing["ids"])
            logger.info("Removed %d old chunks for %s", len(existing["ids"]), pdf_path.name)
    except Exception as exc:
        logger.warning("Could not clear old chunks: %s", exc)

    logger.info("Ingesting PDF: %s", pdf_path.name)
    documents = load_pdf_documents(pdf_path)
    chunks = chunk_documents(documents)

    if not chunks:
        logger.warning("No text extracted from %s", pdf_path.name)
        return 0

    ids = []
    texts = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        # Chroma metadata values must be str, int, float, or bool
        chunk_id = f"{pdf_path.stem}_p{page}_c{idx}"
        ids.append(chunk_id)
        texts.append(chunk.page_content)
        metadatas.append(
            {
                "source": pdf_path.name,
                "page": int(page) + 1,  # 1-based page for display
                "chunk_id": chunk_id,
            }
        )

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    logger.info("Added %d chunks from %s", len(chunks), pdf_path.name)
    return len(chunks)


def ingest_all_pdfs(data_dir: Optional[Path] = None) -> dict:
    """Ingest all PDFs from the medical_pdfs directory."""
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return {"files_processed": 0, "total_chunks": 0, "files": []}

    total_chunks = 0
    processed = []
    for pdf_path in pdf_files:
        count = ingest_pdf(pdf_path, collection)
        total_chunks += count
        processed.append({"file": pdf_path.name, "chunks": count})

    return {
        "files_processed": len(processed),
        "total_chunks": total_chunks,
        "files": processed,
    }


if __name__ == "__main__":
    result = ingest_all_pdfs()
    print(result)
