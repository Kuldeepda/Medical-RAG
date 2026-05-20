"""RAG pipeline: retrieve, threshold-filter, and generate grounded answers."""

from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from backend.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K,
    UNKNOWN_RESPONSE,
    USE_EXTRACTIVE_FALLBACK,
)
from backend.ingest import get_or_create_collection
from backend.prompt import SYSTEM_PROMPT, build_user_prompt
from backend.utils import (
    build_extractive_answer,
    distance_to_similarity,
    filter_relevant_chunks,
    format_context_block,
    is_refusal_answer,
    logger,
    normalize_medical_query,
    sanitize_question,
)


class MedicalRAGPipeline:
    """End-to-end RAG for medical Q&A with strict grounding."""

    def __init__(self):
        self.collection = get_or_create_collection()
        self._llm = None

    def _llm_configured(self) -> bool:
        if LLM_PROVIDER == "ollama":
            return True
        return bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        if LLM_PROVIDER == "ollama":
            self._llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0,
            )
        else:
            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
                )
            self._llm = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0,
            )
        return self._llm

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        """Query ChromaDB and return chunks with similarity scores."""
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict[str, Any]] = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return chunks

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            similarity = distance_to_similarity(dist, space="cosine")
            chunks.append(
                {
                    "text": doc,
                    "metadata": meta or {},
                    "distance": dist,
                    "similarity": similarity,
                }
            )

        chunks.sort(key=lambda c: c["similarity"], reverse=True)
        return chunks

    def _unknown_response(
        self, question: str, reason: str = "insufficient_context"
    ) -> dict[str, Any]:
        logger.info("Returning unknown response for question (reason=%s)", reason)
        return {
            "question": question,
            "answer": UNKNOWN_RESPONSE,
            "source": None,
            "page": None,
            "score": None,
            "sources": [],
            "grounded": False,
            "mode": None,
            "notice": None,
        }

    def _build_sources_list(self, chunks: list[dict[str, Any]]) -> list[dict]:
        return [
            {
                "source": c["metadata"].get("source"),
                "page": c["metadata"].get("page"),
                "score": round(c["similarity"], 4),
                "chunk_id": c["metadata"].get("chunk_id"),
            }
            for c in chunks
        ]

    def _grounded_response(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        answer_text: str,
        mode: str,
        notice: Optional[str] = None,
    ) -> dict[str, Any]:
        primary = chunks[0]
        meta = primary.get("metadata", {})
        best_score = chunks[0]["similarity"]

        return {
            "question": question,
            "answer": answer_text.strip(),
            "source": meta.get("source"),
            "page": meta.get("page"),
            "score": round(best_score, 4),
            "sources": self._build_sources_list(chunks),
            "grounded": True,
            "mode": mode,
            "notice": notice,
        }

    def _extractive_notice(self, reason: str) -> str:
        if reason == "no_key":
            return (
                "OpenAI API key missing in `.env`. Showing document excerpts only. "
                "Add `OPENAI_API_KEY` and restart the API for summarized answers."
            )
        return (
            "AI summarization unavailable (quota, billing, or connection error). "
            "Showing matching excerpts from your documents only."
        )

    def _extractive_response(
        self,
        question: str,
        chunks: list[dict[str, Any]],
        reason: str = "llm_error",
    ) -> dict[str, Any]:
        answer_text = build_extractive_answer(chunks)
        if is_refusal_answer(answer_text):
            return self._unknown_response(question, reason="empty_extractive")

        return self._grounded_response(
            question,
            chunks,
            answer_text,
            mode="extractive",
            notice=self._extractive_notice(reason),
        )

    def generate_answer(self, question: str) -> dict[str, Any]:
        """
        Full RAG flow with similarity threshold and strict prompting.
        Falls back to extractive (document-only) answers when LLM is unavailable.
        """
        question = sanitize_question(question)
        if not question:
            return self._unknown_response(question, reason="empty_question")

        search_query = normalize_medical_query(question)
        if search_query != question.lower():
            logger.info("Query normalized: %r -> %r", question, search_query)

        chunks = self.retrieve(search_query)
        if not chunks:
            return self._unknown_response(question, reason="no_chunks")

        best_score = chunks[0]["similarity"]
        if best_score < SIMILARITY_THRESHOLD:
            logger.info(
                "Best similarity %.3f below threshold %.3f",
                best_score,
                SIMILARITY_THRESHOLD,
            )
            return self._unknown_response(question, reason="low_similarity")

        chunks = filter_relevant_chunks(chunks, SIMILARITY_THRESHOLD, max_chunks=2)
        if not chunks:
            return self._unknown_response(question, reason="low_similarity")

        # No API key: use document excerpts instead of falsely saying "I don't know"
        if USE_EXTRACTIVE_FALLBACK and not self._llm_configured():
            logger.warning(
                "LLM not configured; using extractive fallback (score=%.3f)",
                best_score,
            )
            return self._extractive_response(question, chunks, reason="no_key")

        context = format_context_block(chunks)
        user_prompt = build_user_prompt(context, question)

        try:
            llm = self._get_llm()
            response = llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            answer_text = (
                response.content if hasattr(response, "content") else str(response)
            )
        except Exception as exc:
            logger.exception("LLM generation failed: %s", exc)
            if USE_EXTRACTIVE_FALLBACK:
                return self._extractive_response(question, chunks, reason="llm_error")
            return self._unknown_response(question, reason="llm_error")

        if is_refusal_answer(answer_text):
            return self._unknown_response(question, reason="model_refusal")

        return self._grounded_response(question, chunks, answer_text, mode="llm")


_pipeline: Optional[MedicalRAGPipeline] = None


def get_pipeline() -> MedicalRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MedicalRAGPipeline()
    return _pipeline
