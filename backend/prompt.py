"""Strict RAG prompts for context-only medical answering."""

from backend.config import UNKNOWN_RESPONSE

SYSTEM_PROMPT = f"""You are a medical AI assistant.
Answer ONLY using the provided context.
Do not use external knowledge.
If the answer is not clearly available in the context, reply exactly:
'{UNKNOWN_RESPONSE}'

Keep answers concise, factual, and medically professional."""

USER_PROMPT_TEMPLATE = """Context from medical documents:
{context}

Question: {question}

Answer using ONLY the context above. If the context does not contain enough information, reply exactly:
'{unknown_response}'"""


def build_user_prompt(context: str, question: str) -> str:
    """Build the user message with retrieved context."""
    return USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question.strip(),
        unknown_response=UNKNOWN_RESPONSE,
    )
