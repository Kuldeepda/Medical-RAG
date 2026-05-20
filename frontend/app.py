"""Streamlit doctor dashboard for Medical RAG Q&A."""

import os
from datetime import datetime

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
UNKNOWN_PHRASE = "I don't know based on the provided medical data."

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False


def apply_theme():
    if st.session_state.dark_mode:
        st.markdown(
            """
            <style>
            .stApp { background-color: #0e1117; color: #fafafa; }
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea {
                background-color: #1c1f26; color: #fafafa;
            }
            .source-box {
                background: #1a2332; border-left: 4px solid #4da3ff;
                padding: 12px 16px; border-radius: 6px; margin-top: 12px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def check_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return {}


def ask_question(question: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def upload_pdf(file) -> bool:
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        r = requests.post(f"{API_URL}/upload", files=files, timeout=180)
        r.raise_for_status()
        data = r.json()
        st.success(data.get("message", "Upload complete."))
        return True
    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")
        return False


def render_source_citation(data: dict):
    source = data.get("source")
    page = data.get("page")
    score = data.get("score")

    if not source and UNKNOWN_PHRASE in data.get("answer", ""):
        st.info("No matching context in the medical knowledge base.")
        return

    score_text = f"{score:.2%}" if score is not None else "N/A"
    st.markdown(
        f"""
        <div class="source-box">
        <strong>Source citation</strong><br>
        Document: <code>{source or '—'}</code><br>
        Page: <code>{page or '—'}</code><br>
        Confidence (similarity): <code>{score_text}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sources = data.get("sources") or []
    if len(sources) > 1:
        with st.expander("All retrieved sources"):
            for i, s in enumerate(sources, 1):
                sc = s.get("score")
                sc_display = f"{sc:.2%}" if sc is not None else "N/A"
                st.write(
                    f"**{i}.** {s.get('source')} — page {s.get('page')} "
                    f"(score: {sc_display})"
                )


def main():
    init_session()
    apply_theme()

    st.title("🩺 Medical RAG Assistant")
    st.caption(
        "Answers are generated **only** from your uploaded medical documents. "
        "No external medical knowledge is used."
    )

    with st.sidebar:
        st.header("Settings")
        st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)
        apply_theme()

        st.divider()
        st.subheader("API status")
        health = check_health()
        if health.get("status") == "running":
            st.success(f"Backend online — `{API_URL}`")
            if not health.get("llm_configured"):
                st.warning(
                    "No OpenAI API key in `.env`. Answers use **document excerpts** only. "
                    "Add `OPENAI_API_KEY` and restart the API for AI summaries."
                )
        else:
            st.error(f"Backend offline — run `.\\start_all.ps1` or `.\\run_api.ps1`")

        st.divider()
        st.subheader("Upload medical PDF")
        uploaded = st.file_uploader("Add to knowledge base", type=["pdf"])
        if uploaded and st.button("Ingest PDF", use_container_width=True):
            with st.spinner("Indexing document…"):
                upload_pdf(uploaded)

        st.divider()
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_processed = None
            st.rerun()

        st.markdown("---")
        st.markdown(
            "**Note:** Low-confidence or missing context returns:\n\n"
            f"*\"{UNKNOWN_PHRASE}\"*"
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                notice = msg["meta"].get("notice")
                if notice:
                    st.info(notice)
                render_source_citation(msg["meta"])
                score = msg["meta"].get("score")
                if score is not None:
                    st.progress(min(float(score), 1.0))

    question = st.chat_input("Ask a medical question…")

    col1, col2 = st.columns([5, 1])
    with col1:
        manual_q = st.text_area(
            "Or type your question here",
            height=100,
            placeholder="e.g. What causes asthma?",
            label_visibility="collapsed",
        )
    with col2:
        submit = st.button("Submit", type="primary", use_container_width=True)

    active_question = question or (manual_q.strip() if submit and manual_q.strip() else None)

    if active_question and active_question != st.session_state.get("last_processed"):
        st.session_state.last_processed = active_question
        st.session_state.messages.append(
            {"role": "user", "content": active_question, "meta": None}
        )

        with st.spinner("Searching medical documents…"):
            result = ask_question(active_question)

        if result:
            answer = result.get("answer", UNKNOWN_PHRASE)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "meta": result,
                    "time": datetime.now().isoformat(),
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Could not reach the backend. Please try again.",
                    "meta": None,
                }
            )
        st.rerun()


if __name__ == "__main__":
    main()
