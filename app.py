"""Document Q&A RAG Agent - Streamlit UI.

This UI is a pure HTTP client of the FastAPI backend (api.py). It never
imports src/ and never calls the LLM directly - every upload and every
question is sent to the API over HTTP, and the API does all the real work.

Run the API first, then this UI, in two separate terminals:
    uvicorn api:app --reload
    streamlit run app.py
"""
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Q&A RAG Agent", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None  # assigned by the API on first upload
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


def api_get(path):
    return requests.get(f"{API_BASE_URL}{path}", timeout=10)


def api_post(path, **kwargs):
    return requests.post(f"{API_BASE_URL}{path}", timeout=120, **kwargs)


def api_delete(path):
    return requests.delete(f"{API_BASE_URL}{path}", timeout=10)


def error_detail(resp):
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Document Q&A RAG")
    st.caption("LangChain + LangGraph + Groq  •  via FastAPI")

    st.subheader("Backend status")
    try:
        health = api_get("/health").json()
        if health.get("groq_configured"):
            st.success(f"Connected to API at {API_BASE_URL}")
        else:
            st.error("API reachable, but GROQ_API_KEY is not configured on the server.")
    except requests.exceptions.RequestException:
        st.error(f"Could not reach the API at {API_BASE_URL}. Is `uvicorn api:app` running?")

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    index_clicked = st.button("📥 Index Documents", use_container_width=True)

    if index_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Indexing documents via API..."):
                errors = []
                total_chunks = 0
                for f in uploaded_files:
                    try:
                        files = {"file": (f.name, f.getvalue())}
                        data = {"thread_id": st.session_state.thread_id} if st.session_state.thread_id else {}
                        resp = api_post("/documents/upload", files=files, data=data)

                        if resp.status_code == 200:
                            body = resp.json()
                            st.session_state.thread_id = body["thread_id"]
                            total_chunks += body["chunks_indexed"]
                            if body["indexed_file"] not in st.session_state.indexed_files:
                                st.session_state.indexed_files.append(body["indexed_file"])
                        else:
                            errors.append(f"{f.name}: {error_detail(resp)}")

                    except requests.exceptions.RequestException as e:
                        errors.append(f"{f.name}: could not reach the API ({e})")

                if total_chunks:
                    st.success(
                        f"Indexed {total_chunks} chunks across {len(st.session_state.indexed_files)} file(s)."
                    )
                for err in errors:
                    st.error(err)

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for name in st.session_state.indexed_files:
            st.markdown(f"- {name}")

    st.divider()
    if st.button("🗑️ Clear Session / New Chat", use_container_width=True):
        if st.session_state.thread_id:
            try:
                api_delete(f"/documents/{st.session_state.thread_id}")
            except requests.exceptions.RequestException:
                pass  # best-effort cleanup; a fresh thread_id is minted next time regardless
        for key in ["thread_id", "messages", "indexed_files"]:
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Ask questions about your documents")

if not st.session_state.thread_id:
    st.info("Upload and index at least one document from the sidebar to get started.")


def render_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            page_str = f", page {s['page']}" if s.get("page") else ""
            chunk_str = f", chunk {s['chunk']}" if s.get("chunk") is not None else ""
            st.markdown(f"- **{s['source']}**{page_str}{chunk_str}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if not st.session_state.thread_id:
        st.error("Please upload and index a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = api_post(
                        "/chat",
                        json={"question": question, "thread_id": st.session_state.thread_id},
                    )
                    if resp.status_code == 200:
                        body = resp.json()
                        answer = body["answer"]
                        sources = body["sources"]
                    else:
                        answer = f"Sorry, the API returned an error: {error_detail(resp)}"
                        sources = []
                except requests.exceptions.RequestException as e:
                    answer = f"Sorry, could not reach the API: {e}"
                    sources = []

                st.markdown(answer)
                if sources:
                    render_sources(sources)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})