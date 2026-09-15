"""Document Q&A RAG Agent - Streamlit app.

Simple single-file UI on top of the LangChain / LangGraph pipeline in `src/`.
Run with:  streamlit run app.py
"""
import os
import uuid

import streamlit as st
from langchain_groq import ChatGroq

from src.config import Config
from src.graph import build_workflow, run_query
from src.ingestion import EmptyDocumentError, UnsupportedFileType, chunk_documents, load_document
from src.utils import get_logger
from src.vectorstore import add_documents, clear_vectorstore, get_retriever, get_vectorstore

logger = get_logger("app")

st.set_page_config(page_title="Document Q&A RAG Agent", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state (FR-10: thread/session identifier)
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

PERSIST_DIR = os.path.join(Config.VECTORSTORE_DIR, st.session_state.thread_id)
UPLOAD_DIR = os.path.join(Config.UPLOAD_DIR, st.session_state.thread_id)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Document Q&A RAG")
    st.caption("LangChain + LangGraph + Groq")

    st.subheader("1. Groq API Key")
    api_key_input = st.text_input(
        "GROQ_API_KEY",
        value=os.getenv("GROQ_API_KEY", Config.GROQ_API_KEY),
        type="password",
        help="Get a free key at console.groq.com",
    )

    model_name = st.selectbox(
        "Model",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
        index=0,
        help="Current production models on Groq. (llama-3.3-70b-versatile and "
        "llama-3.1-8b-instant were deprecated by Groq on 2026-08-16.)",
    )

    st.subheader("2. Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT or DOCX (FR-01)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    with st.expander("Chunking / retrieval settings"):
        chunk_size = st.slider("Chunk size", 300, 2000, Config.CHUNK_SIZE, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, Config.CHUNK_OVERLAP, step=50)
        top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, Config.TOP_K)

    index_clicked = st.button("📥 Index Documents", use_container_width=True)

    if index_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Indexing documents..."):
                vectorstore = get_vectorstore(PERSIST_DIR, Config.EMBEDDING_MODEL)
                total_chunks = 0
                errors = []

                for f in uploaded_files:
                    try:
                        file_path = os.path.join(UPLOAD_DIR, f.name)
                        with open(file_path, "wb") as out:
                            out.write(f.getbuffer())

                        docs = load_document(file_path)
                        for d in docs:
                            d.metadata["source"] = f.name  # clean, user-facing filename

                        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
                        add_documents(vectorstore, chunks)
                        total_chunks += len(chunks)

                        if f.name not in st.session_state.indexed_files:
                            st.session_state.indexed_files.append(f.name)
                        logger.info("Indexed %s: %d chunks", f.name, len(chunks))

                    except (UnsupportedFileType, EmptyDocumentError) as e:
                        errors.append(f"{f.name}: {e}")
                        logger.warning("Skipped %s: %s", f.name, e)
                    except Exception as e:  # noqa: BLE001 - surface any loader/API error to the user
                        errors.append(f"{f.name}: unexpected error ({e})")
                        logger.error("Failed to index %s: %s", f.name, e)

                if total_chunks:
                    st.session_state.vectorstore_ready = True
                    st.success(
                        f"Indexed {total_chunks} chunks from "
                        f"{len(st.session_state.indexed_files)} file(s)."
                    )
                for err in errors:
                    st.error(err)

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for name in st.session_state.indexed_files:
            st.markdown(f"- {name}")

    st.divider()
    if st.button("🗑️ Clear Session / New Chat", use_container_width=True):
        clear_vectorstore(PERSIST_DIR)
        for key in ["thread_id", "messages", "indexed_files", "vectorstore_ready"]:
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Ask questions about your documents")

if not st.session_state.vectorstore_ready:
    st.info("Upload and index at least one document from the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    page_str = f", page {s['page']}" if s.get("page") else ""
                    st.markdown(f"- **{s['source']}**{page_str}")

question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if not api_key_input:
        st.error("Please provide your Groq API key in the sidebar.")
    elif not st.session_state.vectorstore_ready:
        st.error("Please upload and index a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatGroq(api_key=api_key_input, model=model_name, temperature=0)
                    vectorstore = get_vectorstore(PERSIST_DIR, Config.EMBEDDING_MODEL)
                    retriever = get_retriever(vectorstore, top_k)
                    workflow = build_workflow(llm, retriever, max_retries=Config.MAX_RETRIES)
                    result = run_query(workflow, question)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:  # noqa: BLE001 - FR-11 controlled error response (TC-07)
                    logger.error("Error answering question: %s", e)
                    answer = f"Sorry, something went wrong while generating the answer: {e}"
                    sources = []

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            page_str = f", page {s['page']}" if s.get("page") else ""
                            st.markdown(f"- **{s['source']}**{page_str}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
