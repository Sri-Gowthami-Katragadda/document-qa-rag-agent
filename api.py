"""Basic FastAPI backend for the Document Q&A RAG Agent.

Wraps the existing src/ modules (ingestion, vectorstore, graph) behind a
minimal HTTP API. Uploads one file per request - this renders as a single,
unambiguous file-picker button in Swagger UI (no array-of-files quirks).

SECURITY: the Groq API key is read only from the server's own .env file
(via src/config.py) and is never accepted from a request - no endpoint
below has an api_key field.

Run with:   uvicorn api:app --reload
Docs at:    http://localhost:8000/docs
"""
import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from langchain_groq import ChatGroq
from pydantic import BaseModel

from src.config import Config
from src.graph import build_workflow, run_query
from src.ingestion import chunk_documents, load_document
from src.vectorstore import add_documents, clear_vectorstore, get_hybrid_retriever, get_vectorstore

app = FastAPI(title="Document Q&A RAG Agent API")


class ChatRequest(BaseModel):
    question: str
    thread_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.get("/health")
def health():
    return {"status": "ok", "groq_configured": bool(Config.GROQ_API_KEY)}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None),
):
    thread_id = thread_id or str(uuid.uuid4())
    upload_dir = os.path.join(Config.UPLOAD_DIR, thread_id)
    os.makedirs(upload_dir, exist_ok=True)

    persist_dir = os.path.join(Config.VECTORSTORE_DIR, thread_id)
    vectorstore = get_vectorstore(persist_dir, Config.EMBEDDING_MODEL)

    file_path = os.path.join(upload_dir, file.filename)
    content = await file.read()
    with open(file_path, "wb") as out:
        out.write(content)

    try:
        docs = load_document(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{file.filename}: {e}")

    for d in docs:
        d.metadata["source"] = file.filename

    chunks = chunk_documents(docs)
    # add_documents(vectorstore, chunks)
    add_documents(vectorstore, chunks, persist_directory=persist_dir)

    return {
        "thread_id": thread_id,
        "indexed_file": file.filename,
        "chunks_indexed": len(chunks),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not Config.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")

    persist_dir = os.path.join(Config.VECTORSTORE_DIR, request.thread_id)
    vectorstore = get_vectorstore(persist_dir, Config.EMBEDDING_MODEL)
    # retriever = get_retriever(vectorstore, Config.TOP_K)
    retriever = get_hybrid_retriever(vectorstore, persist_dir, Config.TOP_K)

    llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL, temperature=0)
    workflow = build_workflow(llm, retriever, max_retries=Config.MAX_RETRIES)

    result = run_query(workflow, request.question)

    return ChatResponse(answer=result["answer"], sources=result["sources"])