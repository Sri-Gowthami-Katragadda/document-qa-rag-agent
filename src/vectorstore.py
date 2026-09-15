"""Embedding generation and vector storage using Chroma.

Covers FR-04 (Embeddings) and FR-05 (Vector Store).

Embeddings run locally via sentence-transformers, so no extra API key
is needed beyond the Groq key used for the LLM.
"""
import os
import shutil

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings_cache = {}


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Cache embedding model instances so we don't reload weights every call."""
    if model_name not in _embeddings_cache:
        _embeddings_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings_cache[model_name]


def get_vectorstore(persist_directory: str, embedding_model: str) -> Chroma:
    os.makedirs(persist_directory, exist_ok=True)
    embeddings = get_embeddings(embedding_model)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="documents",
    )


def add_documents(vectorstore: Chroma, chunks) -> None:
    if not chunks:
        return
    vectorstore.add_documents(chunks)


def clear_vectorstore(persist_directory: str) -> None:
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
    os.makedirs(persist_directory, exist_ok=True)


def get_retriever(vectorstore: Chroma, top_k: int = 4):
    """Return a retriever for semantic search (FR-06)."""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})
