"""Embedding generation and vector storage using Chroma.

Covers FR-04 (Embeddings) and FR-05 (Vector Store).

Embeddings run locally via sentence-transformers, so no extra API key
is needed beyond the Groq key used for the LLM.
"""
import os
import re
import shutil

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings_cache = {}
_bm25_cache = {}  # persist_directory -> BM25Retriever, rebuilt only when stale


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


def add_documents(vectorstore: Chroma, chunks, persist_directory: str = None) -> None:
    if not chunks:
        return
    vectorstore.add_documents(chunks)
    if persist_directory:
        _bm25_cache.pop(persist_directory, None)  # new chunks -> stale BM25 index


def clear_vectorstore(persist_directory: str) -> None:
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
    os.makedirs(persist_directory, exist_ok=True)
    _bm25_cache.pop(persist_directory, None)  # wiped store -> stale BM25 index


def get_retriever(vectorstore: Chroma, top_k: int = 4):
    """Return a retriever for plain semantic search (FR-06)."""
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def _load_all_chunks(vectorstore: Chroma):
    """BM25 is in-memory, so rebuild it from what's persisted in Chroma."""
    data = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


def _doc_key(doc: Document):
    # chunk_index restarts per file, so key on source + page + content instead
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)


def get_hybrid_retriever(
    vectorstore: Chroma,
    persist_directory: str,
    top_k: int = 4,
    bm25_weight: float = 0.5,
    rrf_k: int = 60,
    candidate_k: int = None,
):
    """BM25 + vector search fused with weighted Reciprocal Rank Fusion.

    The BM25 index is cached per persist_directory (see _bm25_cache) so it's
    only rebuilt when something actually changed - not on every question.
    add_documents() and clear_vectorstore() both invalidate the cache entry
    for the directory they touched.
    """
    candidate_k = candidate_k or max(top_k * 3, 10)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": candidate_k})

    bm25 = _bm25_cache.get(persist_directory)
    if bm25 is None:
        chunks = _load_all_chunks(vectorstore)
        if not chunks:
            return get_retriever(vectorstore, top_k)  # nothing indexed yet
        bm25 = BM25Retriever.from_documents(chunks, k=candidate_k, preprocess_func=_tokenize)
        _bm25_cache[persist_directory] = bm25
    else:
        bm25.k = candidate_k  # keep in sync if candidate_k differs from the cached build

    def _retrieve(query: str):
        scores, docs_by_key = {}, {}
        for weight, results in (
            (1 - bm25_weight, vector_retriever.invoke(query)),
            (bm25_weight, bm25.invoke(query)),
        ):
            for rank, doc in enumerate(results):
                key = _doc_key(doc)
                docs_by_key[key] = doc
                scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank + 1)

        ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [docs_by_key[k] for k in ranked]

    return RunnableLambda(_retrieve)