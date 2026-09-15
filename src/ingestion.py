"""Document ingestion: parsing, text extraction and chunking.
"""
import os
from typing import List

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config


class UnsupportedFileType(Exception):
    """Raised when a file extension is not supported (TC-02)."""


class EmptyDocumentError(Exception):
    """Raised when a document contains no extractable text."""


def load_document(file_path: str) -> List[Document]:
    """Parse a single file (PDF / TXT / DOCX) into LangChain Documents."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise UnsupportedFileType(
            f"'{ext}' is not supported. Allowed types: {', '.join(Config.SUPPORTED_EXTENSIONS)}"
        )

    docs = loader.load()
    docs = [d for d in docs if d.page_content and d.page_content.strip()]

    if not docs:
        raise EmptyDocumentError(
            f"No readable text could be extracted from '{os.path.basename(file_path)}'."
        )

    return docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """Split documents into overlapping chunks (FR-03)."""
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
