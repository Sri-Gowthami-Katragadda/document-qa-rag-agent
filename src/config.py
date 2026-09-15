"""Central configuration for the Document Q&A RAG Agent.

Values are read from environment variables (see .env) with
sensible defaults so the app also runs out-of-the-box.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM (Groq)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Embeddings (local, no API key required)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", 4))

    # Max number of times the "generate" node is allowed to run per question
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 1))

    # Storage
    VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "data/vectorstore")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")

    # Supported document types
    SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".docx")
