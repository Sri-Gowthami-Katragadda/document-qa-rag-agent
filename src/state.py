"""Shared state that flows between LangGraph nodes."""
from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    grounded: bool
    retries: int
    sources: List[dict]
