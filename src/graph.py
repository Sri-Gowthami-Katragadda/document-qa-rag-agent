"""LangGraph orchestration of the RAG pipeline (FR-09).

Graph shape (matches section 5 of the requirements doc):

    START -> retrieve -> grade -> [generate | no_context]
                                        |
                                        v
                                      check --(not grounded, retries left)--> generate
                                        |
                                        v (grounded, or retries exhausted)
                                       END
             no_context ----------------------------------------------------> END
"""
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.state import GraphState

# --------------------------------------------------------------------------
# Structured-output schemas used by the "grading" LLM calls
# --------------------------------------------------------------------------


class DocumentRelevance(BaseModel):
    relevant: bool = Field(
        description="True if the document chunk contains information relevant "
        "to answering the user's question, otherwise False."
    )


class GroundednessGrade(BaseModel):
    grounded: bool = Field(
        description="True if the answer is fully supported by the given context "
        "and does not contain invented information, otherwise False."
    )


GENERATION_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
provided context extracted from the user's uploaded documents.

Rules:
- Base your answer strictly on the CONTEXT below.
- If the CONTEXT does not contain the answer, respond exactly with: \
"I could not find this information in the provided documents."
- Be concise and clear.
- Do not mention chunks, context, or retrieval - answer naturally, as if you \
had simply read the documents."""


# --------------------------------------------------------------------------
# Node factories
# --------------------------------------------------------------------------


def retrieve_node(retriever) -> Callable[[GraphState], dict]:
    """Node: Retrieve Relevant Documents."""

    def _node(state: GraphState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "retries": state.get("retries", 0)}

    return _node


def grade_documents_node(llm) -> Callable[[GraphState], dict]:
    """Node: Grade / Validate Retrieved Context.

    Filters out retrieved chunks that the LLM judges irrelevant to the
    question, so hallucination-prone noise never reaches generation.
    """
    grader = llm.with_structured_output(DocumentRelevance)

    def _node(state: GraphState) -> dict:
        question = state["question"]
        filtered = []
        for doc in state["documents"]:
            try:
                result = grader.invoke(
                    f"User question: {question}\n\n"
                    f"Document chunk:\n{doc.page_content}\n\n"
                    "Is this document chunk relevant to answering the question?"
                )
                if result.relevant:
                    filtered.append(doc)
            except Exception:
                # If grading fails for some reason, err on the side of keeping
                # the chunk rather than silently dropping useful context.
                filtered.append(doc)
        return {"documents": filtered}

    return _node


def generate_node(llm, max_retries: int) -> Callable[[GraphState], dict]:
    """Node: Generate Answer."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATION_SYSTEM_PROMPT),
            ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
        ]
    )
    chain = prompt | llm

    def _node(state: GraphState) -> dict:
        docs = state["documents"]
        context = (
            "\n\n---\n\n".join(d.page_content for d in docs)
            if docs
            else "No context available."
        )
        response = chain.invoke({"context": context, "question": state["question"]})

        sources = []
        seen = set()
        for d in docs:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            key = (src, page)
            if key not in seen:
                seen.add(key)
                sources.append(
                    {"source": src, "page": (page + 1) if isinstance(page, int) else None}
                )

        return {
            "generation": response.content,
            "sources": sources,
            "retries": state.get("retries", 0) + 1,
        }

    return _node


def check_answer_node(llm, max_retries: int) -> Callable[[GraphState], dict]:
    """Node: Check Answer / Grounding.

    Verifies the generated answer is actually supported by the retrieved
    context. If not, and a retry is still available, routes back to
    "generate"; otherwise falls back to a safe "not found" style answer
    instead of letting a hallucination through (TC-08).
    """
    grader = llm.with_structured_output(GroundednessGrade)

    def _node(state: GraphState) -> dict:
        docs = state["documents"]
        if not docs:
            return {"grounded": True}

        context = "\n\n---\n\n".join(d.page_content for d in docs)
        try:
            result = grader.invoke(
                f"CONTEXT:\n{context}\n\nANSWER:\n{state['generation']}\n\n"
                "Is the ANSWER fully supported by the CONTEXT?"
            )
            grounded = result.grounded
        except Exception:
            grounded = True

        if not grounded and state.get("retries", 0) >= max_retries:
            # Out of retries: don't risk surfacing a hallucinated answer.
            return {
                "grounded": True,
                "generation": (
                    "I could not verify a reliable answer to this question "
                    "using the provided documents."
                ),
                "sources": [],
            }

        return {"grounded": grounded}

    return _node


def no_context_node(state: GraphState) -> dict:
    """Node: reached when grading filters out every retrieved chunk."""
    return {
        "generation": "I could not find relevant information in the provided documents to answer this question.",
        "sources": [],
    }


# --------------------------------------------------------------------------
# Conditional routing
# --------------------------------------------------------------------------


def _decide_after_grading(state: GraphState) -> str:
    return "generate" if state["documents"] else "no_context"


def _decide_after_check(state: GraphState) -> str:
    return "end" if state["grounded"] else "retry"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def build_workflow(llm, retriever, max_retries: int = 1):
    """Assemble and compile the LangGraph StateGraph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node(retriever))
    workflow.add_node("grade", grade_documents_node(llm))
    workflow.add_node("generate", generate_node(llm, max_retries))
    workflow.add_node("check", check_answer_node(llm, max_retries))
    workflow.add_node("no_context", no_context_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade", _decide_after_grading, {"generate": "generate", "no_context": "no_context"}
    )
    workflow.add_edge("generate", "check")
    workflow.add_conditional_edges(
        "check", _decide_after_check, {"retry": "generate", "end": END}
    )
    workflow.add_edge("no_context", END)

    return workflow.compile()


def run_query(app, question: str) -> dict:
    """Run one question through the compiled graph and return answer + sources."""
    initial_state: GraphState = {
        "question": question,
        "documents": [],
        "generation": "",
        "grounded": False,
        "retries": 0,
        "sources": [],
    }
    result = app.invoke(initial_state)
    return {
        "answer": result.get("generation", ""),
        "sources": result.get("sources", []),
    }
