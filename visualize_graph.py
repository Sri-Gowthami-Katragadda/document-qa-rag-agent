"""Print an ASCII diagram of the compiled LangGraph workflow.

Doesn't call a real LLM or retriever - a fake stand-in LLM satisfies the
`prompt | llm` chaining and `.with_structured_output()` calls made while
*building* the graph (see rewrite_question_node, grade_documents_node,
etc. in graph.py), without ever actually running an inference call.

Run with: python visualize_graph.py
"""
from langchain_core.runnables import RunnableLambda

from src.graph import build_workflow


class _FakeLLM(RunnableLambda):
    """Stands in for a real ChatGroq instance just long enough for
    build_workflow() to wire up its prompt chains - never invoked."""

    def __init__(self):
        super().__init__(lambda x: None)

    def with_structured_output(self, schema):
        return self


workflow = build_workflow(llm=_FakeLLM(), retriever=None, max_retries=1)

print(workflow.get_graph().draw_ascii())