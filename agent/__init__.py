__version__ = "1.0.5"

from .archive import list_agent_outputs, save_agent_output
from .llm_provider import build_openai_compatible_chat_model
from .paper_agent import create_paper_agent_session
from .rag.hybrid import (
    HybridRetriever,
    build_hybrid_retriever,
    build_local_evidence_retriever_with_settings,
    build_local_vector_retriever_with_settings,
)
from .rag.local import build_local_vector_retriever
from .stream import iter_agent_response_deltas
from .tools.builder import build_agent_tools

__all__ = [
    "build_agent_tools",
    "build_openai_compatible_chat_model",
    "build_local_vector_retriever",
    "build_hybrid_retriever",
    "build_local_evidence_retriever_with_settings",
    "build_local_vector_retriever_with_settings",
    "HybridRetriever",
    "create_paper_agent_session",
    "iter_agent_response_deltas",
    "list_agent_outputs",
    "save_agent_output",
]
