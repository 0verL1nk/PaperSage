__version__ = "1.0.5"

from .a2a import (
    coordinator as multi_agent_a2a,
)
from .a2a import (
    replan_policy as a2a_replan_policy,
)
from .a2a import (
    router as workflow_router,
)
from .a2a import (
    standard as a2a_standard,
)
from .a2a import (
    state_machine as a2a_state_machine,
)
from .a2a.coordinator import (
    WORKFLOW_PLAN_ACT,
    WORKFLOW_PLAN_ACT_REPLAN,
    WORKFLOW_REACT,
    create_multi_agent_a2a_session,
)
from .a2a.router import WORKFLOW_LABELS, auto_select_workflow_mode
from .a2a.standard import (
    A2AInMemoryServer,
    build_agent_card,
    build_coordinator_executor,
)
from .archive import list_agent_outputs, save_agent_output
from .capabilities import build_agent_tools
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

__all__ = [
    "build_agent_tools",
    "build_agent_card",
    "build_coordinator_executor",
    "A2AInMemoryServer",
    "build_openai_compatible_chat_model",
    "build_local_vector_retriever",
    "build_hybrid_retriever",
    "build_local_evidence_retriever_with_settings",
    "build_local_vector_retriever_with_settings",
    "HybridRetriever",
    "create_multi_agent_a2a_session",
    "create_paper_agent_session",
    "iter_agent_response_deltas",
    "list_agent_outputs",
    "save_agent_output",
    "WORKFLOW_REACT",
    "WORKFLOW_PLAN_ACT",
    "WORKFLOW_PLAN_ACT_REPLAN",
    "WORKFLOW_LABELS",
    "auto_select_workflow_mode",
    "multi_agent_a2a",
    "workflow_router",
    "a2a_standard",
    "a2a_state_machine",
    "a2a_replan_policy",
]
