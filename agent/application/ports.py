from collections.abc import Callable
from typing import Any, Protocol

from ..domain.orchestration import OrchestratedTurn, TraceEvent


class AgentInvoker(Protocol):
    def invoke(
        self,
        payload: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> Any: ...


class EvidenceRetriever(Protocol):
    def __call__(self, query: str) -> dict[str, Any]: ...


class OrchestratedTurnExecutor(Protocol):
    def __call__(
        self,
        *,
        prompt: str,
        hinted_prompt: str,
        leader_agent: AgentInvoker,
        leader_runtime_config: dict[str, Any] | None,
        llm: Any | None = None,
        policy_llm: Any | None = None,
        search_document_fn: Callable[[str], str] | None = None,
        search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
        routing_context: str = "",
        max_team_members: int | None = None,
        max_team_rounds: int | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> OrchestratedTurn: ...
