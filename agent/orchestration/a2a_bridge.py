from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agent.a2a import (
    WORKFLOW_PLAN_ACT,
    A2AMessage,
    A2AMultiAgentSession,
    create_multi_agent_a2a_session,
)
from agent.a2a.coordinator import canonicalize_workflow_mode

DEFAULT_WORKFLOW_MODE = WORKFLOW_PLAN_ACT
DEFAULT_MAX_REPLAN_ROUNDS = 2


@dataclass(frozen=True)
class A2AInvocation:
    question: str
    workflow_mode: str
    max_replan_rounds: int


class A2ABridge:
    def __init__(
        self,
        *,
        session_factory: Callable[..., A2AMultiAgentSession] = create_multi_agent_a2a_session,
    ) -> None:
        self._session_factory = session_factory

    def create_session(
        self,
        *,
        llm: Any,
        search_document_fn: Callable[[str], str],
        search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
        context_hint: str = "",
    ) -> A2AMultiAgentSession:
        return self._session_factory(
            llm=llm,
            search_document_fn=search_document_fn,
            search_document_evidence_fn=search_document_evidence_fn,
            context_hint=context_hint,
        )

    @staticmethod
    def orchestration_to_a2a_input(orchestration_input: Mapping[str, Any]) -> A2AInvocation:
        question = _extract_question(orchestration_input)
        workflow_mode = canonicalize_workflow_mode(
            str(orchestration_input.get("workflow_mode") or DEFAULT_WORKFLOW_MODE).strip()
            or DEFAULT_WORKFLOW_MODE
        )
        max_replan_rounds = _normalize_max_replan_rounds(
            orchestration_input.get("max_replan_rounds")
        )
        return A2AInvocation(
            question=question,
            workflow_mode=workflow_mode,
            max_replan_rounds=max_replan_rounds,
        )

    @staticmethod
    def a2a_to_orchestration_output(
        *,
        answer: str,
        trace: list[A2AMessage],
        session_id: str,
        workflow_mode: str,
    ) -> dict[str, Any]:
        return {
            "status": "completed",
            "final_answer": answer,
            "trace": trace,
            "session_id": session_id,
            "workflow_mode": workflow_mode,
        }

    def run_with_session(
        self,
        session: A2AMultiAgentSession,
        orchestration_input: Mapping[str, Any],
        *,
        on_event: Callable[[A2AMessage], None] | None = None,
    ) -> dict[str, Any]:
        invocation = self.orchestration_to_a2a_input(orchestration_input)
        answer, trace = session.coordinator.run(
            invocation.question,
            workflow_mode=invocation.workflow_mode,
            on_event=on_event,
            max_replan_rounds=invocation.max_replan_rounds,
        )
        return self.a2a_to_orchestration_output(
            answer=answer,
            trace=trace,
            session_id=session.session_id,
            workflow_mode=invocation.workflow_mode,
        )


def _extract_question(orchestration_input: Mapping[str, Any]) -> str:
    candidates = (
        orchestration_input.get("question"),
        orchestration_input.get("user_prompt"),
        orchestration_input.get("prompt"),
    )
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _normalize_max_replan_rounds(raw_value: Any) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_REPLAN_ROUNDS
    return max(0, value)
