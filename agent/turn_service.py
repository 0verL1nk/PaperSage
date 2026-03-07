"""
Backward-compatible facade for turn execution.

The canonical implementation now lives in `agent.application.turn_engine`.
"""

from collections.abc import Callable
from typing import Any

from .application.turn_engine import (
    build_search_document_fn as _build_search_document_fn_impl,
)
from .application.turn_engine import (
    execute_turn_core as _execute_turn_core_impl,
)
from .application.turn_engine import (
    normalize_evidence_items as _normalize_evidence_items_impl,
)
from .application.turn_engine import (
    try_parse_mindmap as _try_parse_mindmap_impl,
)
from .domain.trace import phase_label_from_performative, phase_summary
from .orchestration.orchestrator import execute_orchestrated_turn


def _phase_label_from_performative(performative: str) -> str:
    return phase_label_from_performative(performative)


def _phase_summary(phase_labels: list[str]) -> str:
    return phase_summary(phase_labels)


def _normalize_evidence_items(raw_payload: Any) -> list[dict[str, Any]]:
    return _normalize_evidence_items_impl(raw_payload)


def _build_search_document_fn(
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None,
) -> Callable[[str], str]:
    return _build_search_document_fn_impl(search_document_evidence_fn)


def _try_parse_mindmap(answer: str) -> dict[str, Any] | None:
    return _try_parse_mindmap_impl(answer)


def execute_turn_core(
    *,
    prompt: str,
    hinted_prompt: str,
    leader_agent: Any,
    leader_runtime_config: dict[str, Any] | None,
    leader_llm: Any | None = None,
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
    force_plan: bool | None = None,
    force_team: bool | None = None,
    routing_context: str = "",
    on_event: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, Any]:
    return _execute_turn_core_impl(
        prompt=prompt,
        hinted_prompt=hinted_prompt,
        leader_agent=leader_agent,
        leader_runtime_config=leader_runtime_config,
        leader_llm=leader_llm,
        search_document_evidence_fn=search_document_evidence_fn,
        force_plan=force_plan,
        force_team=force_team,
        routing_context=routing_context,
        on_event=on_event,
        orchestrated_turn_executor=execute_orchestrated_turn,
    )

