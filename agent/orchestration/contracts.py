"""
Backward-compatible export layer.

The canonical orchestration contracts live under `agent.domain`.
"""

from ..domain.orchestration import (
    OrchestratedTurn,
    PolicyDecision,
    TeamExecution,
    TeamRole,
    TraceContext,
    TraceEvent,
    TracePayload,
    build_trace_event,
    create_trace_context,
    ensure_valid_trace_route,
    is_valid_trace_route,
)

__all__ = [
    "OrchestratedTurn",
    "PolicyDecision",
    "TeamExecution",
    "TeamRole",
    "TraceContext",
    "TraceEvent",
    "TracePayload",
    "build_trace_event",
    "create_trace_context",
    "ensure_valid_trace_route",
    "is_valid_trace_route",
]
