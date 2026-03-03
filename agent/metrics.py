from typing import Any

from .multi_agent_a2a import WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN, WORKFLOW_REACT


VALID_WORKFLOWS = {WORKFLOW_REACT, WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN}


def create_session_metrics() -> dict[str, Any]:
    return {
        "total_queries": 0,
        "workflow_counts": {
            WORKFLOW_REACT: 0,
            WORKFLOW_PLAN_ACT: 0,
            WORKFLOW_PLAN_ACT_REPLAN: 0,
        },
        "total_latency_ms": 0.0,
        "max_latency_ms": 0.0,
        "replan_rounds_total": 0,
        "replan_rounds_max": 0,
        "trace_events_total": 0,
    }


def extract_replan_rounds(trace_payload: list[dict[str, str]] | None) -> int:
    if not isinstance(trace_payload, list):
        return 0
    rounds = 0
    for item in trace_payload:
        if not isinstance(item, dict):
            continue
        if item.get("performative") == "replan":
            rounds += 1
    return rounds


def record_query_metrics(
    metrics: dict[str, Any],
    *,
    workflow_mode: str,
    latency_ms: float,
    trace_payload: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        metrics = create_session_metrics()

    if "workflow_counts" not in metrics or not isinstance(metrics["workflow_counts"], dict):
        metrics["workflow_counts"] = {}
    for workflow in VALID_WORKFLOWS:
        metrics["workflow_counts"].setdefault(workflow, 0)

    metrics["total_queries"] = int(metrics.get("total_queries", 0)) + 1
    if workflow_mode in VALID_WORKFLOWS:
        metrics["workflow_counts"][workflow_mode] += 1

    safe_latency = max(0.0, float(latency_ms))
    metrics["total_latency_ms"] = float(metrics.get("total_latency_ms", 0.0)) + safe_latency
    metrics["max_latency_ms"] = max(float(metrics.get("max_latency_ms", 0.0)), safe_latency)

    trace_events = len(trace_payload) if isinstance(trace_payload, list) else 0
    metrics["trace_events_total"] = int(metrics.get("trace_events_total", 0)) + trace_events

    replan_rounds = extract_replan_rounds(trace_payload)
    metrics["replan_rounds_total"] = int(metrics.get("replan_rounds_total", 0)) + replan_rounds
    metrics["replan_rounds_max"] = max(
        int(metrics.get("replan_rounds_max", 0)),
        replan_rounds,
    )
    return metrics


def summarize_session_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        metrics = create_session_metrics()

    total_queries = int(metrics.get("total_queries", 0))
    workflow_counts = metrics.get("workflow_counts", {})
    if not isinstance(workflow_counts, dict):
        workflow_counts = {}
    average_latency_ms = (
        float(metrics.get("total_latency_ms", 0.0)) / total_queries if total_queries > 0 else 0.0
    )
    average_replan_rounds = (
        float(metrics.get("replan_rounds_total", 0)) / total_queries if total_queries > 0 else 0.0
    )

    workflow_ratios: dict[str, float] = {}
    for workflow in VALID_WORKFLOWS:
        count = int(workflow_counts.get(workflow, 0))
        workflow_ratios[workflow] = (count / total_queries) if total_queries > 0 else 0.0

    return {
        "total_queries": total_queries,
        "workflow_counts": {
            WORKFLOW_REACT: int(workflow_counts.get(WORKFLOW_REACT, 0)),
            WORKFLOW_PLAN_ACT: int(workflow_counts.get(WORKFLOW_PLAN_ACT, 0)),
            WORKFLOW_PLAN_ACT_REPLAN: int(workflow_counts.get(WORKFLOW_PLAN_ACT_REPLAN, 0)),
        },
        "workflow_ratios": workflow_ratios,
        "average_latency_ms": average_latency_ms,
        "max_latency_ms": float(metrics.get("max_latency_ms", 0.0)),
        "trace_events_total": int(metrics.get("trace_events_total", 0)),
        "replan_rounds_total": int(metrics.get("replan_rounds_total", 0)),
        "replan_rounds_max": int(metrics.get("replan_rounds_max", 0)),
        "average_replan_rounds": average_replan_rounds,
    }

