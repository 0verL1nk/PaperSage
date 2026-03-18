from typing import Any


def create_session_metrics() -> dict[str, Any]:
    return {
        "total_queries": 0,
        "plan_enabled_count": 0,
        "team_enabled_count": 0,
        "team_rounds_total": 0,
        "team_rounds_max": 0,
        "team_fallback_count": 0,
        "total_latency_ms": 0.0,
        "max_latency_ms": 0.0,
        "replan_rounds_total": 0,
        "replan_rounds_max": 0,
        "step_total": 0,
        "step_retry_total": 0,
        "step_verify_fail_total": 0,
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


def _extract_team_rounds(
    team_execution: dict[str, Any] | None,
    trace_payload: list[dict[str, str]] | None,
) -> int:
    if isinstance(team_execution, dict):
        rounds = team_execution.get("rounds")
        if isinstance(rounds, int):
            return max(0, rounds)
    return extract_replan_rounds(trace_payload)


def _extract_step_metrics(trace_payload: list[dict[str, str]] | None) -> tuple[int, int, int]:
    if not isinstance(trace_payload, list):
        return 0, 0, 0
    step_total = 0
    step_retry_total = 0
    step_verify_fail_total = 0
    for item in trace_payload:
        if not isinstance(item, dict):
            continue
        performative = str(item.get("performative") or "").strip()
        if performative == "step_complete":
            step_total += 1
        elif performative == "step_retry":
            step_retry_total += 1
        elif performative == "step_verify":
            meta = item.get("meta")
            if isinstance(meta, dict) and str(meta.get("verification_status") or "").strip() == "failed":
                step_verify_fail_total += 1
    return step_total, step_retry_total, step_verify_fail_total


def record_query_metrics(
    metrics: dict[str, Any],
    *,
    latency_ms: float,
    trace_payload: list[dict[str, str]] | None = None,
    workflow_mode: str | None = None,
    policy_decision: dict[str, Any] | None = None,
    team_execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        metrics = create_session_metrics()

    if "workflow_counts" not in metrics or not isinstance(metrics["workflow_counts"], dict):
        metrics["workflow_counts"] = {}
    for workflow in VALID_WORKFLOWS:
        metrics["workflow_counts"].setdefault(workflow, 0)

    metrics["total_queries"] = int(metrics.get("total_queries", 0)) + 1

    if isinstance(workflow_mode, str) and workflow_mode in VALID_WORKFLOWS:
        metrics["workflow_counts"][workflow_mode] += 1

    if isinstance(policy_decision, dict):
        if bool(policy_decision.get("plan_enabled", False)):
            metrics["plan_enabled_count"] = int(metrics.get("plan_enabled_count", 0)) + 1
        if bool(policy_decision.get("team_enabled", False)):
            metrics["team_enabled_count"] = int(metrics.get("team_enabled_count", 0)) + 1

    if isinstance(team_execution, dict) and team_execution.get("fallback_reason"):
        metrics["team_fallback_count"] = int(metrics.get("team_fallback_count", 0)) + 1

    safe_latency = max(0.0, float(latency_ms))
    metrics["total_latency_ms"] = float(metrics.get("total_latency_ms", 0.0)) + safe_latency
    metrics["max_latency_ms"] = max(float(metrics.get("max_latency_ms", 0.0)), safe_latency)

    trace_events = len(trace_payload) if isinstance(trace_payload, list) else 0
    metrics["trace_events_total"] = int(metrics.get("trace_events_total", 0)) + trace_events

    # Backward-compatible replan metrics
    replan_rounds = extract_replan_rounds(trace_payload)
    metrics["replan_rounds_total"] = int(metrics.get("replan_rounds_total", 0)) + replan_rounds
    metrics["replan_rounds_max"] = max(
        int(metrics.get("replan_rounds_max", 0)),
        replan_rounds,
    )
    step_total, step_retry_total, step_verify_fail_total = _extract_step_metrics(trace_payload)
    metrics["step_total"] = int(metrics.get("step_total", 0)) + step_total
    metrics["step_retry_total"] = int(metrics.get("step_retry_total", 0)) + step_retry_total
    metrics["step_verify_fail_total"] = int(
        metrics.get("step_verify_fail_total", 0)
    ) + step_verify_fail_total

    team_rounds = _extract_team_rounds(team_execution, trace_payload)
    metrics["team_rounds_total"] = int(metrics.get("team_rounds_total", 0)) + team_rounds
    metrics["team_rounds_max"] = max(int(metrics.get("team_rounds_max", 0)), team_rounds)
    return metrics


def summarize_session_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        metrics = create_session_metrics()

    total_queries = int(metrics.get("total_queries", 0))

    average_latency_ms = (
        float(metrics.get("total_latency_ms", 0.0)) / total_queries if total_queries > 0 else 0.0
    )
    average_replan_rounds = (
        float(metrics.get("replan_rounds_total", 0)) / total_queries if total_queries > 0 else 0.0
    )
    average_team_rounds = (
        float(metrics.get("team_rounds_total", 0)) / total_queries if total_queries > 0 else 0.0
    )

    plan_enabled_count = int(metrics.get("plan_enabled_count", 0))
    team_enabled_count = int(metrics.get("team_enabled_count", 0))
    plan_enabled_ratio = (plan_enabled_count / total_queries) if total_queries > 0 else 0.0
    team_enabled_ratio = (team_enabled_count / total_queries) if total_queries > 0 else 0.0

    return {
        "total_queries": total_queries,
        "plan_enabled_count": plan_enabled_count,
        "plan_enabled_ratio": plan_enabled_ratio,
        "team_enabled_count": team_enabled_count,
        "team_enabled_ratio": team_enabled_ratio,
        "team_rounds_total": int(metrics.get("team_rounds_total", 0)),
        "team_rounds_max": int(metrics.get("team_rounds_max", 0)),
        "average_team_rounds": average_team_rounds,
        "team_fallback_count": int(metrics.get("team_fallback_count", 0)),
        "average_latency_ms": average_latency_ms,
        "max_latency_ms": float(metrics.get("max_latency_ms", 0.0)),
        "trace_events_total": int(metrics.get("trace_events_total", 0)),
        "replan_rounds_total": int(metrics.get("replan_rounds_total", 0)),
        "replan_rounds_max": int(metrics.get("replan_rounds_max", 0)),
        "average_replan_rounds": average_replan_rounds,
        "step_total": int(metrics.get("step_total", 0)),
        "step_retry_total": int(metrics.get("step_retry_total", 0)),
        "step_verify_fail_total": int(metrics.get("step_verify_fail_total", 0)),
    }
