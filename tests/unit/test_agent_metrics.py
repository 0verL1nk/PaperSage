from agent.a2a.coordinator import WORKFLOW_PLAN_ACT, WORKFLOW_REACT, WORKFLOW_TEAM
from agent.metrics import (
    create_session_metrics,
    extract_replan_rounds,
    record_query_metrics,
    summarize_session_metrics,
)


def test_extract_replan_rounds_counts_only_replan_events():
    trace = [
        {"performative": "request"},
        {"performative": "policy_switch"},
        {"performative": "plan"},
        {"performative": "replan"},
        {"performative": "replan"},
        {"performative": "final"},
    ]
    assert extract_replan_rounds(trace) == 2


def test_record_query_metrics_updates_counters():
    metrics = create_session_metrics()
    updated = record_query_metrics(
        metrics,
        workflow_mode=WORKFLOW_REACT,
        latency_ms=120.5,
        trace_payload=[{"performative": "request"}, {"performative": "final"}],
    )

    assert updated["total_queries"] == 1
    assert updated["workflow_counts"][WORKFLOW_REACT] == 1
    assert updated["total_latency_ms"] == 120.5
    assert updated["max_latency_ms"] == 120.5
    assert updated["trace_events_total"] == 2


def test_record_query_metrics_accumulates_replan_rounds():
    metrics = create_session_metrics()
    record_query_metrics(
        metrics,
        workflow_mode=WORKFLOW_PLAN_ACT,
        latency_ms=300.0,
        trace_payload=[
            {"performative": "request"},
            {"performative": "policy_switch"},
            {
                "performative": "step_verify",
                "meta": {"verification_status": "failed"},
            },
            {"performative": "step_retry"},
            {"performative": "step_complete"},
            {"performative": "replan"},
            {"performative": "replan"},
            {"performative": "final"},
        ],
    )
    summary = summarize_session_metrics(metrics)

    assert summary["total_queries"] == 1
    assert summary["replan_rounds_total"] == 2
    assert summary["replan_rounds_max"] == 2
    assert summary["average_replan_rounds"] == 2.0
    assert summary["step_total"] == 1
    assert summary["step_retry_total"] == 1
    assert summary["step_verify_fail_total"] == 1


def test_record_query_metrics_counts_team_mode() -> None:
    metrics = create_session_metrics()
    updated = record_query_metrics(
        metrics,
        workflow_mode=WORKFLOW_TEAM,
        latency_ms=80.0,
        trace_payload=[{"performative": "request"}, {"performative": "final"}],
    )

    assert updated["workflow_counts"][WORKFLOW_TEAM] == 1
    summary = summarize_session_metrics(updated)
    assert summary["workflow_counts"][WORKFLOW_TEAM] == 1
