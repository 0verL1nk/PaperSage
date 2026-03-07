from agent.a2a.coordinator import WORKFLOW_PLAN_ACT_REPLAN, WORKFLOW_REACT
from agent.metrics import (
    create_session_metrics,
    extract_replan_rounds,
    record_query_metrics,
    summarize_session_metrics,
)


def test_extract_replan_rounds_counts_only_replan_events():
    trace = [
        {"performative": "request"},
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
        workflow_mode=WORKFLOW_PLAN_ACT_REPLAN,
        latency_ms=300.0,
        trace_payload=[
            {"performative": "request"},
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
