from __future__ import annotations

from langgraph.types import Send

from agent.orchestration.langgraph_team_dag import build_ready_task_dispatches


def _todo(
    todo_id: str,
    *,
    status: str = "todo",
    dependencies: list[str] | None = None,
    round_idx: int = 1,
    assignee: str = "researcher",
) -> dict[str, object]:
    return {
        "id": todo_id,
        "status": status,
        "dependencies": list(dependencies or []),
        "round": round_idx,
        "assignee": assignee,
    }


def test_build_ready_task_dispatches_only_includes_unblocked_todo_tasks() -> None:
    todo_records = [
        _todo("t1"),
        _todo("t2", dependencies=["t1"]),
        _todo("t3", dependencies=["t_missing"]),
        _todo("t4", status="done"),
        _todo("t5", status="in_progress"),
    ]

    ready_dispatches = build_ready_task_dispatches(todo_records)

    assert len(ready_dispatches) == 1
    dispatch = ready_dispatches[0]
    assert isinstance(dispatch, Send)
    assert dispatch.node == "dispatch_team_task"
    assert dispatch.arg == {"todo_id": "t1"}


def test_build_ready_task_dispatches_resolves_dependency_when_upstream_is_done() -> None:
    todo_records = [
        _todo("t1", status="done"),
        _todo("t2", dependencies=["t1"]),
    ]

    ready_dispatches = build_ready_task_dispatches(todo_records)

    assert [item.arg for item in ready_dispatches] == [{"todo_id": "t2"}]


def test_build_ready_task_dispatches_keeps_runtime_order_parity() -> None:
    todo_records = [
        _todo("t_writer", round_idx=1, assignee="writer"),
        _todo("t_researcher", round_idx=1, assignee="researcher"),
        _todo("t_round2", round_idx=2, assignee="researcher"),
    ]

    ready_dispatches = build_ready_task_dispatches(
        todo_records,
        role_order={"researcher": 0, "writer": 1},
    )

    assert [item.arg for item in ready_dispatches] == [
        {"todo_id": "t_researcher"},
        {"todo_id": "t_writer"},
        {"todo_id": "t_round2"},
    ]
