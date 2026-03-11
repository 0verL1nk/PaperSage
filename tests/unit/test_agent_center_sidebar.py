from ui.agent_center_sidebar import _select_pinned_todo_records


def test_select_pinned_todo_records_prefers_active_non_plan_items() -> None:
    records = [
        {"id": "team_1", "plan_id": "team:run-old", "status": "done"},
        {"id": "team_2", "plan_id": "team:run-old", "status": "done"},
        {"id": "manual_1", "plan_id": "", "status": "todo"},
    ]

    selected, history_only = _select_pinned_todo_records(records)

    assert history_only is False
    assert [item["id"] for item in selected] == ["manual_1"]


def test_select_pinned_todo_records_returns_latest_history_when_all_terminal() -> None:
    records = [
        {"id": "old_1", "plan_id": "team:run-1", "status": "done"},
        {"id": "new_1", "plan_id": "team:run-2", "status": "done"},
        {"id": "new_2", "plan_id": "team:run-2", "status": "canceled"},
    ]

    selected, history_only = _select_pinned_todo_records(records)

    assert history_only is True
    assert [item["id"] for item in selected] == ["new_1", "new_2"]


def test_select_pinned_todo_records_merges_latest_active_plan_with_non_plan_active() -> None:
    records = [
        {"id": "old_done", "plan_id": "team:run-old", "status": "done"},
        {"id": "p2_todo", "plan_id": "team:run-2", "status": "todo"},
        {"id": "manual_active", "plan_id": "", "status": "in_progress"},
        {"id": "p2_done", "plan_id": "team:run-2", "status": "done"},
    ]

    selected, history_only = _select_pinned_todo_records(records)

    assert history_only is False
    assert [item["id"] for item in selected] == ["p2_todo", "manual_active"]
