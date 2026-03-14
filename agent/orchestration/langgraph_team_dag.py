from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.types import Send

from .role_dispatcher import RoleDispatchPlan

TODO_STATUS = "todo"
DONE_STATUS = "done"
DEFAULT_ROLE_ORDER = 999


def _normalize_status(record: Mapping[str, Any]) -> str:
    return str(record.get("status") or "").strip().lower()


def _build_records_by_id(todo_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in todo_records:
        todo_id = str(record.get("id") or "").strip()
        if not todo_id:
            continue
        records[todo_id] = record
    return records


def _dependencies_satisfied(
    record: Mapping[str, Any],
    *,
    records_by_id: Mapping[str, dict[str, Any]],
) -> bool:
    dependencies = record.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        return True
    for dependency in dependencies:
        dep_id = str(dependency or "").strip()
        if not dep_id:
            continue
        dep_record = records_by_id.get(dep_id)
        if not isinstance(dep_record, dict):
            return False
        if _normalize_status(dep_record) != DONE_STATUS:
            return False
    return True


def _ready_record_sort_key(
    record: Mapping[str, Any],
    *,
    role_order: Mapping[str, int],
) -> tuple[int, int, str]:
    round_idx = int(record.get("round") or 0)
    role_name = str(record.get("assignee") or "").strip()
    todo_id = str(record.get("id") or "")
    return (
        round_idx,
        role_order.get(role_name, DEFAULT_ROLE_ORDER),
        todo_id,
    )


def build_ready_task_dispatches(
    todo_records: list[dict[str, Any]],
    *,
    role_order: Mapping[str, int] | None = None,
    target_node: str = "dispatch_team_task",
) -> list[Send]:
    records_by_id = _build_records_by_id(todo_records)
    normalized_role_order = role_order or {}
    ready_records = [
        record
        for record in todo_records
        if _normalize_status(record) == TODO_STATUS
        and _dependencies_satisfied(record, records_by_id=records_by_id)
    ]
    ready_records.sort(
        key=lambda item: _ready_record_sort_key(item, role_order=normalized_role_order)
    )
    return [
        Send(target_node, {"todo_id": str(record.get("id") or "")})
        for record in ready_records
        if str(record.get("id") or "").strip()
    ]


def _build_records_by_todo_id(todo_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("id") or "").strip(): record
        for record in todo_records
        if str(record.get("id") or "").strip()
    }


def build_ready_role_dispatches(
    todo_records: list[dict[str, Any]],
    *,
    role_plan: RoleDispatchPlan,
    target_node: str = "dispatch_team_task",
) -> list[Send]:
    ready_dispatches = build_ready_task_dispatches(
        todo_records,
        role_order=role_plan.role_order,
        target_node=target_node,
    )
    records_by_todo_id = _build_records_by_todo_id(todo_records)
    role_dispatches: list[Send] = []
    for dispatch in ready_dispatches:
        dispatch_arg = dispatch.arg if isinstance(dispatch.arg, dict) else {}
        todo_id = str(dispatch_arg.get("todo_id") or "").strip()
        if not todo_id:
            continue
        record = records_by_todo_id.get(todo_id)
        assignee = str((record or {}).get("assignee") or "").strip()
        role = role_plan.role_map.get(assignee)
        role_dispatches.append(
            Send(
                dispatch.node,
                {
                    "todo_id": todo_id,
                    "assignee": assignee,
                    "role_goal": role.goal if role else "",
                },
            )
        )
    return role_dispatches
