import json
import os
from pathlib import Path
from typing import Any

import streamlit as st


def build_session_timestamps_caption(selected: dict[str, Any]) -> str:
    return (
        "创建时间："
        f"{str(selected.get('created_at') or '-')}"
        " ｜ 最近更新："
        f"{str(selected.get('updated_at') or '-')}"
    )


def _resolve_todo_path() -> Path:
    root_value = str(os.getenv("AGENT_FILE_TOOLS_ROOT", "") or "").strip()
    file_value = str(os.getenv("AGENT_TODO_FILE", "") or "").strip()
    root = Path(root_value) if root_value else Path.cwd()
    path = Path(file_value) if file_value else Path(".agent/todo.json")
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _load_todo_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _todo_status(record: dict[str, Any]) -> str:
    return str(record.get("status") or "todo").strip().lower()


def _select_pinned_todo_records(
    all_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Pick records for pinned panel.

    Returns:
    - selected records
    - history_only flag (True when no active todo exists)
    """
    records = [item for item in all_records if isinstance(item, dict)]
    if not records:
        return [], False

    terminal_statuses = {"done", "canceled"}
    active_records = [item for item in records if _todo_status(item) not in terminal_statuses]
    if active_records:
        latest_active_plan_id = ""
        for item in reversed(active_records):
            plan_id = str(item.get("plan_id") or "").strip()
            if plan_id:
                latest_active_plan_id = plan_id
                break
        if latest_active_plan_id:
            selected = [
                item
                for item in active_records
                if str(item.get("plan_id") or "").strip() in {"", latest_active_plan_id}
            ]
        else:
            selected = [
                item for item in active_records if not str(item.get("plan_id") or "").strip()
            ]
        return selected, False

    latest_plan_id = ""
    for item in reversed(records):
        plan_id = str(item.get("plan_id") or "").strip()
        if plan_id:
            latest_plan_id = plan_id
            break
    selected = (
        [item for item in records if str(item.get("plan_id") or "").strip() == latest_plan_id]
        if latest_plan_id
        else records
    )
    return selected, True


def render_pinned_todo_panel(
    *,
    project_uid: str,
    expanded: bool = True,
) -> None:
    path = _resolve_todo_path()
    all_records = _load_todo_records(path)
    records, history_only = _select_pinned_todo_records(all_records)

    terminal_statuses = {"done", "canceled"}
    all_terminal = bool(records) and all(_todo_status(r) in terminal_statuses for r in records)
    done_count = sum(1 for r in records if _todo_status(r) in terminal_statuses)

    if not records:
        with st.expander("📌 Todo（Pinned）", expanded=expanded):
            st.caption("暂无任务。可通过 write_todo 工具创建。")
        return

    header = (
        f"📌 Todo（已结束 · {done_count}/{len(records)} 完成）"
        if all_terminal
        else f"📌 Todo（{done_count}/{len(records)} 完成）"
    )

    with st.expander(header, expanded=expanded and not all_terminal):
        if all_terminal and history_only:
            st.caption("✅ 本次 Team 规划所有任务已完成/取消，以下为历史记录。")

        show_done = st.toggle(
            "显示已完成",
            value=all_terminal and history_only,
            key=f"todo_show_done_{project_uid}",
        )
        items = []
        for item in records:
            status = _todo_status(item)
            if not show_done and status in {"done", "canceled"}:
                continue
            items.append(item)
        if not items:
            if history_only and all_terminal:
                st.caption("当前无进行中任务。打开“显示已完成”可查看历史记录。")
            else:
                st.caption("当前筛选条件下无任务。")
            return

        status_order = {
            "in_progress": 0,
            "todo": 1,
            "blocked": 2,
            "done": 3,
            "canceled": 4,
        }
        items.sort(
            key=lambda item: (
                status_order.get(str(item.get("status") or "todo"), 99),
                str(item.get("updated_at") or ""),
            ),
        )
        for item in items[:20]:
            todo_id = str(item.get("id") or "-")
            title = str(item.get("title") or "未命名任务")
            status = str(item.get("status") or "todo")
            priority = str(item.get("priority") or "medium")
            step_ref = str(item.get("step_ref") or "").strip()
            assignee = str(item.get("assignee") or "").strip()
            dependencies = item.get("dependencies")
            dep_count = len(dependencies) if isinstance(dependencies, list) else 0
            badge = {
                "todo": "⬜",
                "in_progress": "🟨",
                "blocked": "🟥",
                "done": "🟩",
                "canceled": "⬛",
            }.get(status, "⬜")
            line = f"{badge} {title} (`{status}`/{priority})"
            if step_ref:
                line += f" · step `{step_ref}`"
            if assignee:
                line += f" · @{assignee}"
            if dep_count > 0:
                line += f" · deps {dep_count}"
            st.markdown(line)
            st.caption(f"id: {todo_id}")


def render_pinned_human_requests_panel(
    *,
    project_uid: str,
    chat_messages: list[dict[str, Any]],
    expanded: bool = False,
) -> None:
    requests: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for message in reversed(chat_messages[-20:]):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        raw = message.get("ask_human_requests")
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            context = str(item.get("context") or "").strip()
            urgency = str(item.get("urgency") or "normal").strip().lower()
            if urgency not in {"low", "normal", "high"}:
                urgency = "normal"
            key = (question, context, urgency)
            if key in seen:
                continue
            seen.add(key)
            requests.append({"question": question, "context": context, "urgency": urgency})
    if not requests:
        return
    with st.expander("🧑 人工确认（Pinned）", expanded=expanded):
        st.caption("以下问题由 ask_human 工具提出，等待你的确认。")
        for idx, item in enumerate(requests[:10], start=1):
            st.markdown(f"**{idx}. {item['question']}**")
            if item["context"]:
                st.caption(f"上下文：{item['context']}")
            st.text_input(
                "你的回复",
                key=f"human_reply_{project_uid}_{idx}",
                placeholder="输入确认/补充信息（当前仅展示，不自动回传）",
            )


def render_pinned_plan_panel(
    *,
    project_uid: str,
    chat_messages: list[dict[str, Any]],
    expanded: bool = True,
) -> None:
    """Render pinned plan panel showing the current execution plan."""
    # First try to get from session_state (real-time)
    agent_plan = st.session_state.get("current_agent_plan")

    # Fallback to chat messages if not in session_state
    if not isinstance(agent_plan, dict) or not agent_plan.get("goal"):
        for message in reversed(chat_messages[-20:]):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            plan_data = message.get("agent_plan")
            if isinstance(plan_data, dict) and plan_data.get("goal"):
                agent_plan = plan_data
                break

    if not isinstance(agent_plan, dict) or not agent_plan.get("goal"):
        return

    goal = str(agent_plan.get("goal", "")).strip()
    description = str(agent_plan.get("description", "")).strip()

    with st.expander("📋 执行计划（Plan）", expanded=expanded):
        st.markdown(f"**目标：** {goal}")
        if description:
            st.markdown("**策略：**")
            st.text(description)
