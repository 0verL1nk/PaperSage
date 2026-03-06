import json
import os
from pathlib import Path
from typing import Any

import streamlit as st


def resolve_strategy_flags(
    *,
    strategy_mode: str,
    force_plan_value: bool,
    force_team_value: bool,
) -> tuple[bool | None, bool | None]:
    if strategy_mode == "手动":
        return bool(force_plan_value), bool(force_team_value)
    return None, None


def render_strategy_controls(
    *,
    project_uid: str,
    session_uid: str,
    turn_in_progress: bool,
) -> tuple[bool | None, bool | None]:
    with st.expander("执行策略", expanded=False):
        strategy_mode = st.radio(
            "策略模式",
            options=["自动", "手动"],
            key=f"agent_strategy_mode_{project_uid}_{session_uid}",
            disabled=turn_in_progress,
            horizontal=True,
        )
        force_plan_value = False
        force_team_value = False
        if strategy_mode == "手动":
            force_plan_value = st.toggle(
                "启用 Plan",
                value=False,
                key=f"agent_force_plan_{project_uid}_{session_uid}",
                disabled=turn_in_progress,
            )
            force_team_value = st.toggle(
                "启用 Team",
                value=False,
                key=f"agent_force_team_{project_uid}_{session_uid}",
                disabled=turn_in_progress,
            )
            st.caption("手动模式下将覆盖自动策略判定。")
        else:
            st.caption("自动模式：由策略路由器决定是否启用 Plan/Team。")
    return resolve_strategy_flags(
        strategy_mode=strategy_mode,
        force_plan_value=force_plan_value,
        force_team_value=force_team_value,
    )


def render_session_info_panel(
    *,
    selected_session_name: str,
    selected_project_uid: str,
    conversation_key: str,
    turn_in_progress: bool,
    render_output_archive_fn,
    render_workflow_metrics_fn,
    render_context_usage_fn,
) -> None:
    st.markdown("### 会话信息")
    st.caption(f"当前会话：{selected_session_name}")
    render_output_archive_fn(selected_project_uid, disable_interaction=turn_in_progress)
    render_workflow_metrics_fn(conversation_key)
    render_context_usage_fn(conversation_key)
    if turn_in_progress:
        st.info("正在生成回答，已临时锁定归档与文档切换，避免中断当前对话。")


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


def render_pinned_todo_panel(
    *,
    project_uid: str,
    expanded: bool = True,
) -> None:
    path = _resolve_todo_path()
    records = _load_todo_records(path)
    if not records:
        with st.expander("📌 Todo（Pinned）", expanded=expanded):
            st.caption("暂无任务。可通过 write_todo 工具创建。")
        return

    with st.expander("📌 Todo（Pinned）", expanded=expanded):
        show_done = st.toggle(
            "显示已完成",
            value=False,
            key=f"todo_show_done_{project_uid}",
        )
        items = []
        for item in records:
            status = str(item.get("status") or "todo").strip().lower()
            if not show_done and status in {"done", "canceled"}:
                continue
            items.append(item)
        if not items:
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
            reverse=False,
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
            requests.append(
                {
                    "question": question,
                    "context": context,
                    "urgency": urgency,
                }
            )
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
