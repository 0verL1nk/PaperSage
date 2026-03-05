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
