import json
import re
from typing import Any

import streamlit as st

from agent.ui_helpers import (
    _phase_label_from_performative,
    _preview_text,
    _render_ask_human_requests,
    _render_evidence_panel,
    _render_method_compare_if_any,
    _render_mindmap_if_any,
    _render_team_todo_panel,
    _render_trace_by_mode,
)


def _count_replan_rounds(trace_payload: list[dict[str, str]]) -> int:
    return sum(
        1
        for item in trace_payload
        if isinstance(item, dict) and str(item.get("performative") or "").strip() == "replan"
    )


def build_status_event_line(*, event_index: int, item: dict[str, str]) -> tuple[str, str]:
    phase = _phase_label_from_performative(str(item.get("performative", "")))
    label = f"执行中... [阶段: {phase}]"
    line = (
        f"{event_index}. "
        f"`{item.get('sender', 'unknown')} -> {item.get('receiver', 'unknown')}` | "
        f"`{item.get('performative', 'message')}` | "
        f"{phase} | {_preview_text(str(item.get('content', '')))}"
    )
    return label, line


def render_turn_result(
    *,
    answer: str,
    policy_decision: dict[str, Any],
    team_execution: dict[str, Any],
    trace_payload: list[dict[str, str]],
    evidence_items: list[dict],
    mindmap_data: dict | None,
    method_compare_data: dict | None,
    ask_human_requests: list[dict[str, str]] | None,
    run_latency_ms: float,
    phase_path: str,
    mindmap_html: str | None = None,
    mindmap_render_error: str | None = None,
) -> None:
    # Extract report if exists
    report_match = re.search(r"<report>(.*?)</report>", str(answer or ""), flags=re.DOTALL | re.IGNORECASE)
    report_content = report_match.group(1).strip() if report_match else None

    # Remove tags for summary text
    summary_text = re.sub(
        r"<mindmap>\s*\{.*?\}\s*</mindmap>",
        "",
        str(answer or ""),
        flags=re.DOTALL | re.IGNORECASE,
    )
    summary_text = re.sub(
        r"<report>.*?</report>",
        "",
        summary_text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    st.markdown(
        (
            "<div class='llm-chip-row'>"
            f"<span class='llm-chip'>Plan {'ON' if policy_decision.get('plan_enabled') else 'OFF'}</span>"
            f"<span class='llm-chip'>Team {'ON' if policy_decision.get('team_enabled') else 'OFF'}</span>"
            f"<span class='llm-chip'>Source {policy_decision.get('source', 'fallback')}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    policy_reason = str(policy_decision.get("reason") or "").strip()
    if policy_reason:
        st.caption(f"策略原因：{policy_reason}")

    if mindmap_data:
        if summary_text:
            st.write(summary_text)
        _render_mindmap_if_any(
            mindmap_data,
            mindmap_html=mindmap_html,
            render_error=mindmap_render_error,
        )
        with st.expander("查看思维导图 JSON", expanded=False):
            st.code(json.dumps(mindmap_data, ensure_ascii=False, indent=2), language="json")
    else:
        if report_content:
            if summary_text:
                st.write(summary_text)

            st.markdown(f"""
            <div style="
                border: 1px solid var(--line-soft);
                border-top: 4px solid var(--ink-500);
                border-radius: 8px;
                padding: 0;
                margin-top: 16px;
                background-color: var(--paper);
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            ">
                <div style="
                    background-color: var(--sky-50);
                    padding: 12px 20px;
                    border-bottom: 1px solid var(--line-soft);
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    display: flex;
                    align-items: center;
                ">
                    <span style="font-size: 18px; margin-right: 8px;">📑</span>
                    <span style="font-weight: 600; color: var(--ink-900);">综合研究报告</span>
                    <span style="margin-left: auto; font-size: 12px; color: var(--ink-500); background: var(--sky-100); padding: 2px 8px; border-radius: 12px; font-weight: 500;">
                        {"Team Output" if policy_decision.get("team_enabled") else "Agent Output"}
                    </span>
                </div>
                <div style="padding: 24px; color: var(--ink-900);">
            """, unsafe_allow_html=True)

            st.markdown(report_content)

            st.markdown("</div></div>", unsafe_allow_html=True)

        else:
            st.write(summary_text)

    if phase_path and phase_path != "无":
        st.caption(f"执行阶段：{phase_path}")
    if trace_payload:
        _render_trace_by_mode(trace_payload, policy_decision=policy_decision)

    _render_method_compare_if_any(method_compare_data, key_prefix="live")
    _render_ask_human_requests(ask_human_requests, key_prefix="live")
    _render_team_todo_panel(team_execution, key_prefix="live")
    _render_evidence_panel(evidence_items, key_prefix="live")
    replan_rounds = _count_replan_rounds(trace_payload)
    st.caption(
        "本次耗时："
        f"{run_latency_ms:.0f} ms | "
        f"Team rounds：{int(team_execution.get('rounds', 0))} | "
        f"Replan rounds：{replan_rounds}"
    )
