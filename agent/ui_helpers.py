import json
import re
from html import escape
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from pyecharts import options as opts
from pyecharts.charts import Tree

from utils.compare_parser import method_compare_to_csv, parse_method_compare_payload
from utils.utils import extract_json_string

from .archive import list_agent_outputs
from .domain.trace import phase_label_from_performative, phase_summary
from .metrics import create_session_metrics, summarize_session_metrics

LEGACY_WORKFLOW_LABELS = {
    "react": "ReAct（Tool+Memory）",
    "plan_act": "Plan-Act（A2A协调）",
    "plan_act_replan": "Plan-Act-RePlan（A2A协调）",
}

_MINDMAP_HEIGHT_RE = re.compile(
    r"#mindmap\s*\{[^}]*height:\s*(\d+)px",
    re.IGNORECASE | re.DOTALL,
)


def _infer_mindmap_iframe_height(mindmap_html: str, default_height: int = 660) -> int:
    html = str(mindmap_html or "")
    match = _MINDMAP_HEIGHT_RE.search(html)
    if not match:
        return default_height
    try:
        chart_height = int(match.group(1))
    except Exception:
        return default_height
    # Chart height + panel header + body paddings/margins.
    frame_height = chart_height + 160
    return max(460, min(frame_height, 1200))


def _confidence_badge(score: float) -> tuple[str, str]:
    if score >= 0.66:
        return "高置信", "llm-badge-high"
    if score >= 0.4:
        return "中置信", "llm-badge-mid"
    return "低置信", "llm-badge-low"


def _render_acp_trace(
    trace_payload: list[dict[str, str]],
    *,
    expander_title: str = "查看 Agent 执行轨迹",
) -> None:
    def _render_trace_cards(items: list[tuple[int, dict[str, str]]]) -> None:
        trace_rows: list[str] = []
        for idx, entry in items:
            sender = escape(str(entry.get("sender", "unknown")))
            receiver = escape(str(entry.get("receiver", "unknown")))
            perf = escape(str(entry.get("performative", "message")))
            content = escape(str(entry.get("content", ""))).replace("\n", "<br>")
            trace_rows.append(
                (
                    "<div class='llm-trace-item'>"
                    f"<div class='llm-trace-item-head'>{idx}. {sender} -> {receiver} | {perf}</div>"
                    f"<div class='llm-trace-item-content'>{content}</div>"
                    "</div>"
                )
            )
        st.markdown(
            f"<div class='llm-trace-scroll'>{''.join(trace_rows)}</div>",
            unsafe_allow_html=True,
        )

    phase_groups: dict[str, list[tuple[int, dict[str, str]]]] = {}
    phase_order: list[str] = []
    indexed_payload = list(enumerate(trace_payload, start=1))
    for idx, entry in indexed_payload:
        phase = entry.get("phase")
        if not isinstance(phase, str) or not phase.strip():
            phase = _phase_label_from_performative(str(entry.get("performative", "")))
        if phase not in phase_groups:
            phase_groups[phase] = []
            phase_order.append(phase)
        phase_groups[phase].append((idx, entry))

    with st.expander(expander_title, expanded=False):
        tab_timeline, tab_phase = st.tabs(["时间线", "阶段分组"])
        with tab_timeline:
            _render_trace_cards(indexed_payload)
        with tab_phase:
            for phase in phase_order:
                items = phase_groups.get(phase, [])
                with st.expander(f"{phase}（{len(items)}）", expanded=False):
                    _render_trace_cards(items)


def _is_react_mode(policy_decision: dict[str, Any] | None) -> bool:
    if not isinstance(policy_decision, dict):
        return False
    return (
        not bool(policy_decision.get("plan_enabled", False))
        and not bool(policy_decision.get("team_enabled", False))
    )


def _render_trace_by_mode(
    trace_payload: list[dict[str, str]] | None,
    *,
    policy_decision: dict[str, Any] | None = None,
) -> None:
    if not isinstance(trace_payload, list) or not trace_payload:
        return
    if _is_react_mode(policy_decision):
        allowed = {
            "tool_load",
            "tool_call",
            "tool_result",
            "tool_activate",
            "skill_activate",
        }
        filtered = [
            item
            for item in trace_payload
            if isinstance(item, dict)
            and str(item.get("performative") or "").strip() in allowed
        ]
        if filtered:
            _render_acp_trace(filtered, expander_title="查看工具执行轨迹")
        return
    _render_acp_trace(trace_payload)


def _preview_text(text: str, limit: int = 140) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def _phase_label_from_performative(performative: str) -> str:
    return phase_label_from_performative(performative)


def _phase_summary(phase_labels: list[str]) -> str:
    return phase_summary(phase_labels)


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _message_attr(message, key: str, default=""):
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _build_react_trace(result: dict) -> list[dict[str, str]]:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not isinstance(messages, list):
        return []
    trace: list[dict[str, str]] = []
    last_sender = "user"
    for msg in messages:
        msg_type = str(_message_attr(msg, "type", "") or "").lower()
        role = str(_message_attr(msg, "role", "") or "").lower()
        content = _content_to_text(_message_attr(msg, "content", ""))
        tool_calls = _message_attr(msg, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool_name = str(call.get("name") or "tool")
                args = call.get("args")
                trace.append(
                    {
                        "sender": "react_agent",
                        "receiver": tool_name,
                        "performative": "tool_call",
                        "content": str(args) if args else content or "(tool call)",
                    }
                )
                last_sender = tool_name
        if msg_type == "tool":
            tool_name = _message_attr(msg, "name", "tool")
            trace.append(
                {
                    "sender": "react_agent",
                    "receiver": tool_name or "tool",
                    "performative": "tool_call",
                    "content": content or "(tool call)",
                }
            )
            last_sender = tool_name or "tool"
            continue
        if role == "tool":
            sender = _message_attr(msg, "name", "tool") or "tool"
            trace.append(
                {
                    "sender": sender,
                    "receiver": "react_agent",
                    "performative": "tool_result",
                    "content": content or "(tool result)",
                }
            )
            last_sender = sender
            continue
        if role == "user" or msg_type in {"human", "user"}:
            trace.append(
                {
                    "sender": "user",
                    "receiver": "react_agent",
                    "performative": "request",
                    "content": content,
                }
            )
            last_sender = "user"
            continue
        if role == "assistant" or msg_type in {"ai", "assistant"}:
            receiver = "user" if last_sender != "user" else "user"
            trace.append(
                {
                    "sender": "react_agent",
                    "receiver": receiver,
                    "performative": "final",
                    "content": content,
                }
            )
            last_sender = "react_agent"
    return trace


def _create_mindmap_chart(data: dict) -> Tree:
    return (
        Tree()
        .add(
            series_name="",
            data=[data],
            orient="LR",
            initial_tree_depth=3,
            layout="orthogonal",
            pos_left="3%",
            width="75%",
            height="420px",
            edge_fork_position="12%",
            symbol_size=7,
            label_opts=opts.LabelOpts(
                position="right",
                horizontal_align="left",
                vertical_align="middle",
                font_size=13,
                padding=[0, 0, 0, -18],
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="思维导图"),
            tooltip_opts=opts.TooltipOpts(trigger="item", trigger_on="mousemove"),
        )
    )


def _try_parse_mindmap(answer: str) -> dict | None:
    if not answer:
        return None
    try:
        json_str = extract_json_string(answer)
        parsed = json.loads(json_str)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    if "name" not in parsed:
        return None
    children = parsed.get("children")
    if children is not None and not isinstance(children, list):
        return None
    return parsed


def _render_mindmap_if_any(
    mindmap_data: dict | None,
    *,
    mindmap_html: str | None = None,
    render_error: str | None = None,
) -> None:
    if not mindmap_data:
        return
    if isinstance(mindmap_html, str) and mindmap_html.strip():
        try:
            iframe_height = _infer_mindmap_iframe_height(mindmap_html)
            components.html(mindmap_html, height=iframe_height, scrolling=False)
        except Exception:
            render_error = "思维导图 HTML 渲染失败，已自动回退到内置渲染。"
        else:
            if isinstance(render_error, str) and render_error.strip():
                st.warning(render_error)
            return
    try:
        chart = _create_mindmap_chart(mindmap_data)
        components.html(chart.render_embed(), height=480, scrolling=False)
    except Exception:
        st.warning("思维导图 JSON 已识别，但渲染失败。")
    if isinstance(render_error, str) and render_error.strip():
        st.warning(render_error)


def _normalize_evidence_items(raw_payload) -> list[dict]:
    if not isinstance(raw_payload, dict):
        return []
    evidences = raw_payload.get("evidences")
    if not isinstance(evidences, list):
        return []
    normalized: list[dict] = []
    for item in evidences:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        normalized.append(item)
    return normalized


_TODO_STATUS_ICON = {
    "todo": "⬜",
    "in_progress": "🔄",
    "done": "✅",
    "blocked": "🚫",
    "canceled": "⏹",
}
_TODO_PRIORITY_ICON = {
    "high": "🔴",
    "medium": "🟡",
    "low": "🟢",
}


def _render_team_todo_panel(
    team_execution: dict[str, Any],
    *,
    key_prefix: str = "todo",
) -> None:
    """渲染 team 模式下 leader 规划的 Todo 看板。"""
    if not isinstance(team_execution, dict):
        return
    if not team_execution.get("enabled"):
        return
    todo_records: list[dict[str, Any]] = team_execution.get("todo_records") or []
    if not todo_records:
        return

    todo_stats: dict[str, int] = team_execution.get("todo_stats") or {}
    done_n = int(todo_stats.get("done", 0))
    total_n = len(todo_records)
    blocked_n = int(todo_stats.get("blocked", 0))
    leader_planned = any(rec.get("leader_planned") for rec in todo_records)

    header = f"📋 Team Todo（{done_n}/{total_n} 完成"
    if blocked_n:
        header += f"，{blocked_n} 阻塞"
    if leader_planned:
        header += "，Leader 语义规划"
    header += "）"

    with st.expander(header, expanded=False):
        # 统计芯片行
        chips: list[str] = []
        for status in ("done", "in_progress", "todo", "blocked", "canceled"):
            n = int(todo_stats.get(status, 0))
            if n == 0:
                continue
            icon = _TODO_STATUS_ICON.get(status, "")
            chips.append(f"<span class='llm-chip'>{icon} {status} {n}</span>")
        if chips:
            st.markdown(
                f"<div class='llm-chip-row'>{''.join(chips)}</div>",
                unsafe_allow_html=True,
            )

        # 按轮次分组展示
        rounds: dict[int, list[dict[str, Any]]] = {}
        for rec in todo_records:
            r = int(rec.get("round") or 1)
            rounds.setdefault(r, []).append(rec)

        for round_idx in sorted(rounds.keys()):
            st.markdown(f"**Round {round_idx}**")
            for rec in rounds[round_idx]:
                status = str(rec.get("status") or "todo").lower()
                priority = str(rec.get("priority") or "medium").lower()
                s_icon = _TODO_STATUS_ICON.get(status, "❓")
                p_icon = _TODO_PRIORITY_ICON.get(priority, "")
                assignee = str(rec.get("assignee") or "n/a")
                title = str(rec.get("title") or rec.get("id") or "task")
                details = str(rec.get("details") or "").strip()
                deps = rec.get("dependencies") or []
                output = str(rec.get("output") or "").strip()

                with st.container(border=True):
                    # 标题行
                    st.markdown(
                        f"{s_icon} {p_icon} **{title}** "
                        f"&nbsp;`{assignee}`&nbsp; `{status}`",
                        unsafe_allow_html=True,
                    )
                    # 任务描述（leader 规划的 details）
                    if details:
                        st.caption(details)
                    # 依赖关系
                    if deps:
                        st.caption(f"依赖：{', '.join(str(d) for d in deps)}")
                    # 子 agent 输出（done 才展示）
                    if output and status == "done":
                        with st.expander("查看输出", expanded=False):
                            st.markdown(output)


def _render_evidence_panel(
    evidence_items: list[dict] | None,
    key_prefix: str,
    top_k: int = 3,
) -> None:
    if not isinstance(evidence_items, list) or not evidence_items:
        return
    with st.expander("查看证据详情", expanded=False):
        for index, item in enumerate(evidence_items[:top_k], start=1):
            chunk_id = item.get("chunk_id") or f"chunk_{index - 1}"
            doc_name = item.get("doc_name")
            score = item.get("score", 0.0)
            score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
            numeric_score = float(score) if isinstance(score, (int, float)) else 0.0
            confidence_text, confidence_cls = _confidence_badge(numeric_score)
            page_no = item.get("page_no")
            offset_start = item.get("offset_start")
            offset_end = item.get("offset_end")
            locator_parts: list[str] = []
            if isinstance(page_no, int):
                locator_parts.append(f"页码: {page_no}")
            if isinstance(offset_start, int) and isinstance(offset_end, int):
                locator_parts.append(f"偏移: {offset_start}-{offset_end}")
            locator_text = " | ".join(locator_parts) if locator_parts else "定位: n/a"

            source_doc_text = (
                f" | 来源文档: `{doc_name}`"
                if isinstance(doc_name, str) and doc_name.strip()
                else ""
            )
            with st.container(border=True):
                st.markdown(
                    (
                        f"**证据 {index}** | `{chunk_id}` | 分数: `{score_text}` | {locator_text}{source_doc_text}"
                        f" <span class='{confidence_cls}'>{confidence_text}</span>"
                    ),
                    unsafe_allow_html=True,
                )
                st.code(item.get("text", ""), language=None)
                citation = (
                    f"[{chunk_id}] score={score_text} {locator_text}"
                    f"{' source=' + str(doc_name) if isinstance(doc_name, str) and doc_name.strip() else ''}\n"
                    f"{item.get('text', '')}"
                )
                st.download_button(
                    "下载引用片段",
                    data=citation,
                    file_name=f"evidence_{chunk_id}.txt",
                    mime="text/plain",
                    key=f"{key_prefix}_evidence_{index}",
                )


def _normalize_ask_human_requests(raw_payload: Any) -> list[dict[str, str]]:
    if not isinstance(raw_payload, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_payload:
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
        normalized.append(
            {
                "question": question,
                "context": context,
                "urgency": urgency,
            }
        )
    return normalized


def _render_ask_human_requests(
    requests: list[dict[str, str]] | None,
    *,
    key_prefix: str,
) -> None:
    normalized = _normalize_ask_human_requests(requests)
    if not normalized:
        return
    urgency_badge = {"low": "低", "normal": "中", "high": "高"}
    with st.expander("需要人工确认", expanded=True):
        for index, item in enumerate(normalized, start=1):
            question = item["question"]
            context = item["context"]
            urgency = item["urgency"]
            st.markdown(
                f"**Q{index}**（紧急度：`{urgency_badge.get(urgency, '中')}`）"
            )
            st.write(question)
            if context:
                st.caption(f"上下文：{context}")
            st.text_input(
                "你的回复",
                key=f"{key_prefix}_ask_human_reply_{index}",
                placeholder="请输入你的确认或补充信息（当前仅展示，不自动回传）",
            )


def _try_parse_method_compare(answer: str) -> dict | None:
    return parse_method_compare_payload(answer)


def _render_method_compare_if_any(
    method_compare_data: dict | None,
    key_prefix: str,
) -> None:
    if not method_compare_data:
        return
    topic = method_compare_data.get("topic")
    if isinstance(topic, str) and topic:
        st.markdown(f"### 方法对比：{topic}")
    else:
        st.markdown("### 方法对比")
    rows = method_compare_data.get("rows")
    if isinstance(rows, list):
        st.table(rows)
    recommendation = method_compare_data.get("recommendation")
    if isinstance(recommendation, str) and recommendation:
        st.markdown(f"**建议**：{recommendation}")
    csv_data = method_compare_to_csv(method_compare_data)
    if csv_data:
        st.download_button(
            "下载对比 CSV",
            data=csv_data,
            file_name="method_compare.csv",
            mime="text/csv",
            key=f"{key_prefix}_compare_csv",
        )


def _infer_output_type(prompt: str, mindmap_data: dict | None) -> str:
    if mindmap_data:
        return "mindmap"
    return "qa"


def _render_output_archive(project_uid: str, disable_interaction: bool = False) -> None:
    user_uuid = st.session_state.get("uuid", "local-user")
    records = list_agent_outputs(uuid=user_uuid, project_uid=project_uid, limit=20)
    if disable_interaction:
        st.caption("对话生成中，已临时锁定历史归档交互。")
        return
    with st.expander("查看历史产出归档", expanded=False):
        if not records:
            st.caption("暂无归档记录")
            return

        options = [
            f"{item['created_at']} | {item['output_type']} | {item.get('doc_name') or 'project'}"
            for item in records
        ]
        selected_label = st.selectbox(
            "选择归档记录",
            options,
            key=f"archive_{project_uid}",
        )
        selected_index = options.index(selected_label)
        selected = records[selected_index]
        content = selected["content"]
        if selected["output_type"] == "mindmap":
            try:
                _render_mindmap_if_any(json.loads(content))
            except Exception:
                st.write(content)
        else:
            st.write(content)


def _get_doc_metrics(doc_uid: str) -> dict:
    doc_metrics_map = st.session_state.get("paper_project_metrics", {})
    current = doc_metrics_map.get(doc_uid)
    if not isinstance(current, dict):
        current = create_session_metrics()
        doc_metrics_map[doc_uid] = current
        st.session_state.paper_project_metrics = doc_metrics_map
    return current


def _render_workflow_metrics(doc_uid: str) -> None:
    summary = summarize_session_metrics(_get_doc_metrics(doc_uid))
    with st.expander("查看执行策略指标（会话内）", expanded=False):
        cols = st.columns(3)
        cols[0].metric("总问答数", summary["total_queries"])
        cols[1].metric("平均耗时 (ms)", f"{summary['average_latency_ms']:.0f}")
        cols[2].metric("最大耗时 (ms)", f"{summary['max_latency_ms']:.0f}")
        st.caption(
            "策略开关："
            f"plan={summary['plan_enabled_count']} ({summary['plan_enabled_ratio'] * 100:.1f}%) | "
            f"team={summary['team_enabled_count']} ({summary['team_enabled_ratio'] * 100:.1f}%)"
        )
        st.caption(
            f"Team 轮次：total={summary['team_rounds_total']} | "
            f"max={summary['team_rounds_max']} | "
            f"avg={summary['average_team_rounds']:.2f} | "
            f"fallback={summary['team_fallback_count']}"
        )
        st.caption(f"Trace 事件总数：{summary['trace_events_total']}")


def _render_context_usage(doc_uid: str) -> None:
    usage_map = st.session_state.get("paper_project_context_usage", {})
    usage = usage_map.get(doc_uid)
    if not isinstance(usage, dict):
        return
    breakdown = usage.get("breakdown")
    if not isinstance(breakdown, dict):
        return

    model_window_tokens = int(usage.get("model_window_tokens", 0))
    used_tokens = int(usage.get("used_tokens", 0))
    free_tokens = int(usage.get("free_tokens", 0))
    reserve_output = int(usage.get("reserved_output_tokens", 0))
    tools_count = int(usage.get("tools_count", 0))
    skills_loaded_count = int(usage.get("skills_loaded_count", 0))
    primary_agent_name = str(usage.get("primary_agent_name", "react_agent") or "react_agent")
    ratio = 0.0
    if model_window_tokens > 0:
        ratio = (used_tokens / model_window_tokens) * 100.0

    health_text = "健康"
    health_cls = "llm-context-health"
    if ratio >= 90:
        health_text = "高负载"
        health_cls = "llm-context-health llm-context-health-risk"
    elif ratio >= 75:
        health_text = "接近上限"
        health_cls = "llm-context-health llm-context-health-warn"

    with st.expander("Context Usage（可视化）", expanded=False):
        st.markdown(
            (
                "<div class='llm-chip-row'>"
                "<span class='llm-chip'>视角 主对话代理优先</span>"
                f"<span class='llm-chip'>主代理 {primary_agent_name}</span>"
                f"<span class='llm-chip'>Skills 按需加载 {skills_loaded_count}</span>"
                f"<span class='llm-chip'>Tools 已注册 {tools_count}（无需激活）</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<div class='llm-context-card'>"
                "<div class='llm-context-head'>"
                "<span class='llm-context-title'>模型上下文窗口</span>"
                f"<span class='{health_cls}'>{health_text} {ratio:.1f}%</span>"
                "</div>"
                "<div class='llm-context-kpi-row'>"
                "<div class='llm-context-kpi'>"
                "<div class='k'>已使用</div>"
                f"<div class='v'>{used_tokens:,}</div>"
                "</div>"
                "<div class='llm-context-kpi'>"
                "<div class='k'>窗口总量</div>"
                f"<div class='v'>{model_window_tokens:,}</div>"
                "</div>"
                "<div class='llm-context-kpi'>"
                "<div class='k'>预留输出</div>"
                f"<div class='v'>{reserve_output:,}</div>"
                "</div>"
                "<div class='llm-context-kpi'>"
                "<div class='k'>剩余空间</div>"
                f"<div class='v'>{free_tokens:,}</div>"
                "</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        color_map = {
            "system_prompt": "var(--ctx-cat-1)",
            "custom_agents": "var(--ctx-cat-2)",
            "memory_files": "var(--ctx-cat-3)",
            "skills": "var(--ctx-cat-4)",
            "tools": "var(--ctx-cat-5)",
            "messages": "var(--ctx-cat-6)",
            "free_space": "var(--ctx-cat-7)",
            "autocompact_buffer": "var(--ctx-cat-8)",
        }
        fallback_order = [
            ("system_prompt", "System prompt"),
            ("custom_agents", "Custom agents"),
            ("memory_files", "Memory files"),
            ("skills", "Skills"),
            ("tools", "Tools"),
            ("messages", "Messages"),
            ("free_space", "Free space"),
            ("autocompact_buffer", "Autocompact buffer"),
        ]
        raw_segments = usage.get("context_segments")
        segments: list[dict[str, float | int | str]] = []
        if isinstance(raw_segments, list):
            for item in raw_segments:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("key") or "")
                if not key:
                    continue
                label = str(item.get("label") or key)
                tokens = int(item.get("tokens") or 0)
                pct = max(0.0, float(item.get("pct") or 0.0))
                segments.append(
                    {
                        "key": key,
                        "label": label,
                        "tokens": tokens,
                        "pct": pct,
                        "color": color_map.get(key, "var(--ctx-cat-1)"),
                    }
                )
        if not segments:
            for key, label in fallback_order:
                item = breakdown.get(key, {})
                if not isinstance(item, dict):
                    continue
                tokens = int(item.get("tokens", 0))
                pct = max(0.0, float(item.get("pct", 0.0)))
                if pct <= 0:
                    continue
                segments.append(
                    {
                        "key": key,
                        "label": label,
                        "tokens": tokens,
                        "pct": pct,
                        "color": color_map.get(key, "var(--ctx-cat-1)"),
                    }
                )

        matrix_cell_count = 240
        entries: list[dict[str, float | int | str]] = []
        for item in segments:
            key = str(item["key"])
            label = str(item["label"])
            color = str(item["color"])
            tokens = int(item["tokens"])
            pct = float(item["pct"])
            exact_cells = (pct / 100.0) * matrix_cell_count
            base_cells = int(exact_cells)
            remainder = exact_cells - float(base_cells)
            entries.append(
                {
                    "key": key,
                    "label": label,
                    "color": color,
                    "tokens": tokens,
                    "pct": pct,
                    "cells": base_cells,
                    "remainder": remainder,
                }
            )

        assigned_cells = int(sum(int(item["cells"]) for item in entries))
        remaining_cells = max(0, matrix_cell_count - assigned_cells)
        for item in sorted(entries, key=lambda x: float(x["remainder"]), reverse=True):
            if remaining_cells <= 0:
                break
            item["cells"] = int(item["cells"]) + 1
            remaining_cells -= 1

        used_cells = int(sum(int(item["cells"]) for item in entries))
        if used_cells > matrix_cell_count:
            overflow = used_cells - matrix_cell_count
            for item in sorted(entries, key=lambda x: int(x["cells"]), reverse=True):
                if overflow <= 0:
                    break
                cut = min(overflow, int(item["cells"]))
                item["cells"] = int(item["cells"]) - cut
                overflow -= cut

        # Ensure each non-zero segment is visible at least once in the matrix.
        for item in entries:
            if float(item["pct"]) <= 0 or int(item["cells"]) > 0:
                continue
            donor = max(entries, key=lambda x: int(x["cells"]))
            if int(donor["cells"]) <= 1:
                continue
            donor["cells"] = int(donor["cells"]) - 1
            item["cells"] = 1

        matrix_cells: list[str] = []
        for item in entries:
            color = str(item["color"])
            for _ in range(int(item["cells"])):
                matrix_cells.append(
                    f"<span class='llm-context-cell' style='background:{color}'></span>"
                )
        if len(matrix_cells) < matrix_cell_count:
            matrix_cells.extend(
                "<span class='llm-context-cell empty'></span>"
                for _ in range(matrix_cell_count - len(matrix_cells))
            )

        st.markdown(
            f"<div class='llm-context-matrix'>{''.join(matrix_cells)}</div>",
            unsafe_allow_html=True,
        )

        rows_html: list[str] = []
        for item in entries:
            label = str(item["label"])
            color = str(item["color"])
            tokens = int(item["tokens"])
            pct = float(item["pct"])
            rows_html.append(
                (
                    "<div class='llm-context-legend-item'>"
                    "<div class='llm-context-legend-left'>"
                    f"<span class='llm-context-dot' style='background:{color}'></span>"
                    f"<span class='llm-context-legend-label'>{label}</span>"
                    "</div>"
                    f"<span class='llm-context-legend-value'>{tokens:,} tokens · {pct:.1f}%</span>"
                    "</div>"
                )
            )
        st.markdown("".join(rows_html), unsafe_allow_html=True)


def _render_chat_history(chat_messages: list[dict]) -> None:
    for message_index, message in enumerate(chat_messages):
        with st.chat_message(message["role"]):
            if message.get("auto_compact"):
                with st.expander("自动压缩摘要（历史对话）", expanded=False):
                    st.write(message["content"])
                continue
            mindmap_data = message.get("mindmap_data")
            if mindmap_data:
                _render_mindmap_if_any(
                    mindmap_data,
                    mindmap_html=message.get("mindmap_html"),
                    render_error=message.get("mindmap_render_error"),
                )
                with st.expander("查看思维导图 JSON", expanded=False):
                    st.code(str(message.get("content", "")), language="json")
            else:
                st.write(message["content"])
            policy_decision = message.get("policy_decision")
            if isinstance(policy_decision, dict):
                plan_enabled = bool(policy_decision.get("plan_enabled", False))
                team_enabled = bool(policy_decision.get("team_enabled", False))
                reason = str(policy_decision.get("reason") or "").strip()
                source = str(policy_decision.get("source") or "").strip()
                chips = (
                    f"<span class='llm-chip'>Plan {'ON' if plan_enabled else 'OFF'}</span>"
                    f"<span class='llm-chip'>Team {'ON' if team_enabled else 'OFF'}</span>"
                )
                if source:
                    chips += f"<span class='llm-chip'>Source {source}</span>"
                if reason:
                    chips += f"<span class='llm-chip'>{reason}</span>"
                st.markdown(
                    (
                        "<div class='llm-chip-row'>"
                        f"{chips}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            else:
                workflow_mode = message.get("workflow_mode")
                if isinstance(workflow_mode, str):
                    label = LEGACY_WORKFLOW_LABELS.get(workflow_mode, workflow_mode)
                    reason = message.get("workflow_reason") or ""
                    reason_chip = (
                        f"<span class='llm-chip'>{reason}</span>"
                        if isinstance(reason, str) and reason
                        else ""
                    )
                    st.markdown(
                        (
                            "<div class='llm-chip-row'>"
                            f"<span class='llm-chip'>自动路由 {label}</span>"
                            f"{reason_chip}"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

            team_execution = message.get("team_execution")
            if isinstance(team_execution, dict):
                member_count = int(team_execution.get("member_count", 0))
                rounds = int(team_execution.get("rounds", 0))
                fallback_reason = str(team_execution.get("fallback_reason") or "").strip()
                fallback_chip = (
                    f"<span class='llm-chip'>fallback {fallback_reason}</span>"
                    if fallback_reason
                    else ""
                )
                st.markdown(
                    (
                        "<div class='llm-chip-row'>"
                        f"<span class='llm-chip'>成员 {member_count}</span>"
                        f"<span class='llm-chip'>轮次 {rounds}</span>"
                        f"{fallback_chip}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            latency_ms = message.get("latency_ms")
            if isinstance(latency_ms, (int, float)):
                team_rounds = int(message.get("team_rounds", message.get("replan_rounds", 0)))
                st.markdown(
                    (
                        "<div class='llm-chip-row'>"
                        f"<span class='llm-chip'>耗时 {float(latency_ms):.0f} ms</span>"
                        f"<span class='llm-chip'>Team rounds {team_rounds}</span>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            phase_path = message.get("phase_path")
            if isinstance(phase_path, str) and phase_path:
                st.markdown(
                    f"<div class='llm-chip-row'><span class='llm-chip'>执行阶段 {phase_path}</span></div>",
                    unsafe_allow_html=True,
                )
            trace_payload = message.get("acp_trace")
            _render_trace_by_mode(
                trace_payload if isinstance(trace_payload, list) else None,
                policy_decision=policy_decision if isinstance(policy_decision, dict) else None,
            )
            _render_method_compare_if_any(
                message.get("method_compare_data"),
                key_prefix=f"history_{message_index}",
            )
            _render_ask_human_requests(
                message.get("ask_human_requests"),
                key_prefix=f"history_{message_index}",
            )
            _render_evidence_panel(
                message.get("evidence_items"),
                key_prefix=f"history_{message_index}",
            )
