import logging
import time
from typing import Any

import streamlit as st

from .logging_utils import logging_context
from .metrics import extract_replan_rounds
from .multi_agent_a2a import WORKFLOW_REACT
from .output_cleaner import sanitize_public_answer, split_public_answer_and_reasoning
from .stream import extract_result_text, extract_stream_text, extract_trace_events_from_update
from .ui_helpers import (
    _build_react_trace,
    _normalize_evidence_items,
    _phase_label_from_performative,
    _phase_summary,
    _preview_text,
    _render_acp_trace,
    _render_evidence_panel,
    _render_method_compare_if_any,
    _render_mindmap_if_any,
    _try_parse_method_compare,
    _try_parse_mindmap,
)
from .workflow_router import WORKFLOW_LABELS, auto_select_workflow_mode


def execute_assistant_turn(
    *,
    prompt: str,
    hinted_prompt: str,
    user_uuid: str,
    project_uid: str,
    selected_uid: str,
    run_id: str,
    coordinator: Any,
    session_id: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    trace_payload: list[dict[str, str]] = []
    evidence_items: list[dict] = []
    mindmap_data: dict | None = None
    method_compare_data: dict | None = None
    run_latency_ms = 0.0
    replan_rounds = 0
    phase_path = ""
    workflow_mode = WORKFLOW_REACT
    workflow_reason = ""
    answer = ""

    with logging_context(
        uid=user_uuid,
        project_uid=project_uid,
        doc_uid=selected_uid,
        run_id=run_id,
        session_id=session_id,
    ):
        logger.info("Agent run started: prompt_len=%s", len(prompt))
        try:
            with st.chat_message("assistant"):
                workflow_mode, workflow_reason = auto_select_workflow_mode(
                    prompt,
                    coordinator=coordinator,
                )
                with logging_context(workflow=workflow_mode):
                    logger.info(
                        "Workflow selected: mode=%s reason=%s",
                        workflow_mode,
                        workflow_reason,
                    )
                    st.caption(f"自动路由：{WORKFLOW_LABELS.get(workflow_mode, workflow_mode)}")
                    run_started = time.perf_counter()
                    if workflow_mode == WORKFLOW_REACT:
                        runtime_config = st.session_state.paper_agent_runtime_config
                        react_thread_id = (
                            runtime_config.get("configurable", {}).get("thread_id")
                            if isinstance(runtime_config, dict)
                            else None
                        )
                        with st.status("ReAct 执行中...", expanded=True) as status:
                            trace_seen_keys: set[tuple[str, str, str, str]] = set()
                            live_trace: list[dict[str, str]] = []

                            def _append_trace(item: dict[str, str]) -> None:
                                sender = str(item.get("sender", "unknown"))
                                receiver = str(item.get("receiver", "unknown"))
                                perf = str(item.get("performative", "message"))
                                content = str(item.get("content", ""))
                                key = (sender, receiver, perf, content)
                                if key in trace_seen_keys:
                                    return
                                trace_seen_keys.add(key)
                                live_trace.append(
                                    {
                                        "sender": sender,
                                        "receiver": receiver,
                                        "performative": perf,
                                        "content": content,
                                    }
                                )
                                phase = _phase_label_from_performative(perf)
                                status.write(
                                    f"{len(live_trace)}. "
                                    f"`{sender} -> {receiver}` | "
                                    f"`{perf}` | {phase} | {_preview_text(content)}"
                                )

                            _append_trace(
                                {
                                    "sender": "user",
                                    "receiver": "react_agent",
                                    "performative": "request",
                                    "content": prompt,
                                }
                            )
                            answer_placeholder = st.empty()
                            streamed_parts: list[str] = []
                            result: Any = None
                            with logging_context(session_id=react_thread_id or session_id):
                                stream_failed = False
                                try:
                                    for stream_item in st.session_state.paper_agent.stream(
                                        {"messages": [{"role": "user", "content": hinted_prompt}]},
                                        config=runtime_config,
                                        stream_mode=["updates", "messages"],
                                    ):
                                        mode = ""
                                        payload = stream_item
                                        if (
                                            isinstance(stream_item, tuple)
                                            and len(stream_item) == 2
                                            and isinstance(stream_item[0], str)
                                        ):
                                            mode = stream_item[0]
                                            payload = stream_item[1]

                                        if mode in {"", "messages"}:
                                            delta = extract_stream_text(payload)
                                            if delta:
                                                streamed_parts.append(delta)
                                                answer_placeholder.markdown("".join(streamed_parts))

                                        if mode in {"", "updates"}:
                                            for item in extract_trace_events_from_update(payload):
                                                if item.get("performative") == "final":
                                                    continue
                                                _append_trace(item)
                                except Exception as exc:
                                    stream_failed = True
                                    logger.warning("ReAct streaming failed, fallback to invoke: %s", exc)

                                if stream_failed or not streamed_parts:
                                    result = st.session_state.paper_agent.invoke(
                                        {"messages": [{"role": "user", "content": hinted_prompt}]},
                                        config=runtime_config,
                                    )
                            if streamed_parts:
                                answer = "".join(streamed_parts)
                            else:
                                answer = (
                                    extract_result_text(result) if isinstance(result, dict) else str(result)
                                )
                            if isinstance(result, dict):
                                for item in _build_react_trace(result):
                                    _append_trace(item)
                            answer, reasoning_text = split_public_answer_and_reasoning(answer)
                            if reasoning_text:
                                _append_trace(
                                    {
                                        "sender": "react_agent",
                                        "receiver": "user",
                                        "performative": "reasoning",
                                        "content": reasoning_text,
                                    },
                                )
                            if answer and not any(
                                item.get("performative") == "final" for item in live_trace
                            ):
                                _append_trace(
                                    {
                                        "sender": "react_agent",
                                        "receiver": "user",
                                        "performative": "final",
                                        "content": answer,
                                    }
                                )
                            trace_payload = live_trace
                            if not trace_payload:
                                status.write("本次未产生可展示轨迹。")
                            status.update(
                                label="ReAct 执行完成（轨迹已保留）",
                                state="complete",
                                expanded=True,
                            )
                        answer = sanitize_public_answer(answer)
                        if not answer:
                            answer = "抱歉，我暂时没有生成有效回复。"
                        answer_placeholder.markdown(answer)
                        phase_labels = [
                            _phase_label_from_performative(item.get("performative", ""))
                            for item in trace_payload
                        ]
                        phase_path = _phase_summary(phase_labels)
                        if phase_path and phase_path != "无":
                            st.caption(f"执行阶段：{phase_path}")
                        if trace_payload:
                            _render_acp_trace(trace_payload)
                    else:
                        event_logs: list[dict[str, str]] = []
                        phase_labels: list[str] = []
                        with st.status(
                            f"{WORKFLOW_LABELS.get(workflow_mode, workflow_mode)} 执行中...",
                            expanded=True,
                        ) as status:

                            def _on_event(item) -> None:
                                phase = _phase_label_from_performative(item.performative)
                                phase_labels.append(phase)
                                event_logs.append(
                                    {
                                        "sender": item.sender,
                                        "receiver": item.receiver,
                                        "performative": item.performative,
                                        "content": item.content,
                                        "phase": phase,
                                    }
                                )
                                status.update(
                                    label=(
                                        f"{WORKFLOW_LABELS.get(workflow_mode, workflow_mode)} 执行中..."
                                        f" [阶段: {phase}]"
                                    ),
                                    state="running",
                                    expanded=True,
                                )
                                status.write(
                                    f"{len(event_logs)}. "
                                    f"`{item.sender} -> {item.receiver}` | `{item.performative}` | "
                                    f"{phase} | {_preview_text(item.content)}"
                                )

                            answer, trace = coordinator.run(
                                hinted_prompt,
                                workflow_mode=workflow_mode,
                                on_event=_on_event,
                            )
                            status.update(label="A2A 协调完成", state="complete", expanded=False)
                        answer = sanitize_public_answer(answer)
                        if not answer:
                            answer = "抱歉，我暂时没有生成有效回复。"
                        st.write(answer)
                        trace_payload = event_logs if event_logs else [
                            {
                                "sender": item.sender,
                                "receiver": item.receiver,
                                "performative": item.performative,
                                "content": item.content,
                                "phase": _phase_label_from_performative(item.performative),
                            }
                            for item in trace
                        ]
                        if not phase_labels and trace_payload:
                            phase_labels = [
                                entry.get("phase", "处理中")
                                for entry in trace_payload
                                if isinstance(entry, dict)
                            ]
                        phase_path = _phase_summary(phase_labels)
                        if phase_path and phase_path != "无":
                            st.caption(f"执行阶段：{phase_path}")
                        if trace_payload:
                            _render_acp_trace(trace_payload)

                    evidence_retriever = st.session_state.get("paper_evidence_retriever")
                    if callable(evidence_retriever):
                        try:
                            evidence_payload = evidence_retriever(prompt)
                            evidence_items = _normalize_evidence_items(evidence_payload)
                            logger.info("Evidence retrieved: count=%s", len(evidence_items))
                        except Exception as exc:
                            logger.warning("Evidence retrieval failed: %s", exc)
                            evidence_items = []

                    method_compare_data = _try_parse_method_compare(answer)
                    mindmap_data = _try_parse_mindmap(answer)
                    _render_mindmap_if_any(mindmap_data)
                    _render_method_compare_if_any(method_compare_data, key_prefix="live")
                    _render_evidence_panel(evidence_items, key_prefix="live")
                    run_latency_ms = (time.perf_counter() - run_started) * 1000.0
                    replan_rounds = extract_replan_rounds(trace_payload)
                    logger.info(
                        "Agent run finished: latency_ms=%.2f trace_events=%s evidence_items=%s replan_rounds=%s",
                        run_latency_ms,
                        len(trace_payload),
                        len(evidence_items),
                        replan_rounds,
                    )
                    st.caption(f"本次耗时：{run_latency_ms:.0f} ms | 重规划轮次：{replan_rounds}")
        except Exception:
            logger.exception("Agent run failed")
            raise

    return {
        "answer": answer,
        "workflow_mode": workflow_mode,
        "workflow_reason": workflow_reason,
        "trace_payload": trace_payload,
        "evidence_items": evidence_items,
        "mindmap_data": mindmap_data,
        "method_compare_data": method_compare_data,
        "run_latency_ms": run_latency_ms,
        "replan_rounds": replan_rounds,
        "phase_path": phase_path,
    }
