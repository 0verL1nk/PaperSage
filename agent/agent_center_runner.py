import logging
from typing import Any

import streamlit as st

from .application.agent_center.turn_execution import (
    execute_turn_with_runtime,
    resolve_turn_runtime_inputs,
)
from .logging_utils import logging_context
from ui.agent_center_turn_view import (
    build_status_event_line,
    render_turn_result,
)


def execute_assistant_turn(
    *,
    prompt: str,
    hinted_prompt: str,
    user_uuid: str,
    project_uid: str,
    selected_uid: str,
    run_id: str,
    session_id: str,
    logger: logging.Logger,
    force_plan: bool | None = None,
    force_team: bool | None = None,
    routing_context: str = "",
) -> dict[str, Any]:
    trace_payload: list[dict[str, str]] = []
    evidence_items: list[dict] = []
    mindmap_data: dict | None = None
    method_compare_data: dict | None = None
    run_latency_ms = 0.0
    phase_path = ""
    answer = ""
    policy_decision: dict[str, Any] = {
        "plan_enabled": False,
        "team_enabled": False,
        "reason": "",
        "confidence": None,
        "source": "heuristic",
    }
    team_execution: dict[str, Any] = {
        "enabled": False,
        "roles": [],
        "member_count": 0,
        "rounds": 0,
        "summary": "",
        "fallback_reason": None,
    }

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
                runtime_inputs = resolve_turn_runtime_inputs(dict(st.session_state))

                event_count = [0]

                with st.status("执行策略编排中...", expanded=True) as status:

                    def _on_event(item: dict[str, str]) -> None:
                        event_count[0] += 1
                        label, line = build_status_event_line(
                            event_index=event_count[0],
                            item=item,
                        )
                        status.update(label=label, state="running", expanded=True)
                        status.write(line)

                    core_result = execute_turn_with_runtime(
                        prompt=prompt,
                        hinted_prompt=hinted_prompt,
                        runtime_inputs=runtime_inputs,
                        force_plan=force_plan,
                        force_team=force_team,
                        routing_context=routing_context,
                        on_event=_on_event,
                    )
                    status.update(label="执行完成", state="complete", expanded=False)

                answer = core_result["answer"]
                policy_decision = core_result["policy_decision"]
                team_execution = core_result["team_execution"]
                trace_payload = core_result["trace_payload"]
                evidence_items = core_result["evidence_items"]
                run_latency_ms = core_result["run_latency_ms"]
                phase_path = core_result["phase_path"]
                method_compare_data = core_result["method_compare_data"]
                mindmap_data = core_result["mindmap_data"]

                render_turn_result(
                    answer=answer,
                    policy_decision=policy_decision,
                    team_execution=team_execution,
                    trace_payload=trace_payload,
                    evidence_items=evidence_items,
                    mindmap_data=mindmap_data,
                    method_compare_data=method_compare_data,
                    run_latency_ms=run_latency_ms,
                    phase_path=phase_path,
                )
                logger.info(
                    "Agent run finished: latency_ms=%.2f trace_events=%s evidence_items=%s team_rounds=%s",
                    run_latency_ms,
                    len(trace_payload),
                    len(evidence_items),
                    int(team_execution.get("rounds", 0)),
                )
        except Exception:
            logger.exception("Agent run failed")
            raise

    return {
        "answer": answer,
        "policy_decision": policy_decision,
        "team_execution": team_execution,
        "trace_payload": trace_payload,
        "evidence_items": evidence_items,
        "mindmap_data": mindmap_data,
        "method_compare_data": method_compare_data,
        "run_latency_ms": run_latency_ms,
        "team_rounds": int(team_execution.get("rounds", 0)),
        "phase_path": phase_path,
    }
