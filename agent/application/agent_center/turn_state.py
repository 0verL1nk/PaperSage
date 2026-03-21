from typing import Any


def resolve_active_prompt(
    *,
    turn_in_progress: bool,
    pending_turn: Any,
    prompt_input: str | None,
    project_uid: str,
    session_uid: str,
    scope_signature: str,
) -> tuple[str | None, str]:
    if (
        turn_in_progress
        and isinstance(pending_turn, dict)
        and pending_turn.get("project_uid") == project_uid
        and pending_turn.get("session_uid") == session_uid
        and pending_turn.get("scope_signature") == scope_signature
        and isinstance(pending_turn.get("prompt"), str)
    ):
        return str(pending_turn["prompt"]), "resume_pending"
    if turn_in_progress and isinstance(pending_turn, dict):
        return None, "mismatch_pending"
    if not isinstance(prompt_input, str) or not prompt_input:
        return None, "no_prompt"
    return prompt_input, "new_prompt"


def enqueue_user_turn(
    *,
    session_state: dict[str, Any],
    prompt: str,
    project_uid: str,
    session_uid: str,
    conversation_key: str,
    scope_signature: str,
) -> None:
    messages = session_state.get("agent_messages", [])
    if not isinstance(messages, list):
        messages = []
    messages.append({"role": "user", "content": prompt})
    session_state["agent_messages"] = messages
    session_state["agent_pending_turn"] = {
        "prompt": prompt,
        "project_uid": project_uid,
        "session_uid": session_uid,
        "conversation_key": conversation_key,
        "scope_signature": scope_signature,
    }
    session_state["agent_turn_in_progress"] = True


def clear_turn_lock(session_state: dict[str, Any]) -> None:
    session_state["agent_turn_in_progress"] = False
    session_state["agent_pending_turn"] = None


def append_skill_context_texts(
    *,
    session_state: dict[str, Any],
    conversation_key: str,
    skill_texts: list[str],
    max_items: int = 20,
) -> None:
    if not skill_texts:
        return
    skill_texts_map = session_state.get("paper_project_skill_context_texts", {})
    existing_texts = skill_texts_map.get(conversation_key, [])
    if not isinstance(existing_texts, list):
        existing_texts = []
    seen_texts = {str(item) for item in existing_texts}
    for text in skill_texts:
        value = str(text)
        if value in seen_texts:
            continue
        existing_texts.append(value)
        seen_texts.add(value)
    skill_texts_map[conversation_key] = existing_texts[-max_items:]
    session_state["paper_project_skill_context_texts"] = skill_texts_map


def store_turn_metrics(
    *,
    session_state: dict[str, Any],
    conversation_key: str,
    metrics: dict[str, Any],
) -> None:
    metrics_map = session_state.get("paper_project_metrics", {})
    if not isinstance(metrics_map, dict):
        metrics_map = {}
    metrics_map[conversation_key] = metrics
    session_state["paper_project_metrics"] = metrics_map


def append_assistant_turn_message(
    *,
    session_state: dict[str, Any],
    answer: str,
    trace_payload: list[dict[str, Any]],
    mindmap_data: dict[str, Any] | None,
    mindmap_html: str | None = None,
    mindmap_render_error: str | None = None,
    method_compare_data: dict[str, Any] | None,
    ask_human_requests: list[dict[str, str]] | None,
    evidence_items: list[dict[str, Any]],
    policy_decision: dict[str, Any],
    team_execution: dict[str, Any],
    team_handoff: dict[str, Any] | None = None,
    todo_scheduler_hint: dict[str, Any] | None = None,
    latency_ms: float,
    team_rounds: int,
    phase_path: str,
) -> None:
    messages = session_state.get("agent_messages", [])
    if not isinstance(messages, list):
        messages = []
    messages.append(
        {
            "role": "assistant",
            "content": answer,
            "acp_trace": trace_payload,
            "mindmap_data": mindmap_data,
            "mindmap_html": mindmap_html,
            "mindmap_render_error": mindmap_render_error,
            "method_compare_data": method_compare_data,
            "ask_human_requests": ask_human_requests if isinstance(ask_human_requests, list) else [],
            "evidence_items": evidence_items,
            "policy_decision": policy_decision,
            "team_execution": team_execution,
            "team_handoff": team_handoff if isinstance(team_handoff, dict) else None,
            "todo_scheduler_hint": (
                todo_scheduler_hint if isinstance(todo_scheduler_hint, dict) else None
            ),
            "latency_ms": latency_ms,
            "team_rounds": team_rounds,
            "phase_path": phase_path,
        }
    )
    session_state["agent_messages"] = messages
