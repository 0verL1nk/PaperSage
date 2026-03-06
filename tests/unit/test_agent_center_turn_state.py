from agent.application.agent_center.turn_state import (
    append_assistant_turn_message,
    append_skill_context_texts,
    clear_turn_lock,
    enqueue_user_turn,
    resolve_active_prompt,
    store_turn_metrics,
)


def test_resolve_active_prompt_paths():
    prompt, state = resolve_active_prompt(
        turn_in_progress=True,
        pending_turn={
            "prompt": "q",
            "project_uid": "p1",
            "session_uid": "s1",
            "scope_signature": "sig",
        },
        prompt_input=None,
        project_uid="p1",
        session_uid="s1",
        scope_signature="sig",
    )
    assert prompt == "q"
    assert state == "resume_pending"

    prompt, state = resolve_active_prompt(
        turn_in_progress=True,
        pending_turn={"prompt": "q", "project_uid": "p2", "session_uid": "s2"},
        prompt_input=None,
        project_uid="p1",
        session_uid="s1",
        scope_signature="sig",
    )
    assert prompt is None
    assert state == "mismatch_pending"

    prompt, state = resolve_active_prompt(
        turn_in_progress=False,
        pending_turn=None,
        prompt_input="",
        project_uid="p1",
        session_uid="s1",
        scope_signature="sig",
    )
    assert prompt is None
    assert state == "no_prompt"

    prompt, state = resolve_active_prompt(
        turn_in_progress=False,
        pending_turn=None,
        prompt_input="new",
        project_uid="p1",
        session_uid="s1",
        scope_signature="sig",
    )
    assert prompt == "new"
    assert state == "new_prompt"


def test_turn_state_mutations():
    state = {}
    enqueue_user_turn(
        session_state=state,
        prompt="question",
        project_uid="p1",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_signature="sig",
    )
    assert state["agent_turn_in_progress"] is True
    assert state["agent_messages"][0]["role"] == "user"
    clear_turn_lock(state)
    assert state["agent_turn_in_progress"] is False
    assert state["agent_pending_turn"] is None

    append_skill_context_texts(
        session_state=state,
        conversation_key="p1:s1",
        skill_texts=["a", "b", "a"],
        max_items=2,
    )
    assert state["paper_project_skill_context_texts"]["p1:s1"] == ["a", "b"]

    store_turn_metrics(
        session_state=state,
        conversation_key="p1:s1",
        metrics={"latency_ms": 10},
    )
    assert state["paper_project_metrics"]["p1:s1"]["latency_ms"] == 10

    append_assistant_turn_message(
        session_state=state,
        answer="answer",
        trace_payload=[],
        mindmap_data=None,
        method_compare_data=None,
        ask_human_requests=[],
        evidence_items=[],
        policy_decision={},
        team_execution={},
        latency_ms=12.0,
        team_rounds=0,
        phase_path="phase1",
    )
    assert state["agent_messages"][-1]["role"] == "assistant"
    assert state["agent_messages"][-1]["content"] == "answer"
