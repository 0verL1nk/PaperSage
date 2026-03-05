from agent.application.agent_center.project_sessions import (
    build_session_maps,
    build_session_preview,
    create_and_select_session,
    delete_and_select_next_session,
    drop_agent_session_cache,
    drop_conversation_cache,
    ensure_project_sessions,
    format_session_option,
    normalize_selector_value,
    resolve_current_session_uid,
    should_allow_delete_session,
    update_selected_session_map,
)


def test_ensure_project_sessions_bootstraps_default():
    calls = {"list": 0, "default": 0}

    def _list_sessions(**_kwargs):
        calls["list"] += 1
        if calls["list"] == 1:
            return []
        return [{"session_uid": "s1"}]

    def _ensure_default(**_kwargs):
        calls["default"] += 1

    sessions = ensure_project_sessions(
        list_sessions_fn=_list_sessions,
        ensure_default_session_fn=_ensure_default,
        project_uid="p1",
        user_uuid="u1",
    )
    assert sessions == [{"session_uid": "s1"}]
    assert calls == {"list": 2, "default": 1}


def test_session_selection_and_formatting_helpers():
    by_uid, ordered = build_session_maps(
        [
            {"session_uid": "s1", "session_name": "会话A", "message_count": 3, "last_message": "hello"},
            {"session_uid": "s2", "session_name": "会话B", "message_count": 0, "last_message": ""},
        ]
    )
    assert ordered == ["s1", "s2"]
    assert resolve_current_session_uid(
        selected_map={"p1": "missing"},
        project_uid="p1",
        by_uid=by_uid,
        ordered_uids=ordered,
    ) == "s1"
    assert normalize_selector_value(selector_value="bad", by_uid=by_uid, fallback_uid="s2") == "s2"
    assert format_session_option(by_uid["s1"]) == "会话A · 3 条"
    assert build_session_preview(by_uid["s1"]) == "hello"
    assert should_allow_delete_session([{"session_uid": "s1"}]) is False
    assert should_allow_delete_session([{"session_uid": "s1"}, {"session_uid": "s2"}]) is True
    assert update_selected_session_map(
        selected_map={"p0": "s0"},
        project_uid="p1",
        selected_uid="s2",
    ) == {"p0": "s0", "p1": "s2"}


def test_create_delete_session_helpers():
    create_calls = {}
    delete_calls = {}
    drop_calls = {"agent": 0, "conversation": 0}

    selected_map = create_and_select_session(
        create_session_fn=lambda **kwargs: create_calls.update(kwargs) or {"session_uid": "s3"},
        selected_map={"p1": "s1"},
        project_uid="p1",
        user_uuid="u1",
        session_name="会话 3",
    )
    assert selected_map["p1"] == "s3"
    assert create_calls["session_name"] == "会话 3"

    selected_map, next_uid = delete_and_select_next_session(
        delete_session_fn=lambda **kwargs: delete_calls.update(kwargs),
        list_sessions_fn=lambda **_kwargs: [{"session_uid": "s2"}],
        drop_agent_session_cache_fn=lambda *_args: drop_calls.__setitem__("agent", drop_calls["agent"] + 1),
        drop_conversation_cache_fn=lambda *_args: drop_calls.__setitem__(
            "conversation", drop_calls["conversation"] + 1
        ),
        selected_map={"p1": "s3"},
        project_uid="p1",
        user_uuid="u1",
        selected_uid="s3",
    )
    assert delete_calls["session_uid"] == "s3"
    assert drop_calls == {"agent": 1, "conversation": 1}
    assert selected_map["p1"] == "s2"
    assert next_uid == "s2"


def test_drop_runtime_caches_for_session():
    state = {
        "paper_agent_sessions": {"leader:p1:s1": "agent", "leader:p1:s2": "other"},
        "paper_project_messages": {"p1:s1": [{"role": "user", "content": "q"}]},
        "paper_project_metrics": {"p1:s1": {"turns": 1}},
        "paper_project_compact_summaries": {"p1:s1": "summary"},
        "paper_project_context_usage": {"p1:s1": {"tokens": 10}},
        "paper_project_skill_context_texts": {"p1:s1": ["ctx"]},
    }
    drop_agent_session_cache(
        session_state=state,
        build_session_key_fn=lambda p, s, m: f"{m}:{p}:{s}",
        mode_leader="leader",
        project_uid="p1",
        session_uid="s1",
    )
    assert "leader:p1:s1" not in state["paper_agent_sessions"]
    drop_conversation_cache(
        session_state=state,
        build_conversation_key_fn=lambda p, s: f"{p}:{s}",
        project_uid="p1",
        session_uid="s1",
    )
    assert "p1:s1" not in state["paper_project_messages"]
