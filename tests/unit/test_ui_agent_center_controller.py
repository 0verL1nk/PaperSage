from ui.agent_center.controller import (
    consume_session_selector_uid_override,
    override_session_selector_uid,
    resolve_initial_session_selector_uid,
    resolve_session_selector_uid,
    session_selector_override_key,
)


def test_resolve_session_selector_uid_prefers_valid_widget_state() -> None:
    session_state = {"agent_project_session_selector_p1": "s2"}
    by_uid = {"s1": {"session_uid": "s1"}, "s2": {"session_uid": "s2"}}

    selected = resolve_session_selector_uid(
        session_state=session_state,
        selector_key="agent_project_session_selector_p1",
        fallback_uid="s1",
        by_uid=by_uid,
    )

    assert selected == "s2"


def test_resolve_session_selector_uid_falls_back_when_widget_state_invalid() -> None:
    session_state = {"agent_project_session_selector_p1": "missing"}
    by_uid = {"s1": {"session_uid": "s1"}, "s2": {"session_uid": "s2"}}

    selected = resolve_session_selector_uid(
        session_state=session_state,
        selector_key="agent_project_session_selector_p1",
        fallback_uid="s1",
        by_uid=by_uid,
    )

    assert selected == "s1"


def test_override_session_selector_uid_updates_widget_state() -> None:
    session_state = {"agent_project_session_selector_p1": "s1"}

    override_session_selector_uid(
        session_state=session_state,
        selector_key="agent_project_session_selector_p1",
        selected_uid="s3",
    )

    assert (
        session_state[session_selector_override_key("agent_project_session_selector_p1")] == "s3"
    )


def test_consume_session_selector_uid_override_prefers_pending_override() -> None:
    session_state = {
        "agent_project_session_selector_p1": "s1",
        session_selector_override_key("agent_project_session_selector_p1"): "s3",
    }
    by_uid = {
        "s1": {"session_uid": "s1"},
        "s2": {"session_uid": "s2"},
        "s3": {"session_uid": "s3"},
    }

    selected = consume_session_selector_uid_override(
        session_state=session_state,
        selector_key="agent_project_session_selector_p1",
        by_uid=by_uid,
    )

    assert selected == "s3"
    assert session_selector_override_key("agent_project_session_selector_p1") not in session_state


def test_consume_session_selector_uid_override_ignores_invalid_override() -> None:
    session_state = {
        session_selector_override_key("agent_project_session_selector_p1"): "missing",
    }
    by_uid = {"s1": {"session_uid": "s1"}}

    selected = consume_session_selector_uid_override(
        session_state=session_state,
        selector_key="agent_project_session_selector_p1",
        by_uid=by_uid,
    )

    assert selected == ""
    assert session_selector_override_key("agent_project_session_selector_p1") not in session_state


def test_resolve_initial_session_selector_uid_prefers_override_over_widget_state() -> None:
    session_state = {
        "agent_project_session_selector_p1": "s1",
        session_selector_override_key("agent_project_session_selector_p1"): "s3",
    }
    by_uid = {
        "s1": {"session_uid": "s1"},
        "s2": {"session_uid": "s2"},
        "s3": {"session_uid": "s3"},
    }

    selected = resolve_initial_session_selector_uid(
        session_state=session_state,
        selector_key="agent_project_session_selector_p1",
        fallback_uid="s2",
        by_uid=by_uid,
    )

    assert selected == "s3"
    assert session_selector_override_key("agent_project_session_selector_p1") not in session_state
