from ui.page_bootstrap import bootstrap_page_context


def test_bootstrap_page_context_sets_default_uuid_and_state(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(
        "ui.page_bootstrap.init_database",
        lambda db_name: calls.append(("init_database", (db_name,), {})),
    )
    monkeypatch.setattr(
        "ui.page_bootstrap.ensure_local_user",
        lambda user_uuid, db_name: calls.append(
            ("ensure_local_user", (user_uuid,), {"db_name": db_name})
        ),
    )
    monkeypatch.setattr(
        "ui.page_bootstrap.ensure_default_project_for_user",
        lambda *args, **kwargs: calls.append(("ensure_default_project_for_user", args, kwargs)),
    )

    state: dict[str, object] = {}
    user_uuid = bootstrap_page_context(
        session_state=state,
        db_name="./test.sqlite",
        state_defaults={"files": [], "projects": []},
    )

    assert user_uuid == "local-user"
    assert state["uuid"] == "local-user"
    assert state["files"] == []
    assert state["projects"] == []
    assert ("init_database", ("./test.sqlite",), {}) in calls
    assert ("ensure_local_user", ("local-user",), {"db_name": "./test.sqlite"}) in calls
    assert not any(name == "ensure_default_project_for_user" for name, _, _ in calls)


def test_bootstrap_page_context_calls_default_project_when_enabled(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr("ui.page_bootstrap.init_database", lambda _db_name: None)
    monkeypatch.setattr("ui.page_bootstrap.ensure_local_user", lambda _uuid, db_name: None)
    monkeypatch.setattr(
        "ui.page_bootstrap.ensure_default_project_for_user",
        lambda *args, **kwargs: calls.append(("ensure_default_project_for_user", args, kwargs)),
    )

    state: dict[str, object] = {"uuid": "u-1"}
    user_uuid = bootstrap_page_context(
        session_state=state,
        db_name="./test.sqlite",
        ensure_default_project=True,
        sync_existing_files=True,
    )

    assert user_uuid == "u-1"
    assert calls == [
        (
            "ensure_default_project_for_user",
            ("u-1",),
            {"db_name": "./test.sqlite", "sync_existing_files": True},
        )
    ]
