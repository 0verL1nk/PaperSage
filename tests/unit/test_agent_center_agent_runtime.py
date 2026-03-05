from types import SimpleNamespace

from agent.application.agent_center.agent_runtime import (
    ensure_agent_runtime,
    prepare_agent_session,
)


def test_ensure_agent_runtime_reuses_cached_session():
    cached_agent = object()
    cached_runtime_config = {"configurable": {"thread_id": "t1"}}
    cached_retriever = lambda _q: {"evidences": [{"text": "x"}]}
    session_state = {
        "paper_agent_sessions": {
            "leader:p1:s1": {
                "agent": cached_agent,
                "runtime_config": cached_runtime_config,
                "tool_specs": [{"name": "search_document"}],
            }
        },
        "paper_evidence_retrievers": {"p1": cached_retriever},
        "paper_project_llms": {"p1": "llm"},
        "paper_project_search_document_fns": {"p1": lambda _q: "doc"},
        "paper_project_scope_signatures": {"p1": "sig"},
        "paper_project_tool_specs": {},
    }

    ensure_agent_runtime(
        session_state=session_state,
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        project_uid="p1",
        session_uid="s1",
        project_name="项目A",
        scope_docs=[{"uid": "d1", "file_name": "n1", "text": "t1"}],
        scope_signature="sig",
        mode_leader="leader",
        build_session_key_fn=lambda p, s, m: f"{m}:{p}:{s}",
        clear_project_runtime_fn=lambda _project_uid: None,
        normalize_evidence_items_fn=lambda payload: payload.get("evidences", []),
        get_user_api_key_fn=lambda: "",
        get_user_model_name_fn=lambda: "",
        get_user_base_url_fn=lambda: "",
        create_chat_model_fn=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not build llm")),
        create_project_evidence_retriever_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not build retriever")
        ),
        create_leader_session_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not create session")
        ),
    )

    assert session_state["paper_agent"] is cached_agent
    assert session_state["paper_agent_runtime_config"] == cached_runtime_config
    assert session_state["paper_evidence_retriever"] is cached_retriever


def test_ensure_agent_runtime_creates_fresh_session():
    session_state = {}
    retriever = lambda _q: {"evidences": [{"text": "chunk"}]}

    created = {"leader_session_calls": 0}

    def _create_leader_session(**kwargs):
        created["leader_session_calls"] += 1
        # Ensure search fn is wired to evidence retriever + normalize fn.
        assert kwargs["search_document_fn"]("q") == "chunk"
        return SimpleNamespace(
            agent="agent-obj",
            runtime_config={"configurable": {"thread_id": "t2"}},
            tool_specs=[{"name": "search_document"}],
        )

    ensure_agent_runtime(
        session_state=session_state,
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        project_uid="p1",
        session_uid="s1",
        project_name="项目A",
        scope_docs=[{"uid": "d1", "file_name": "n1", "text": "t1"}],
        scope_signature="sig-new",
        mode_leader="leader",
        build_session_key_fn=lambda p, s, m: f"{m}:{p}:{s}",
        clear_project_runtime_fn=lambda _project_uid: None,
        normalize_evidence_items_fn=lambda payload: payload.get("evidences", []),
        get_user_api_key_fn=lambda: "k",
        get_user_model_name_fn=lambda: "m",
        get_user_base_url_fn=lambda: "u",
        create_chat_model_fn=lambda **_kwargs: "llm-obj",
        create_project_evidence_retriever_fn=lambda **_kwargs: retriever,
        create_leader_session_fn=_create_leader_session,
    )

    assert created["leader_session_calls"] == 1
    assert session_state["paper_leader_llm"] == "llm-obj"
    assert session_state["paper_agent"] == "agent-obj"
    assert session_state["paper_project_scope_signatures"]["p1"] == "sig-new"


def test_prepare_agent_session_cached_and_build_paths():
    calls = {"ensure": 0, "caption": 0, "build": 0}

    def _ensure(*_args):
        calls["ensure"] += 1

    prepare_agent_session(
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        has_cached_session_fn=lambda *_args: True,
        ensure_agent_runtime_fn=_ensure,
        cached_caption_fn=lambda: calls.__setitem__("caption", calls["caption"] + 1),
        build_captioned_fn=lambda run: (calls.__setitem__("build", calls["build"] + 1), run()),
        project_uid="p1",
        session_uid="s1",
        project_name="项目A",
        scope_docs=[],
        scope_signature="sig",
    )
    assert calls == {"ensure": 1, "caption": 1, "build": 0}

    calls = {"ensure": 0, "caption": 0, "build": 0}
    prepare_agent_session(
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        has_cached_session_fn=lambda *_args: False,
        ensure_agent_runtime_fn=_ensure,
        cached_caption_fn=lambda: calls.__setitem__("caption", calls["caption"] + 1),
        build_captioned_fn=lambda run: (calls.__setitem__("build", calls["build"] + 1), run()),
        project_uid="p1",
        session_uid="s1",
        project_name="项目A",
        scope_docs=[],
        scope_signature="sig",
    )
    assert calls == {"ensure": 1, "caption": 0, "build": 1}
