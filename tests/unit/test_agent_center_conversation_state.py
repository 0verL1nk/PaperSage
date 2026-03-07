from types import SimpleNamespace

from agent.application.agent_center.conversation_state import (
    apply_auto_compact,
    ensure_compact_summary,
    ensure_conversation_messages,
    get_history_paging_state,
    load_more_conversation_messages,
    persist_active_conversation,
    update_context_usage,
)


class _State(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _FakeSt:
    def __init__(self):
        self.session_state = _State()
        self.captions: list[str] = []

    def caption(self, text: str) -> None:
        self.captions.append(text)


def test_persist_active_conversation_updates_session_state():
    st = _FakeSt()
    st.session_state["agent_messages"] = [{"role": "user", "content": "q"}]
    calls = {}

    def _save(**kwargs):
        calls.update(kwargs)

    persist_active_conversation(
        st=st,
        save_project_session_messages_fn=_save,
        list_project_session_messages_fn=lambda **_kwargs: [],
        user_uuid="u1",
        project_uid="p1",
        session_uid="s1",
        conversation_key="p1:s1",
    )

    assert calls["uuid"] == "u1"
    assert st.session_state["paper_project_messages"]["p1:s1"][0]["content"] == "q"


def test_ensure_conversation_messages_bootstrap_when_empty():
    st = _FakeSt()
    persisted_calls = {"called": 0}
    saved = {"called": 0}

    def _list_messages(**_kwargs):
        persisted_calls["called"] += 1
        return []

    def _persist(**_kwargs):
        saved["called"] += 1

    ensure_conversation_messages(
        st=st,
        list_project_session_messages_fn=_list_messages,
        persist_active_conversation_fn=_persist,
        user_uuid="u1",
        project_uid="p1",
        project_name="项目A",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_docs_count=2,
    )

    assert persisted_calls["called"] == 1
    assert saved["called"] == 1
    assert st.session_state["agent_messages"][0]["role"] == "assistant"


def test_ensure_conversation_messages_refreshes_cached_bootstrap_scope_count():
    st = _FakeSt()
    st.session_state["paper_project_messages"] = {
        "p1:s1": [
            {
                "role": "assistant",
                "content": "已加载项目《项目A》，当前检索范围 5 篇文档。 工作流将按问题自动路由。",
            }
        ]
    }
    saved = {"called": 0}

    ensure_conversation_messages(
        st=st,
        list_project_session_messages_fn=lambda **_kwargs: [],
        persist_active_conversation_fn=lambda **_kwargs: saved.__setitem__("called", saved["called"] + 1),
        user_uuid="u1",
        project_uid="p1",
        project_name="项目A",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_docs_count=6,
    )

    assert "6 篇文档" in st.session_state["agent_messages"][0]["content"]
    assert saved["called"] == 1


def test_ensure_conversation_messages_refreshes_persisted_bootstrap_scope_count():
    st = _FakeSt()
    saved = {"called": 0}

    ensure_conversation_messages(
        st=st,
        list_project_session_messages_fn=lambda **_kwargs: [
            {
                "role": "assistant",
                "content": "已加载项目《项目A》，当前检索范围 5 篇文档。 工作流将按问题自动路由。",
            }
        ],
        persist_active_conversation_fn=lambda **_kwargs: saved.__setitem__("called", saved["called"] + 1),
        user_uuid="u1",
        project_uid="p1",
        project_name="项目A",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_docs_count=6,
    )

    assert "6 篇文档" in st.session_state["agent_messages"][0]["content"]
    assert saved["called"] == 1


def test_compact_summary_context_and_auto_compact():
    st = _FakeSt()
    st.session_state["agent_messages"] = [{"role": "user", "content": "q"}]
    st.session_state["paper_project_tool_specs"] = {"p1": [{"name": "search_document"}]}
    st.session_state["paper_project_skill_context_texts"] = {"p1:s1": ["skill"]}
    saved_summary = {}

    ensure_compact_summary(
        st=st,
        get_project_session_compact_memory_fn=lambda **_kwargs: {"compact_summary": "seed"},
        user_uuid="u1",
        project_uid="p1",
        session_uid="s1",
        conversation_key="p1:s1",
    )
    update_context_usage(
        st=st,
        build_context_usage_snapshot_fn=lambda **kwargs: kwargs,
        project_uid="p1",
        conversation_key="p1:s1",
    )

    class _Result:
        messages = [{"role": "assistant", "content": "done"}]
        summary = "new-summary"
        compacted = True
        source_message_count = 2
        source_token_estimate = 100
        compacted_token_estimate = 40
        used_llm = False
        anchor_count = 1

    out = apply_auto_compact(
        st=st,
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        should_trigger_auto_compact_fn=lambda _messages: False,
        auto_compact_messages_fn=lambda *_args, **_kwargs: _Result(),
        build_openai_compatible_chat_model_fn=lambda **_kwargs: object(),
        get_user_api_key_fn=lambda: "",
        get_user_model_name_fn=lambda: "",
        get_user_base_url_fn=lambda: "",
        save_project_session_compact_memory_fn=lambda **kwargs: saved_summary.update(kwargs),
        persist_active_conversation_fn=lambda **_kwargs: None,
        project_uid="p1",
        session_uid="s1",
        user_uuid="u1",
        conversation_key="p1:s1",
    )

    assert out == "new-summary"
    assert st.session_state["paper_project_compact_summaries"]["p1:s1"] == "new-summary"
    assert saved_summary["compact_summary"] == "new-summary"
    assert st.captions


def test_update_context_usage_backfills_skill_texts_from_message_trace():
    st = _FakeSt()
    st.session_state["agent_messages"] = [
        {
            "role": "assistant",
            "content": "done",
            "acp_trace": [
                {
                    "performative": "skill_activate",
                    "receiver": "skill:summary",
                    "content": "activate summary",
                }
            ],
        }
    ]
    st.session_state["paper_project_tool_specs"] = {"p1": [{"name": "use_skill"}]}
    captured = {}

    def _snapshot(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    update_context_usage(
        st=st,
        build_context_usage_snapshot_fn=_snapshot,
        project_uid="p1",
        conversation_key="p1:s1",
        extract_skill_context_texts_from_trace_fn=lambda trace: [
            str(item.get("content") or "") for item in trace if isinstance(item, dict)
        ],
    )

    assert captured["skill_context_texts"] == ["activate summary"]
    assert st.session_state["paper_project_skill_context_texts"]["p1:s1"] == [
        "activate summary"
    ]


def test_ensure_conversation_messages_loads_latest_page_and_load_more():
    st = _FakeSt()
    all_messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]

    ensure_conversation_messages(
        st=st,
        list_project_session_messages_fn=lambda **_kwargs: list(all_messages),
        list_project_session_messages_page_fn=lambda *, offset, limit, **_kwargs: list(
            all_messages[offset : offset + limit]
        ),
        count_project_session_messages_fn=lambda **_kwargs: len(all_messages),
        persist_active_conversation_fn=lambda **_kwargs: None,
        user_uuid="u1",
        project_uid="p1",
        project_name="项目A",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_docs_count=2,
        history_page_size=3,
    )

    assert [item["content"] for item in st.session_state["agent_messages"]] == ["m5", "m6", "m7"]
    paging = get_history_paging_state(st=st, conversation_key="p1:s1")
    assert paging["total_count"] == 8
    assert paging["loaded_start"] == 5
    assert paging["has_more_before"] is True

    loaded = load_more_conversation_messages(
        st=st,
        list_project_session_messages_page_fn=lambda *, offset, limit, **_kwargs: list(
            all_messages[offset : offset + limit]
        ),
        user_uuid="u1",
        project_uid="p1",
        session_uid="s1",
        conversation_key="p1:s1",
        page_size=3,
    )
    assert loaded == 3
    assert [item["content"] for item in st.session_state["agent_messages"]] == [
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "m7",
    ]


def test_persist_active_conversation_merges_missing_prefix_when_paginated():
    st = _FakeSt()
    st.session_state["agent_messages"] = [
        {"role": "user", "content": "m2"},
        {"role": "assistant", "content": "m3"},
        {"role": "user", "content": "new"},
    ]
    st.session_state["paper_project_message_paging"] = {
        "p1:s1": {
            "total_count": 4,
            "loaded_start": 2,
            "loaded_count": 2,
            "page_size": 2,
        }
    }
    saved = {}
    persisted_full = [
        {"role": "assistant", "content": "m0"},
        {"role": "user", "content": "m1"},
        {"role": "user", "content": "m2"},
        {"role": "assistant", "content": "m3"},
    ]

    persist_active_conversation(
        st=st,
        save_project_session_messages_fn=lambda **kwargs: saved.update(kwargs),
        list_project_session_messages_fn=lambda **_kwargs: list(persisted_full),
        user_uuid="u1",
        project_uid="p1",
        session_uid="s1",
        conversation_key="p1:s1",
    )

    assert [item["content"] for item in saved["messages"]] == ["m0", "m1", "m2", "m3", "new"]
