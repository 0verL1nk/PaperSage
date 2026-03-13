from agent.adapters.agent_session import create_leader_session
from agent.adapters.archive import save_output
from agent.adapters.document import (
    extract_document_payload,
    load_cached_extraction,
    save_cached_extraction,
)
from agent.adapters.llm import create_chat_model
from agent.adapters.project_store import (
    create_session_for_project,
    delete_session_for_project,
    ensure_default_session_for_project,
    list_project_files_for_user,
    list_session_messages_for_project,
    list_sessions_for_project,
    list_user_projects,
    save_session_messages_for_project,
    update_session_for_project,
)
from agent.adapters.rag import create_project_evidence_retriever
from agent.adapters.user_settings import (
    apply_runtime_tuning_env_for_user,
    list_user_files,
    read_api_key_for_user,
    read_base_url_for_user,
    read_model_name_for_user,
    read_policy_router_api_key_for_user,
    read_policy_router_base_url_for_user,
    read_policy_router_model_name_for_user,
    read_runtime_tuning_settings_for_user,
    read_user_api_key,
    read_user_base_url,
    read_user_model_name,
    save_api_key_for_user,
    save_base_url_for_user,
    save_model_name_for_user,
    save_policy_router_api_key_for_user,
    save_policy_router_base_url_for_user,
    save_policy_router_model_name_for_user,
    save_runtime_tuning_settings_for_user,
)


def test_create_chat_model_delegates(monkeypatch):
    captured = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        return "llm"

    monkeypatch.setattr("agent.adapters.llm.build_openai_compatible_chat_model", _fake_builder)
    model = create_chat_model(api_key="k", model_name="m", base_url="u", temperature=0.2)
    assert model == "llm"
    assert captured["api_key"] == "k"
    assert captured["model_name"] == "m"
    assert captured["base_url"] == "u"
    assert captured["temperature"] == 0.2


def test_create_project_evidence_retriever_routes_single_or_multi(monkeypatch):
    ingest_calls = []

    class _FakeHit:
        def __init__(self, *, resource_id, content, score, metadata):
            self.resource_id = resource_id
            self.content = content
            self.score = score
            self.metadata = metadata

    class _FakeAdapter:
        def add_resource(self, *, project_uid, content, metadata=None):
            ingest_calls.append((project_uid, content, metadata or {}))
            return f"res-{len(ingest_calls)}"

        def search(self, request):
            assert request.project_uid == "p1"
            return [
                _FakeHit(
                    resource_id="res-2",
                    content="matched",
                    score=0.88,
                    metadata={
                        "namespace": "rag",
                        "doc_uid": "d2",
                        "doc_name": "n2",
                    },
                )
            ]

    monkeypatch.setattr("agent.adapters.rag.get_openviking_adapter", lambda: _FakeAdapter())
    monkeypatch.setattr(
        "agent.adapters.rag.load_agent_settings",
        lambda: type("_S", (), {"rag_top_k": 6})(),
    )

    retriever = create_project_evidence_retriever(
        documents=[
            {"doc_uid": "d1", "doc_name": "n1", "text": "t1"},
            {"doc_uid": "d2", "doc_name": "n2", "text": "t2"},
        ],
        project_uid="p1",
    )
    payload = retriever("query")

    assert len(ingest_calls) == 2
    assert all(call[2]["namespace"] == "rag" for call in ingest_calls)
    assert payload["evidences"][0]["text"] == "matched"
    assert payload["evidences"][0]["doc_uid"] == "d2"


def test_document_adapter_load_and_save(monkeypatch):
    monkeypatch.setattr(
        "agent.adapters.document.get_content_by_uid", lambda *_args, **_kwargs: "abc"
    )
    assert load_cached_extraction("u1") == "abc"

    monkeypatch.setattr("agent.adapters.document.get_content_by_uid", lambda *_args, **_kwargs: " ")
    assert load_cached_extraction("u1") is None

    captured = {}
    monkeypatch.setattr(
        "agent.adapters.document.save_content_to_database",
        lambda **kwargs: captured.update(kwargs),
    )
    save_cached_extraction(uid="u1", file_path="/tmp/x.pdf", content="doc")
    assert captured["uid"] == "u1"
    assert captured["content_type"] == "file_extraction"


def test_extract_document_payload_delegates(monkeypatch):
    monkeypatch.setattr(
        "agent.adapters.document.extract_files", lambda _path: {"result": 1, "text": "x"}
    )
    assert extract_document_payload("/tmp/a.pdf")["text"] == "x"


def test_create_leader_session_delegates(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "agent.adapters.agent_session.create_paper_agent_session",
        lambda **kwargs: captured.update(kwargs) or "session",
    )
    out = create_leader_session(
        llm="llm",
        search_document_fn=lambda _q: "",
        search_document_evidence_fn=None,
        read_document_fn=None,
        document_name="d",
        project_name="p",
        scope_summary="s",
    )
    assert out == "session"
    assert captured["llm"] == "llm"
    assert captured["document_name"] == "d"


def test_save_output_delegates(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "agent.adapters.archive.save_agent_output", lambda **kwargs: captured.update(kwargs)
    )
    save_output(uuid="u1", project_uid="p1", session_uid="s1", output_type="text", content="c")
    assert captured["uuid"] == "u1"


def test_project_store_adapters_delegate(monkeypatch):
    monkeypatch.setattr(
        "agent.adapters.project_store.list_projects",
        lambda uuid, include_archived=False: [
            {"project_uid": "p1", "uuid": uuid, "archived": include_archived}
        ],
    )
    monkeypatch.setattr(
        "agent.adapters.project_store.list_project_files",
        lambda **kwargs: [kwargs],
    )
    monkeypatch.setattr(
        "agent.adapters.project_store.list_project_sessions",
        lambda **kwargs: [kwargs],
    )
    monkeypatch.setattr(
        "agent.adapters.project_store.list_project_session_messages",
        lambda **kwargs: [{"role": "user", "content": "q", **kwargs}],
    )
    default_calls = {}
    create_calls = {}
    update_calls = {}
    delete_calls = {}
    save_calls = {}
    monkeypatch.setattr(
        "agent.adapters.project_store.ensure_default_project_session",
        lambda **kwargs: default_calls.update(kwargs),
    )
    monkeypatch.setattr(
        "agent.adapters.project_store.create_project_session",
        lambda **kwargs: create_calls.update(kwargs) or {"session_uid": "s2"},
    )
    monkeypatch.setattr(
        "agent.adapters.project_store.update_project_session",
        lambda **kwargs: update_calls.update(kwargs),
    )
    monkeypatch.setattr(
        "agent.adapters.project_store.delete_project_session",
        lambda **kwargs: delete_calls.update(kwargs),
    )
    monkeypatch.setattr(
        "agent.adapters.project_store.save_project_session_messages",
        lambda **kwargs: save_calls.update(kwargs),
    )

    assert list_user_projects(uuid="u1") == [{"project_uid": "p1", "uuid": "u1", "archived": False}]
    assert list_project_files_for_user(project_uid="p1", uuid="u1")[0]["project_uid"] == "p1"
    assert list_sessions_for_project(project_uid="p1", uuid="u1")[0]["project_uid"] == "p1"
    assert (
        list_session_messages_for_project(session_uid="s1", project_uid="p1", uuid="u1")[0][
            "session_uid"
        ]
        == "s1"
    )
    ensure_default_session_for_project(project_uid="p1", uuid="u1")
    create_session_for_project(project_uid="p1", uuid="u1", session_name="会话")
    update_session_for_project(
        session_uid="s1",
        project_uid="p1",
        uuid="u1",
        session_name="new",
        is_pinned=1,
    )
    delete_session_for_project(session_uid="s1", project_uid="p1", uuid="u1")
    save_session_messages_for_project(
        session_uid="s1",
        project_uid="p1",
        uuid="u1",
        messages=[{"role": "assistant", "content": "a"}],
    )

    assert default_calls["project_uid"] == "p1"
    assert create_calls["session_name"] == "会话"
    assert update_calls["is_pinned"] == 1
    assert delete_calls["session_uid"] == "s1"
    assert isinstance(save_calls["messages"], list)


def test_user_settings_adapters_delegate(monkeypatch):
    monkeypatch.setattr("agent.adapters.user_settings.get_user_api_key", lambda: "k")
    monkeypatch.setattr("agent.adapters.user_settings.get_user_model_name", lambda: "m")
    monkeypatch.setattr("agent.adapters.user_settings.get_user_base_url", lambda: "u")
    monkeypatch.setattr(
        "agent.adapters.user_settings.get_user_files", lambda uuid: [{"uuid": uuid}]
    )
    monkeypatch.setattr("agent.adapters.user_settings.get_api_key", lambda uuid: f"k:{uuid}")
    monkeypatch.setattr("agent.adapters.user_settings.get_model_name", lambda uuid: f"m:{uuid}")
    monkeypatch.setattr("agent.adapters.user_settings.get_base_url", lambda uuid: f"b:{uuid}")
    monkeypatch.setattr(
        "agent.adapters.user_settings.get_policy_router_model_name",
        lambda uuid: f"pm:{uuid}",
    )
    monkeypatch.setattr(
        "agent.adapters.user_settings.get_policy_router_base_url",
        lambda uuid: f"pb:{uuid}",
    )
    monkeypatch.setattr(
        "agent.adapters.user_settings.get_policy_router_api_key",
        lambda uuid: f"pk:{uuid}",
    )
    monkeypatch.setattr(
        "agent.adapters.user_settings.get_runtime_tuning_settings",
        lambda uuid: {
            "agent_policy_async_enabled": True,
            "agent_policy_async_refresh_seconds": 1.5,
            "agent_policy_async_min_confidence": None,
            "agent_policy_async_max_staleness_seconds": None,
            "rag_index_batch_size": None,
            "agent_document_text_cache_max_chars": None,
            "local_rag_project_max_chars": None,
            "local_rag_project_max_chunks": None,
        },
    )
    save_calls = {}
    monkeypatch.setattr(
        "agent.adapters.user_settings.save_api_key",
        lambda uuid, value: save_calls.__setitem__("api_key", (uuid, value)),
    )
    monkeypatch.setattr(
        "agent.adapters.user_settings.save_model_name",
        lambda uuid, value: save_calls.__setitem__("model_name", (uuid, value)),
    )
    monkeypatch.setattr(
        "agent.adapters.user_settings.save_base_url",
        lambda uuid, value: save_calls.__setitem__("base_url", (uuid, value)),
    )
    monkeypatch.setattr(
        "agent.adapters.user_settings.save_policy_router_model_name",
        lambda uuid, value: save_calls.__setitem__("policy_model", (uuid, value)),
    )
    monkeypatch.setattr(
        "agent.adapters.user_settings.save_policy_router_base_url",
        lambda uuid, value: save_calls.__setitem__("policy_base_url", (uuid, value)),
    )
    monkeypatch.setattr(
        "agent.adapters.user_settings.save_policy_router_api_key",
        lambda uuid, value: save_calls.__setitem__("policy_api_key", (uuid, value)),
    )
    monkeypatch.setattr(
        "agent.adapters.user_settings.save_runtime_tuning_settings",
        lambda uuid, **kwargs: save_calls.__setitem__("runtime", (uuid, kwargs)),
    )

    assert read_user_api_key() == "k"
    assert read_user_model_name() == "m"
    assert read_user_base_url() == "u"
    assert list_user_files(uuid="u1")[0]["uuid"] == "u1"
    assert read_api_key_for_user(uuid="u1") == "k:u1"
    assert read_model_name_for_user(uuid="u1") == "m:u1"
    assert read_base_url_for_user(uuid="u1") == "b:u1"
    assert read_policy_router_model_name_for_user(uuid="u1") == "pm:u1"
    assert read_policy_router_base_url_for_user(uuid="u1") == "pb:u1"
    assert read_policy_router_api_key_for_user(uuid="u1") == "pk:u1"
    assert read_runtime_tuning_settings_for_user(uuid="u1")["agent_policy_async_enabled"]

    save_api_key_for_user(uuid="u1", api_key="a")
    save_model_name_for_user(uuid="u1", model_name="m1")
    save_base_url_for_user(uuid="u1", base_url="https://x")
    save_policy_router_model_name_for_user(uuid="u1", model_name="pm1")
    save_policy_router_base_url_for_user(uuid="u1", base_url="https://p")
    save_policy_router_api_key_for_user(uuid="u1", api_key="pk1")
    save_runtime_tuning_settings_for_user(
        "u1",
        agent_policy_async_enabled=True,
        agent_policy_async_refresh_seconds=1.0,
        agent_policy_async_min_confidence=0.7,
        agent_policy_async_max_staleness_seconds=10.0,
        rag_index_batch_size=64,
        agent_document_text_cache_max_chars=1000,
        local_rag_project_max_chars=5000,
        local_rag_project_max_chunks=200,
    )
    applied = apply_runtime_tuning_env_for_user(uuid="u1")
    assert applied["AGENT_POLICY_ASYNC_ENABLED"] == "true"
    assert save_calls["api_key"] == ("u1", "a")
    assert save_calls["model_name"] == ("u1", "m1")
    assert save_calls["base_url"] == ("u1", "https://x")
    assert save_calls["policy_model"] == ("u1", "pm1")
    assert save_calls["policy_base_url"] == ("u1", "https://p")
    assert save_calls["policy_api_key"] == ("u1", "pk1")
    assert save_calls["runtime"][0] == "u1"
