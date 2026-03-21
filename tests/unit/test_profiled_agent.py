import agent.profiled_agent as profiled_agent_module
from agent.profiles import paper_leader_profile, paper_worker_profile


def test_create_profiled_agent_session_resolves_string_profile_and_delegates(monkeypatch):
    captured = {}

    def fake_create_agent_session(*, profile, deps, options):
        captured["profile"] = profile
        captured["deps"] = deps
        captured["options"] = options
        return profiled_agent_module.AgentSession(
            agent="agent",
            thread_id="thread-1",
            tool_specs=[],
            profile_name=profile.name,
        )

    monkeypatch.setattr(profiled_agent_module, "create_agent_session", fake_create_agent_session)

    session = profiled_agent_module.create_profiled_agent_session(
        profile="leader",
        llm="fake-llm",
        search_document_fn=lambda q: q,
        document_name="文档A",
        project_name="项目A",
        scope_summary="范围A",
    )

    assert session.thread_id == "thread-1"
    assert session.profile_name == "paper_leader"
    assert captured["profile"] is paper_leader_profile
    assert captured["deps"].search_document_fn("q") == "q"
    assert captured["options"].llm == "fake-llm"
    assert captured["options"].document_name == "文档A"


def test_create_profiled_agent_session_accepts_profile_object(monkeypatch):
    captured = {}

    def fake_create_agent_session(*, profile, deps, options):
        captured["profile"] = profile
        captured["options"] = options
        return profiled_agent_module.AgentSession(
            agent="agent",
            thread_id="thread-2",
            tool_specs=[],
            profile_name=profile.name,
        )

    monkeypatch.setattr(profiled_agent_module, "create_agent_session", fake_create_agent_session)

    session = profiled_agent_module.create_profiled_agent_session(
        profile=paper_worker_profile,
        llm="fake-llm",
        search_document_fn=lambda q: q,
        thread_id="team:thread-1",
    )

    assert session.thread_id == "thread-2"
    assert captured["profile"] is paper_worker_profile
    assert captured["options"].thread_id == "team:thread-1"
