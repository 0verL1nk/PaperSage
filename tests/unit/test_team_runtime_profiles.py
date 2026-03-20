from types import SimpleNamespace

from agent.profiles import paper_worker_profile
from agent.session_factory import AgentDependencies
from agent.team.runtime import TeamRuntime


def test_team_runtime_spawn_agent_uses_profile_session_factory(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_agent_session(*, profile, deps, options):
        captured["profile"] = profile
        captured["deps"] = deps
        captured["options"] = options
        return SimpleNamespace(
            agent={"kind": "worker-runtime"},
            thread_id=options.thread_id,
            tool_specs=[],
            profile_name=profile.name,
        )

    monkeypatch.setattr("agent.session_factory.create_agent_session", fake_create_agent_session)

    runtime = TeamRuntime("team-1")
    deps = AgentDependencies(search_document_fn=lambda query: query)

    agent_id = runtime.spawn_agent(
        name="researcher",
        model="fake-llm",
        system_prompt="WORKER_OVERRIDE",
        tools=[],
        profile=paper_worker_profile,
        deps=deps,
    )

    instance = runtime.agents[agent_id]
    assert captured["profile"] is paper_worker_profile
    assert captured["deps"] is deps
    assert captured["options"].llm == "fake-llm"
    assert captured["options"].system_prompt == "WORKER_OVERRIDE"
    assert captured["options"].thread_id == f"team:team-1:{agent_id}"
    assert instance.agent == {"kind": "worker-runtime"}
    assert instance.profile_name == "paper_worker"
    assert instance.thread_id == f"team:team-1:{agent_id}"

    runtime.cleanup()
