from pydantic import BaseModel, Field

from agent.profiles import AgentProfile
from agent.session_factory import AgentDependencies, AgentRuntimeOptions, create_agent_session


def test_create_agent_session_uses_profile_prompt_builder_and_runtime_builder(monkeypatch):
    captured = {}

    class _FakeInput(BaseModel):
        query: str = Field(description="q")

    class _FakeTool:
        name = "search_document"
        description = "desc"
        args_schema = _FakeInput

    def fake_prompt_builder(*, document_name=None, project_name=None, scope_summary=None):
        captured["prompt_args"] = {
            "document_name": document_name,
            "project_name": project_name,
            "scope_summary": scope_summary,
        }
        return "PROFILE_PROMPT"

    def fake_create_runtime_agent(*, model, tools, system_prompt, **kwargs):
        captured["runtime_model"] = model
        captured["runtime_tools"] = tools
        captured["runtime_prompt"] = system_prompt
        captured["runtime_kwargs"] = kwargs
        return {"name": "fake-agent"}

    monkeypatch.setattr(
        "agent.session_factory.build_profile_tools",
        lambda *_args, **_kwargs: [_FakeTool()],
    )
    monkeypatch.setattr(
        "agent.session_factory.create_runtime_agent",
        fake_create_runtime_agent,
    )

    profile = AgentProfile(
        name="test_profile",
        description="desc",
        prompt_builder=fake_prompt_builder,
        capability_ids=("document_pack",),
        middleware_ids=(),
    )

    session = create_agent_session(
        profile=profile,
        deps=AgentDependencies(search_document_fn=lambda q: q),
        options=AgentRuntimeOptions(
            llm="fake-llm",
            document_name="文档A",
            project_name="项目A",
            scope_summary="范围A",
        ),
    )

    assert session.agent == {"name": "fake-agent"}
    assert session.thread_id.startswith("paper-qa-")
    assert session.profile_name == "test_profile"
    assert captured["prompt_args"]["document_name"] == "文档A"
    assert captured["runtime_model"] == "fake-llm"
    assert captured["runtime_prompt"] == "PROFILE_PROMPT"
    assert captured["runtime_tools"][0].name == "search_document"



def test_create_agent_session_tool_specs_default_manifest(monkeypatch):
    class _FakeInput(BaseModel):
        query: str = Field(description="q")

    class _FakeTool:
        name = "search_document"
        description = "desc"
        args_schema = _FakeInput

    monkeypatch.delenv("AGENT_TOOL_SCHEMA_LEVEL", raising=False)
    monkeypatch.setattr(
        "agent.session_factory.build_profile_tools",
        lambda *_args, **_kwargs: [_FakeTool()],
    )
    monkeypatch.setattr(
        "agent.session_factory.create_runtime_agent",
        lambda **_kwargs: {"name": "fake-agent"},
    )

    session = create_agent_session(
        profile=AgentProfile(
            name="test_profile",
            description="desc",
            prompt_builder=lambda **_kwargs: "PROMPT",
        ),
        deps=AgentDependencies(search_document_fn=lambda q: q),
        options=AgentRuntimeOptions(llm="fake-llm"),
    )

    assert session.tool_specs
    assert session.tool_specs[0]["name"] == "search_document"
    assert session.tool_specs[0]["args_schema"] == ""
    assert session.tool_specs[0]["schema_level"] == "manifest"


def test_create_agent_session_tool_specs_full_schema(monkeypatch):
    class _FakeInput(BaseModel):
        query: str = Field(description="q")

    class _FakeTool:
        name = "search_document"
        description = "desc"
        args_schema = _FakeInput

    monkeypatch.setenv("AGENT_TOOL_SCHEMA_LEVEL", "full")
    monkeypatch.setattr(
        "agent.session_factory.build_profile_tools",
        lambda *_args, **_kwargs: [_FakeTool()],
    )
    monkeypatch.setattr(
        "agent.session_factory.create_runtime_agent",
        lambda **_kwargs: {"name": "fake-agent"},
    )

    session = create_agent_session(
        profile=AgentProfile(
            name="test_profile",
            description="desc",
            prompt_builder=lambda **_kwargs: "PROMPT",
        ),
        deps=AgentDependencies(search_document_fn=lambda q: q),
        options=AgentRuntimeOptions(llm="fake-llm"),
    )

    assert session.tool_specs
    assert '"properties"' in session.tool_specs[0]["args_schema"]
    assert session.tool_specs[0]["schema_level"] == "full"



def test_create_agent_session_builds_tools_from_profile_capabilities(monkeypatch):
    captured = {}

    def fake_build_profile_tools(profile, deps):
        captured["profile_name"] = profile.name
        captured["deps"] = deps
        return [type("Tool", (), {"name": "search_document", "description": "desc"})()]

    monkeypatch.setattr("agent.session_factory.build_profile_tools", fake_build_profile_tools)
    monkeypatch.setattr(
        "agent.session_factory.create_runtime_agent",
        lambda **kwargs: {"name": "fake-agent", "tools": kwargs["tools"]},
    )

    profile = AgentProfile(
        name="worker_profile",
        description="desc",
        prompt_builder=lambda **_kwargs: "PROMPT",
        capability_ids=("document_pack", "skill_pack"),
    )

    session = create_agent_session(
        profile=profile,
        deps=AgentDependencies(search_document_fn=lambda q: q),
        options=AgentRuntimeOptions(llm="fake-llm"),
    )

    assert captured["profile_name"] == "worker_profile"
    assert session.agent["tools"][0].name == "search_document"


def test_paper_leader_profile_prompt_builder_includes_leader_guidance():
    from agent.profiles import paper_leader_profile

    prompt = paper_leader_profile.prompt_builder(
        document_name="文档A",
        project_name="项目A",
        scope_summary="范围A",
    )

    assert "当前对话项目：项目A" in prompt
    assert "你负责调度与最终回答" in prompt
    assert "你决定是否需要团队分工" in prompt
