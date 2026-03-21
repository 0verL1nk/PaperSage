from pydantic import BaseModel, Field

from agent import paper_agent as paper_agent_module


def test_build_system_prompt_does_not_raise():
    result = paper_agent_module._build_system_prompt(
        document_name="测试文档",
        project_name="测试项目",
        scope_summary="测试范围",
    )
    assert "你是专业论文问答 Agent" in result
    assert "测试文档" in result
    assert "测试项目" in result
    assert "<mindmap>{" in result


def test_build_system_prompt_explicitly_blocks_external_search_for_project_only_queries():
    result = paper_agent_module._build_system_prompt(
        document_name="测试文档",
        project_name="测试项目",
        scope_summary="仅限项目内 4 篇论文",
    )

    assert "当前项目文档范围内" in result
    assert "不要调用 search_papers 或 search_web" in result


def test_build_system_prompt_for_document_free_session_omits_project_document_instructions():
    result = paper_agent_module._build_system_prompt(
        project_name="测试项目",
        scope_summary="仅允许外部检索，不提供项目文档",
        document_access="none",
    )

    assert "当前会话不提供项目文档" in result
    assert "不要调用 search_document" in result
    assert "必须使用 search_document" not in result
    assert "当前对话文档（兼容字段）" not in result


def test_create_paper_agent_session_uses_runtime_agent_builder(monkeypatch):
    captured = {}

    def fake_create_runtime_agent(*, model, tools, system_prompt, **kwargs):
        captured["model"] = model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        return {"name": "fake-agent"}

    monkeypatch.setattr(
        paper_agent_module,
        "build_runtime_tools",
        lambda **_kwargs: ["tool-a", "tool-b"],
    )
    monkeypatch.setattr(
        paper_agent_module,
        "create_runtime_agent",
        fake_create_runtime_agent,
    )

    session = paper_agent_module.create_paper_agent_session(
        llm="fake-llm",
        search_document_fn=lambda q: f"answer:{q}",
    )

    assert session.agent == {"name": "fake-agent"}
    assert session.thread_id.startswith("paper-qa-")
    assert captured["model"] == "fake-llm"
    assert captured["tools"] == ["tool-a", "tool-b"]
    assert "未知文档" in captured["system_prompt"]


def test_paper_agent_session_runtime_config_contains_thread_id():
    session = paper_agent_module.PaperAgentSession(
        agent=object(),
        thread_id="thread-1",
        tool_specs=[],
    )
    assert session.runtime_config == {"configurable": {"thread_id": "thread-1"}}


def test_create_paper_agent_session_tool_specs_default_manifest(monkeypatch):
    class _FakeInput(BaseModel):
        query: str = Field(description="q")

    class _FakeTool:
        name = "search_document"
        description = "desc"
        args_schema = _FakeInput

    monkeypatch.delenv("AGENT_TOOL_SCHEMA_LEVEL", raising=False)
    monkeypatch.setattr(
        paper_agent_module,
        "build_runtime_tools",
        lambda **_kwargs: [_FakeTool()],
    )
    monkeypatch.setattr(
        paper_agent_module,
        "create_runtime_agent",
        lambda **_kwargs: {"name": "fake-agent"},
    )

    session = paper_agent_module.create_paper_agent_session(
        llm="fake-llm",
        search_document_fn=lambda q: q,
    )

    assert session.tool_specs
    assert session.tool_specs[0]["name"] == "search_document"
    assert session.tool_specs[0]["args_schema"] == ""
    assert session.tool_specs[0]["schema_level"] == "manifest"


def test_create_paper_agent_session_tool_specs_full_schema(monkeypatch):
    class _FakeInput(BaseModel):
        query: str = Field(description="q")

    class _FakeTool:
        name = "search_document"
        description = "desc"
        args_schema = _FakeInput

    monkeypatch.setenv("AGENT_TOOL_SCHEMA_LEVEL", "full")
    monkeypatch.setattr(
        paper_agent_module,
        "build_runtime_tools",
        lambda **_kwargs: [_FakeTool()],
    )
    monkeypatch.setattr(
        paper_agent_module,
        "create_runtime_agent",
        lambda **_kwargs: {"name": "fake-agent"},
    )

    session = paper_agent_module.create_paper_agent_session(
        llm="fake-llm",
        search_document_fn=lambda q: q,
    )

    assert session.tool_specs
    assert '"properties"' in session.tool_specs[0]["args_schema"]
    assert session.tool_specs[0]["schema_level"] == "full"


def test_create_paper_agent_session_can_disable_document_tools(monkeypatch):
    monkeypatch.delenv("AGENT_TOOL_SCHEMA_LEVEL", raising=False)
    monkeypatch.setattr(
        paper_agent_module,
        "create_runtime_agent",
        lambda **_kwargs: {"name": "fake-agent"},
    )

    session = paper_agent_module.create_paper_agent_session(
        llm="fake-llm",
        search_document_fn=None,
        search_document_evidence_fn=None,
        read_document_fn=None,
        list_documents_fn=None,
        document_access="none",
        scope_summary="仅允许外部检索，不提供项目文档",
    )

    tool_names = {item["name"] for item in session.tool_specs}
    assert "search_document" not in tool_names
    assert "read_document" not in tool_names
    assert "list_document" not in tool_names
    assert "search_web" in tool_names
