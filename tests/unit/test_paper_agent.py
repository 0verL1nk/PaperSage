from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from agent import paper_agent as paper_agent_module


def test_create_paper_agent_session_uses_checkpointer(monkeypatch):
    captured = {}

    def fake_create_agent(*, model, tools, system_prompt, checkpointer):
        captured["model"] = model
        captured["tools"] = tools
        captured["system_prompt"] = system_prompt
        captured["checkpointer"] = checkpointer
        return {"name": "fake-agent"}

    monkeypatch.setattr(paper_agent_module, "create_agent", fake_create_agent)
    monkeypatch.setattr(
        paper_agent_module,
        "build_agent_tools",
        lambda search_document_fn, search_document_evidence_fn=None, read_document_fn=None, list_documents_fn=None: ["tool-a", "tool-b"],
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
    assert isinstance(captured["checkpointer"], InMemorySaver)


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
        "build_agent_tools",
        lambda search_document_fn, search_document_evidence_fn=None, read_document_fn=None, list_documents_fn=None: [_FakeTool()],
    )
    monkeypatch.setattr(
        paper_agent_module,
        "create_agent",
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
        "build_agent_tools",
        lambda search_document_fn, search_document_evidence_fn=None, read_document_fn=None, list_documents_fn=None: [_FakeTool()],
    )
    monkeypatch.setattr(
        paper_agent_module,
        "create_agent",
        lambda **_kwargs: {"name": "fake-agent"},
    )

    session = paper_agent_module.create_paper_agent_session(
        llm="fake-llm",
        search_document_fn=lambda q: q,
    )

    assert session.tool_specs
    assert "\"properties\"" in session.tool_specs[0]["args_schema"]
    assert session.tool_specs[0]["schema_level"] == "full"
