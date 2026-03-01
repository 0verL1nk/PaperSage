from langgraph.checkpoint.memory import InMemorySaver

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
        lambda _search_document_fn: ["tool-a", "tool-b"],
    )

    session = paper_agent_module.create_paper_agent_session(
        llm="fake-llm",
        search_document_fn=lambda q: f"answer:{q}",
    )

    assert session.agent == {"name": "fake-agent"}
    assert session.thread_id.startswith("paper-qa-")
    assert captured["model"] == "fake-llm"
    assert captured["tools"] == ["tool-a", "tool-b"]
    assert captured["system_prompt"] == paper_agent_module.PAPER_QA_SYSTEM_PROMPT
    assert isinstance(captured["checkpointer"], InMemorySaver)


def test_paper_agent_session_runtime_config_contains_thread_id():
    session = paper_agent_module.PaperAgentSession(agent=object(), thread_id="thread-1")
    assert session.runtime_config == {"configurable": {"thread_id": "thread-1"}}
