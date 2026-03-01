from agent import capabilities


def _tool_map(tools):
    return {tool.name: tool for tool in tools}


def test_build_agent_tools_exposes_structured_tools(monkeypatch):
    class _FakeWebSearch:
        def __init__(self, name):
            self.name = name

        def run(self, query):
            return f"web:{query}"

    monkeypatch.setattr(capabilities, "DuckDuckGoSearchRun", _FakeWebSearch)

    tools = capabilities.build_agent_tools(lambda query: f"doc:{query}")
    names = sorted(tool.name for tool in tools)
    assert names == ["search_document", "search_web", "use_skill"]

    tool_map = _tool_map(tools)
    assert tool_map["search_document"].invoke({"query": "q1"}) == "doc:q1"
    assert tool_map["search_web"].invoke({"query": "q2"}) == "web:q2"


def test_use_skill_returns_known_and_unknown_guidance(monkeypatch):
    monkeypatch.setattr(
        capabilities,
        "DuckDuckGoSearchRun",
        lambda name: type("Web", (), {"run": lambda self, q: "ok"})(),
    )
    tools = capabilities.build_agent_tools(lambda query: query)
    tool_map = _tool_map(tools)

    known = tool_map["use_skill"].invoke({"skill_name": "summary", "task": "t"})
    unknown = tool_map["use_skill"].invoke({"skill_name": "unknown", "task": "t"})

    assert "Skill: summary" in known
    assert "Unknown skill 'unknown'" in unknown


def test_search_web_fallback_when_ddg_unavailable(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("ddg unavailable")

    monkeypatch.setattr(capabilities, "DuckDuckGoSearchRun", _raise)
    tools = capabilities.build_agent_tools(lambda query: query)
    tool_map = _tool_map(tools)

    result = tool_map["search_web"].invoke({"query": "x"})
    assert "Web search is unavailable" in result
