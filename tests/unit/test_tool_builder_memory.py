import json

from agent.tools.builder import build_agent_tools
from agent.tools.memory import build_query_memory_tool, build_write_memory_tool


def test_build_agent_tools_includes_write_memory_tool_when_available() -> None:
    tools = build_agent_tools(
        search_document_fn=lambda _q: "doc",
        write_memory_fn=lambda _items: [{"action": "ADD"}],
    )

    names = [str(getattr(item, "name", "") or "") for item in tools]
    assert "write_memory" in names


def test_build_agent_tools_includes_query_memory_tool_when_available() -> None:
    tools = build_agent_tools(
        search_document_fn=lambda _q: "doc",
        query_memory_fn=lambda _query, _memory_type=None, _limit=5: [],
    )

    names = [str(getattr(item, "name", "") or "") for item in tools]
    assert "query_memory" in names


def test_query_memory_tool_normalizes_args_and_returns_json() -> None:
    captured: dict[str, object] = {}

    def _query_memory(query: str, memory_type: str | None, limit: int) -> list[dict[str, object]]:
        captured["query"] = query
        captured["memory_type"] = memory_type
        captured["limit"] = limit
        return [{"canonical_text": "user prefers concise answers"}]

    tool = build_query_memory_tool(_query_memory)
    result = tool.invoke({"query": "  concise replies  ", "memory_type": "", "limit": 0})

    assert captured == {
        "query": "concise replies",
        "memory_type": None,
        "limit": 1,
    }
    assert json.loads(result) == {
        "results": [{"canonical_text": "user prefers concise answers"}]
    }


def test_write_memory_tool_serializes_items_and_returns_json() -> None:
    captured: dict[str, object] = {}

    def _write_memory(items: list[dict[str, object]]) -> list[dict[str, object]]:
        captured["items"] = items
        return [{"action": "ADD", "canonical_text": items[0]["canonical_text"]}]

    tool = build_write_memory_tool(_write_memory)
    result = tool.invoke(
        {
            "items": [
                {
                    "memory_type": "user_memory",
                    "content": "user prefers concise answers",
                    "canonical_text": "user prefers concise answers",
                    "title": "Answer preference",
                }
            ]
        }
    )

    assert captured["items"] == [
        {
            "memory_type": "user_memory",
            "content": "user prefers concise answers",
            "canonical_text": "user prefers concise answers",
            "title": "Answer preference",
            "dedup_key": "",
            "action": "ADD",
            "confidence": 0.9,
        }
    ]
    assert json.loads(result) == {
        "results": [
            {
                "action": "ADD",
                "canonical_text": "user prefers concise answers",
            }
        ]
    }
