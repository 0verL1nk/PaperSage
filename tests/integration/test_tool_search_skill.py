import json

import pytest

from agent.capabilities import build_agent_tools


@pytest.mark.integration
def test_tool_search_can_find_skills():
    tools = build_agent_tools(
        search_document_fn=lambda q: "doc mock",
        list_documents_fn=lambda: [],
        read_document_fn=lambda o, limit: ("doc", 1),
    )

    search_tool_obj = next((t for t in tools if getattr(t, "name", "") == "search_tools"), None)
    assert search_tool_obj is not None

    # 我们知道内置有 translation、method_compare 等 skill
    result_json = search_tool_obj.invoke({"query": "帮我把这段话翻译成中文", "reason": "需要翻译"})
    data = json.loads(result_json)

    found_names = [t["tool_name"] for t in data["tools"]]
    print(f"\\n[Search for Translation]: {found_names}")

    # 期望因为我们在 capabilities 里面把 skill 的元数据拼接到了 use_skill，所以应该能召回 use_skill
    assert "use_skill" in found_names
