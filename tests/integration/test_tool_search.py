import json

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage

from agent.capabilities import build_agent_tools, build_progressive_tool_middleware


@pytest.mark.integration
def test_tool_search_end_to_end():
    # 1. 模拟构建全量工具
    tools = build_agent_tools(
        search_document_fn=lambda q: "doc mock",
        list_documents_fn=lambda: [],
        read_document_fn=lambda o, limit: ("doc", 1),
    )

    middleware = build_progressive_tool_middleware(tools)
    assert len(middleware) > 0, "Progressive middleware not enabled"
    progressive_md = middleware[0]

    # 2. 确保初始上下文只有 Search Tool
    request = ModelRequest(
        model=object(),  # mock model
        messages=[HumanMessage(content="你好，我需要查一下这篇论文里关于 RAG 的细节。")],
        tools=tools,
    )

    visible_tools = []

    def mock_handler(req):
        nonlocal visible_tools
        visible_tools = [t.name if not isinstance(t, dict) else t.get("name") for t in req.tools]
        return None

    progressive_md.wrap_model_call(request, mock_handler)

    assert "search_tools" in visible_tools
    # start_plan is lazy now, check if it's in initial context
    assert "start_plan" not in visible_tools, "Lazy tool leaked into initial context!"

    # 3. 验证 Tool Search 本身的语义检索召回能力
    search_tool_obj = next((t for t in tools if getattr(t, "name", "") == "search_tools"), None)
    assert search_tool_obj is not None

    # 查找做计划或任务拆解的工具
    result_json = search_tool_obj.invoke(
        {"query": "帮我制定一个详细的任务计划并逐步拆解", "reason": "需要规划"}
    )

    data = json.loads(result_json)
    assert data["type"] == "tool_search_result"

    # "search_document" 的 description 是: "Search uploaded paper content for relevant evidence snippets using RAG."
    # 通过正则/BM25/Dense 应该能精准召回
    found_names = [t["tool_name"] for t in data["tools"]]
    assert "start_plan" in found_names
    print(f"\\n[Search Tools Output]: {found_names}")
