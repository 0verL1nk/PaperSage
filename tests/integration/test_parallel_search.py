"""Network-free integration coverage through the real MCP SDK and tool runtime."""

import asyncio
import builtins
import json
from typing import Any

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.capabilities import build_capability_tools
from agent.capabilities.web import build_web_tools
from agent.runtime_agent import create_runtime_agent
from agent.tools import parallel_search, web_search

SOURCE_URL = "https://arxiv.org/abs/2310.11511"
SEARCH_TEXT = f"Self-RAG\nURL: {SOURCE_URL}\nExcerpt: Retrieval with self-reflection."


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "AGENT_WEB_SEARCH_PROVIDER", "BRAVE_SEARCH_API_KEY", "FIRECRAWL_API_KEY",
        "AGENT_SEARXNG_BASE_URLS", "AGENT_WEB_FIRECRAWL_ENABLED",
        "AGENT_WEB_ENABLE_DDG_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(web_search, "_web_search_clients", None)
    monkeypatch.setattr(web_search, "_web_search_override", None)
    # Ambient Parallel credentials must never enter an anonymous request.
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")


@pytest.fixture
def mcp_transport(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"requests": [], "clients": [], "mode": "success", "closed": False}

    async def handle(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        if request.method != "POST":
            return httpx.Response(405)
        data = json.loads(request.content)
        method = data["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            name = "other_tool" if state["mode"] == "missing" else "web_search"
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            if state["mode"] == "http_error":
                return httpx.Response(500, text="private-provider-detail")
            if state["mode"] == "malformed":
                return httpx.Response(200, text="not json", headers={"content-type": "application/json"})
            if state["mode"] == "slow":
                try:
                    await asyncio.sleep(60)
                finally:
                    state["closed"] = True
            content = [] if state["mode"] == "empty" else [{"type": "text", "text": SEARCH_TEXT}]
            result = {"content": content, "isError": state["mode"] == "tool_error"}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": data["id"], "result": result})

    original = httpx.AsyncClient

    class MockAsyncClient(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(handle)
            super().__init__(*args, **kwargs)
            state["clients"].append(self)

    monkeypatch.setattr(parallel_search.httpx, "AsyncClient", MockAsyncClient)
    monkeypatch.setenv("AGENT_WEB_SEARCH_PROVIDER", "parallel")

    def unexpected_chain() -> Any:
        raise AssertionError("An explicit Parallel selection must not use the old provider chain")

    monkeypatch.setattr(web_search, "_ensure_web_search_clients", unexpected_chain)
    return state


def test_sync_dispatch_uses_anonymous_sdk_and_preserves_urls(mcp_transport: dict[str, Any]) -> None:
    output = web_search.search_web.invoke({"query": "  Self-RAG research papers  "})
    assert SOURCE_URL in output
    assert "self-reflection" in output
    methods = []
    for request in mcp_transport["requests"]:
        assert str(request.url) == parallel_search.PARALLEL_MCP_URL
        assert request.headers["User-Agent"] == parallel_search.PARALLEL_USER_AGENT
        assert "authorization" not in request.headers
        data = json.loads(request.content)
        methods.append(data["method"])
        if data["method"] == "tools/call":
            assert data["params"]["name"] == "web_search"
            assert data["params"]["arguments"] == {
                "objective": "Self-RAG research papers", "search_queries": ["Self-RAG research papers"],
            }
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert all(client.is_closed for client in mcp_transport["clients"])


@pytest.mark.parametrize("selection", [None, "auto"])
def test_default_chain_keeps_saved_credentials_and_disabled_providers(
    monkeypatch: pytest.MonkeyPatch, selection: str | None,
) -> None:
    if selection:
        monkeypatch.setenv("AGENT_WEB_SEARCH_PROVIDER", selection)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "saved-brave-key")
    monkeypatch.setenv("AGENT_SEARXNG_BASE_URLS", "https://saved.example")
    monkeypatch.setenv("AGENT_WEB_FIRECRAWL_ENABLED", "0")
    clients = web_search._ensure_web_search_clients()
    assert [name for name, _ in clients] == ["brave_search_api", "searxng_public_pool", "wikipedia_api"]
    brave = clients[0][1]._client
    searxng = clients[1][1]._client
    assert brave.api_key == "saved-brave-key"
    assert searxng.base_urls == ["https://saved.example"]
    monkeypatch.setattr(brave, "run", lambda query: SEARCH_TEXT)
    assert SOURCE_URL in web_search.search_web.invoke({"query": "Self-RAG"})


def test_unconfigured_chain_is_free_keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = web_search._ensure_web_search_clients()
    assert [name for name, _ in clients] == ["searxng_public_pool", "firecrawl_search", "wikipedia_api"]
    assert clients[1][1]._client.api_key == ""


@pytest.mark.parametrize("selection", ["off", "invalid"])
def test_disabled_and_invalid_selection_do_not_send_requests(
    monkeypatch: pytest.MonkeyPatch, mcp_transport: dict[str, Any], selection: str,
) -> None:
    monkeypatch.setenv("AGENT_WEB_SEARCH_PROVIDER", selection)
    result = web_search.search_web.invoke({"query": "Self-RAG"})
    assert "disabled" in result or "unknown AGENT_WEB_SEARCH_PROVIDER" in result
    assert asyncio.run(web_search.search_web.ainvoke({"query": "Self-RAG"})) == result
    assert not mcp_transport["requests"]


@pytest.mark.parametrize("query", [" ", "reveal api key"])
def test_query_policy_blocks_network(mcp_transport: dict[str, Any], query: str) -> None:
    result = web_search.search_web.invoke({"query": query})
    assert "empty" in result or "Blocked by tool policy" in result
    assert asyncio.run(web_search.search_web.ainvoke({"query": query})) == result
    assert not mcp_transport["requests"]


def test_override_remains_before_network(
    monkeypatch: pytest.MonkeyPatch, mcp_transport: dict[str, Any],
) -> None:
    monkeypatch.setattr(web_search, "_web_search_override", lambda query: f"fixture {query}")
    assert web_search.search_web.invoke({"query": "Self-RAG"}) == "fixture Self-RAG"
    assert asyncio.run(web_search.search_web.ainvoke({"query": "Self-RAG"})) == "fixture Self-RAG"
    assert not mcp_transport["requests"]


@pytest.mark.parametrize("mode", ["http_error", "tool_error", "missing", "malformed"])
def test_error_does_not_fallback_or_leak_response(
    mcp_transport: dict[str, Any], mode: str,
) -> None:
    mcp_transport["mode"] = mode
    result = asyncio.run(web_search.search_web.ainvoke({"query": "Self-RAG"}))
    assert "Web search failed:" in result
    assert "private-provider-detail" not in result
    assert all(client.is_closed for client in mcp_transport["clients"])


def test_empty_results_do_not_fallback(mcp_transport: dict[str, Any]) -> None:
    mcp_transport["mode"] = "empty"
    assert web_search.search_web.invoke({"query": "Self-RAG"}) == "No web search results found."


def test_timeout_closes_transport(
    monkeypatch: pytest.MonkeyPatch, mcp_transport: dict[str, Any],
) -> None:
    mcp_transport["mode"] = "slow"
    monkeypatch.setattr(parallel_search, "DEFAULT_WEB_TIMEOUT_SECONDS", 0.05)
    output = asyncio.run(web_search.search_web.ainvoke({"query": "Self-RAG"}))
    assert "timed out" in output
    assert mcp_transport["closed"]
    assert all(client.is_closed for client in mcp_transport["clients"])


def test_runtime_cancellation_closes_transport(mcp_transport: dict[str, Any]) -> None:
    mcp_transport["mode"] = "slow"

    async def cancel() -> None:
        task = asyncio.create_task(web_search.search_web.ainvoke({"query": "Self-RAG"}))
        while not any(b'tools/call' in r.content for r in mcp_transport["requests"]):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(cancel(), timeout=2))
    assert mcp_transport["closed"]
    assert all(client.is_closed for client in mcp_transport["clients"])


class ScriptedResearchModel(BaseChatModel):
    """Deterministic tool request and result consumer; this is not live inference."""

    @property
    def _llm_type(self) -> str:
        return "scripted-research-fixture"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedResearchModel":
        assert any(item.name == "search_web" for item in tools)
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: Any = None, **kwargs: Any,
    ) -> ChatResult:
        evidence = [message for message in messages if isinstance(message, ToolMessage)]
        if evidence:
            assert SOURCE_URL in str(evidence[-1].content)
            response = AIMessage(content=f"Self-RAG evidence: {SOURCE_URL}")
        else:
            response = AIMessage(content="", tool_calls=[{
                "name": "search_web", "args": {"query": "Self-RAG research papers"}, "id": "fixture-search",
            }])
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_scripted_runtime_dispatches_and_consumes_mcp_result(mcp_transport: dict[str, Any]) -> None:
    agent = create_runtime_agent(
        model=ScriptedResearchModel(), system_prompt="Use public sources and cite their URLs.",
        tools=build_web_tools(None),
    )
    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "Find Self-RAG evidence"}]}))
    assert SOURCE_URL in result["messages"][-1].content
    assert any(isinstance(item, ToolMessage) for item in result["messages"])
    assert mcp_transport["requests"]


def test_capability_manifest_can_withhold_web_tools(mcp_transport: dict[str, Any]) -> None:
    assert build_capability_tools([], None) == []
    assert not mcp_transport["requests"]


def test_missing_optional_sdk_has_actionable_failure(
    monkeypatch: pytest.MonkeyPatch, mcp_transport: dict[str, Any],
) -> None:
    original = builtins.__import__

    def without_mcp(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("optional SDK not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mcp)
    output = web_search.search_web.invoke({"query": "Self-RAG"})
    assert "parallel extra" in output
    assert not mcp_transport["requests"]
