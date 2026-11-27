import json
from types import SimpleNamespace
from typing import Any

from agent.tools import web_search, web_search_youcom
from agent.tools.utils import DEFAULT_WEB_MAX_RESULTS


def _reset_clients(monkeypatch) -> None:
    monkeypatch.setattr(web_search, "_web_search_clients", None)


def _sse_response(frames: list[dict]) -> SimpleNamespace:
    text = "\n".join(f"event: message\ndata: {json.dumps(frame)}" for frame in frames)
    return SimpleNamespace(status_code=200, headers={}, text=text)


def _json_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(status_code=200, headers={}, text=json.dumps(payload))


def _youcom_result_frame(web_items: list[dict], is_error: bool = False) -> dict:
    payload = {"results": {"web": web_items}}
    result: dict = {"content": [{"type": "text", "text": json.dumps(payload)}]}
    if is_error:
        result["isError"] = True
    return {"result": result, "jsonrpc": "2.0", "id": 1}


NOTIFICATION_FRAME = {
    "jsonrpc": "2.0",
    "method": "notifications/message",
    "params": {"level": "info", "data": "Search successful"},
}

WEB_ITEMS = [
    {
        "title": "Self-RAG paper",
        "url": "https://arxiv.org/abs/2310.11511",
        "description": "Self-reflective retrieval-augmented generation.",
    }
]


def _build_client(monkeypatch, *, api_key: str = "") -> Any:
    monkeypatch.setenv("AGENT_WEB_YOUCOM_ENABLED", "1")
    monkeypatch.setattr(web_search_youcom, "_load_secret", lambda name: api_key)
    return web_search_youcom._build_youcom_web_search_client()


def test_youcom_client_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_WEB_YOUCOM_ENABLED", raising=False)

    assert web_search_youcom._build_youcom_web_search_client() is None


def test_youcom_client_enabled_keyless_without_key(monkeypatch):
    client = _build_client(monkeypatch)

    assert client is not None
    assert client.api_key == ""
    assert client.search_url == web_search_youcom.YOUCOM_KEYLESS_MCP_URL


def test_youcom_client_authenticated_url_when_key_configured(monkeypatch):
    client = _build_client(monkeypatch, api_key="ydc-test-key")

    assert client is not None
    assert client.api_key == "ydc-test-key"
    assert client.search_url == web_search_youcom.YOUCOM_AUTHENTICATED_MCP_URL


def test_youcom_client_honors_endpoint_override(monkeypatch):
    monkeypatch.setenv("AGENT_YOUCOM_MCP_URL", "http://localhost:9/v2/mcp")

    client = _build_client(monkeypatch)

    assert client.search_url == "http://localhost:9/v2/mcp"


def test_youcom_run_sends_stateless_tools_call_and_parses_sse(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return _sse_response([NOTIFICATION_FRAME, _youcom_result_frame(WEB_ITEMS)])

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch)

    rendered = client.run("Self-RAG")

    assert captured["url"] == web_search_youcom.YOUCOM_KEYLESS_MCP_URL
    body = captured["json"]
    assert body["method"] == "tools/call"
    assert body["params"]["name"] == "you-search"
    assert body["params"]["arguments"]["query"] == "Self-RAG"
    assert body["params"]["arguments"]["count"] == DEFAULT_WEB_MAX_RESULTS
    assert "Authorization" not in captured["headers"]
    assert "Self-RAG paper" in rendered
    assert "https://arxiv.org/abs/2310.11511" in rendered


def test_youcom_run_parses_plain_json_response(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _json_response(_youcom_result_frame(WEB_ITEMS))

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch)

    rendered = client.run("Self-RAG")

    assert "Self-RAG paper" in rendered


def test_youcom_run_sends_bearer_when_key_configured(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, headers=headers)
        return _sse_response([_youcom_result_frame(WEB_ITEMS)])

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch, api_key="ydc-test-key")

    client.run("Self-RAG")

    assert captured["url"] == web_search_youcom.YOUCOM_AUTHENTICATED_MCP_URL
    assert captured["headers"]["Authorization"] == "Bearer ydc-test-key"


def test_youcom_run_falls_back_to_first_highlight_for_snippet(monkeypatch):
    items = [
        {
            "title": "Survey",
            "url": "https://example.org/survey",
            "contents": {"highlights": ["First highlight text."]},
        }
    ]

    def fake_post(url, json=None, headers=None, timeout=None):
        return _sse_response([_youcom_result_frame(items)])

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch)

    rendered = client.run("Self-RAG")

    assert "Survey" in rendered
    assert "First highlight text." in rendered


def test_youcom_run_raises_on_error_status(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return SimpleNamespace(status_code=429, headers={}, text="")

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch)

    try:
        client.run("Self-RAG")
    except RuntimeError as exc:
        assert "youcom status=429" in str(exc)
    else:
        raise AssertionError("Expected youcom error status to raise")


def test_youcom_run_raises_on_jsonrpc_error_frame(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _sse_response(
            [{"jsonrpc": "2.0", "id": 1, "error": {"message": "rate limited"}}]
        )

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch)

    try:
        client.run("Self-RAG")
    except RuntimeError as exc:
        assert "youcom jsonrpc error" in str(exc)
        assert "rate limited" in str(exc)
    else:
        raise AssertionError("Expected youcom jsonrpc error to raise")


def test_youcom_run_raises_on_tool_error_frame(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _sse_response([_youcom_result_frame(WEB_ITEMS, is_error=True)])

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch)

    try:
        client.run("Self-RAG")
    except RuntimeError as exc:
        assert "youcom tool error" in str(exc)
    else:
        raise AssertionError("Expected youcom tool error to raise")


def test_youcom_run_raises_when_no_result_frame(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _sse_response([NOTIFICATION_FRAME])

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch)

    try:
        client.run("Self-RAG")
    except RuntimeError as exc:
        assert "youcom invalid payload" in str(exc)
    else:
        raise AssertionError("Expected youcom invalid payload to raise")


def test_youcom_run_raises_when_result_missing_web_results(monkeypatch):
    frame = {
        "result": {
            "content": [{"type": "text", "text": json.dumps({"unexpected": True})}]
        },
        "jsonrpc": "2.0",
        "id": 1,
    }

    def fake_post(url, json=None, headers=None, timeout=None):
        return _sse_response([frame])

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch)

    try:
        client.run("Self-RAG")
    except RuntimeError as exc:
        assert "youcom invalid payload" in str(exc)
    else:
        raise AssertionError("Expected youcom invalid payload to raise")


def test_youcom_run_returns_no_results_message_for_empty_items(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _sse_response([_youcom_result_frame([])])

    monkeypatch.setattr(web_search_youcom.httpx, "post", fake_post)
    client = _build_client(monkeypatch)

    assert client.run("Self-RAG") == "No web search results found."


def test_provider_chain_includes_youcom_when_enabled(monkeypatch):
    _reset_clients(monkeypatch)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_WEB_YOUCOM_ENABLED", "1")
    monkeypatch.setattr(web_search_youcom, "_load_secret", lambda name: "")

    try:
        clients = web_search._ensure_web_search_clients()
    finally:
        _reset_clients(monkeypatch)

    names = [name for name, _client in clients]
    assert "youcom_search" in names
    assert names.index("firecrawl_search") < names.index("youcom_search")
    assert names.index("youcom_search") < names.index("wikipedia_api")


def test_provider_chain_excludes_youcom_by_default(monkeypatch):
    _reset_clients(monkeypatch)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_WEB_YOUCOM_ENABLED", raising=False)

    try:
        clients = web_search._ensure_web_search_clients()
    finally:
        _reset_clients(monkeypatch)

    names = [name for name, _client in clients]
    assert "youcom_search" not in names
