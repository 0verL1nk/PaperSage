from types import SimpleNamespace

from agent.stream import (
    extract_trace_events_from_update,
    extract_result_text,
    extract_stream_text,
    iter_agent_response_deltas,
)


class _FakeAgent:
    def __init__(self, chunks, invoke_result):
        self._chunks = chunks
        self._invoke_result = invoke_result
        self.stream_calls = []
        self.invoke_calls = []

    def stream(self, *args, **kwargs):
        self.stream_calls.append((args, kwargs))
        for item in self._chunks:
            yield item

    def invoke(self, *args, **kwargs):
        self.invoke_calls.append((args, kwargs))
        return self._invoke_result


def test_extract_stream_text_from_message_tuple():
    msg = SimpleNamespace(content="hello")
    assert extract_stream_text((msg, {"node": "model"})) == "hello"


def test_extract_stream_text_from_dict_messages():
    msg = {"messages": [{"content": [{"text": "A"}, {"text": "B"}]}]}
    assert extract_stream_text(msg) == "AB"


def test_extract_result_text_from_ai_message():
    ai_msg = SimpleNamespace(type="ai", content=[{"text": "done"}])
    assert extract_result_text({"messages": [ai_msg]}) == "done"


def test_extract_result_text_from_dict_assistant_message():
    message = {"role": "assistant", "content": [{"text": "dict-ok"}]}
    assert extract_result_text({"messages": [message]}) == "dict-ok"


def test_iter_agent_response_deltas_uses_stream_chunks():
    agent = _FakeAgent(
        chunks=[SimpleNamespace(content="He"), SimpleNamespace(content="llo")],
        invoke_result={"messages": []},
    )
    parts = list(iter_agent_response_deltas(agent, [{"role": "user", "content": "x"}]))

    assert "".join(parts) == "Hello"


def test_iter_agent_response_deltas_fallback_to_invoke():
    ai_msg = SimpleNamespace(type="ai", content="fallback")
    agent = _FakeAgent(chunks=[{}], invoke_result={"messages": [ai_msg]})
    parts = list(iter_agent_response_deltas(agent, [{"role": "user", "content": "x"}]))

    assert "".join(parts) == "fallback"


def test_extract_stream_text_ignores_tool_messages():
    tool_msg = SimpleNamespace(type="tool", content="RAG raw text")
    assert extract_stream_text((tool_msg, {})) == ""


def test_iter_agent_response_deltas_passes_runtime_config():
    config = {"configurable": {"thread_id": "thread-1"}}
    ai_msg = SimpleNamespace(type="ai", content="ok")
    agent = _FakeAgent(chunks=[{}], invoke_result={"messages": [ai_msg]})

    parts = list(
        iter_agent_response_deltas(
            agent,
            [{"role": "user", "content": "x"}],
            config=config,
        )
    )

    assert "".join(parts) == "ok"
    assert agent.stream_calls[0][1]["config"] == config
    assert agent.invoke_calls[0][1]["config"] == config


def test_extract_trace_events_from_update_with_tool_calls_and_tool_result():
    update_payload = {
        "model": {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"name": "search_document", "args": {"query": "abc"}},
                    ],
                }
            ]
        },
        "tools": {
            "messages": [
                {"role": "tool", "name": "search_document", "content": "hit"},
            ]
        },
    }

    events = extract_trace_events_from_update(update_payload)
    assert any(
        item["performative"] == "tool_call" and item["receiver"] == "search_document"
        for item in events
    )
    assert any(
        item["performative"] == "tool_result" and item["sender"] == "search_document"
        for item in events
    )
