from types import SimpleNamespace

from agent.stream import (
    extract_ask_human_requests_from_result,
    extract_mode_activation_events_from_result,
    extract_result_text,
    extract_skill_activation_events_from_result,
    extract_stream_text,
    extract_tool_activation_events_from_result,
    extract_tool_names_from_result,
    extract_tool_trace_events_from_result,
    extract_trace_events_from_update,
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


def test_extract_tool_names_from_result_collects_calls_and_tool_messages():
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "search_document", "args": {"query": "abc"}},
                    {"name": "search_web", "args": {"query": "xyz"}},
                ],
            },
            {"role": "tool", "name": "search_document", "content": "hit"},
        ]
    }

    assert extract_tool_names_from_result(result) == {"search_document", "search_web"}


def test_extract_ask_human_requests_from_result_collects_from_tool_messages():
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "ask_human",
                        "args": {
                            "question": "是否继续？",
                            "context": "需要人工批准",
                            "urgency": "high",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "ask_human",
                "content": '{"type":"ask_human","question":"是否继续？","context":"需要人工批准","urgency":"high"}',
            },
        ]
    }

    requests = extract_ask_human_requests_from_result(result)
    assert len(requests) == 1
    assert requests[0]["question"] == "是否继续？"
    assert requests[0]["urgency"] == "high"


def test_extract_tool_trace_events_from_result_collects_calls_and_results():
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "search_document", "args": {"query": "abc"}},
                ],
            },
            {"role": "tool", "name": "search_document", "content": "hit"},
        ]
    }

    events = extract_tool_trace_events_from_result(result)
    assert any(item["performative"] == "tool_call" for item in events)
    assert any(item["performative"] == "tool_result" for item in events)


def test_extract_skill_activation_events_from_result_collects_use_skill_calls():
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "use_skill",
                        "args": {"skill_name": "mindmap", "task": "生成导图"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "use_skill",
                "content": "Skill: mindmap\nDescription: ...",
            },
        ]
    }

    events = extract_skill_activation_events_from_result(result)
    assert events
    assert any(item["performative"] == "skill_activate" for item in events)
    assert any(item["receiver"] == "skill:mindmap" for item in events)


def test_extract_tool_activation_events_from_result_collects_activate_tool_calls():
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "activate_tool",
                        "args": {"tool_name": "search_web", "reason": "need latest"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "activate_tool",
                "content": '{"type":"tool_activate","tool_name":"search_web"}',
            },
        ]
    }

    events = extract_tool_activation_events_from_result(result)
    assert events
    assert any(item["performative"] == "tool_activate" for item in events)
    assert any(item["receiver"] == "tool:search_web" for item in events)


def test_extract_mode_activation_events_from_result_collects_mode_tools():
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "start_plan",
                        "args": {"goal": "拆步骤", "reason": "task is multi-step"},
                    },
                    {
                        "name": "start_team",
                        "args": {"goal": "交叉验证", "reason": "need reviewer"},
                    },
                ],
            },
            {
                "role": "tool",
                "name": "start_plan",
                "content": '{"type":"mode_activate","mode":"plan","goal":"拆步骤"}',
            },
            {
                "role": "tool",
                "name": "start_team",
                "content": '{"type":"mode_activate","mode":"team","goal":"交叉验证"}',
            },
        ]
    }

    events = extract_mode_activation_events_from_result(result)
    assert len(events) == 2
    assert any(item["receiver"] == "mode:plan" for item in events)
    assert any(item["receiver"] == "mode:team" for item in events)


def test_extract_mode_activation_events_from_result_requires_successful_tool_result():
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "start_plan",
                        "args": {"goal": "", "reason": "bad input"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "start_plan",
                "content": "goal is required to start plan mode.",
            },
        ]
    }

    events = extract_mode_activation_events_from_result(result)
    assert events == []
