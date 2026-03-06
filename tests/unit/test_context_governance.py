from agent.context_governance import (
    auto_compact_messages,
    build_context_usage_snapshot,
    extract_active_skills_from_trace,
    extract_skill_context_texts_from_trace,
    inject_compact_summary,
    should_trigger_auto_compact,
)


def _build_messages(count: int) -> list[dict]:
    messages: list[dict] = []
    for idx in range(count):
        messages.append({"role": "user", "content": f"问题 {idx}：请总结实验设置与结论。"})
        messages.append({"role": "assistant", "content": f"回答 {idx}：实验设置为 A，结论为 B。"})
    return messages


def test_auto_compact_noop_under_budget(monkeypatch):
    monkeypatch.setenv("AGENT_CONTEXT_MAX_INPUT_TOKENS", "200000")
    messages = _build_messages(2)
    result = auto_compact_messages(messages, current_summary="")
    assert result.compacted is False
    assert result.messages == messages
    assert result.used_llm is False
    assert result.anchor_count == 0


def test_auto_compact_reduces_history_when_over_budget(monkeypatch):
    monkeypatch.setenv("AGENT_CONTEXT_MAX_INPUT_TOKENS", "2048")
    monkeypatch.setenv("AGENT_AUTO_COMPACT_TRIGGER_RATIO", "0.30")
    monkeypatch.setenv("AGENT_AUTO_COMPACT_TARGET_RATIO", "0.20")
    monkeypatch.setenv("AGENT_AUTO_COMPACT_RECENT_MESSAGES", "4")
    messages = _build_messages(120)

    result = auto_compact_messages(messages, current_summary="")

    assert result.compacted is True
    assert result.summary
    assert result.compacted_token_estimate < result.source_token_estimate
    assert any(msg.get("auto_compact") for msg in result.messages)
    assert len(result.messages) < len(messages)
    assert result.anchor_count >= 0


def test_auto_compact_uses_llm_summary_and_anchors(monkeypatch):
    class _FakeResp:
        def __init__(self, content):
            self.content = content

    class _FakeLLM:
        def invoke(self, _prompt):
            return _FakeResp(
                '{"summary":"项目重点：方法与结论。",'
                '"anchors":[{"id":"F1","claim":"关键方法为A","refs":["m1","m2"]}]}'
            )

    monkeypatch.setenv("AGENT_CONTEXT_MAX_INPUT_TOKENS", "2048")
    monkeypatch.setenv("AGENT_AUTO_COMPACT_TRIGGER_RATIO", "0.30")
    monkeypatch.setenv("AGENT_AUTO_COMPACT_TARGET_RATIO", "0.20")
    monkeypatch.setenv("AGENT_AUTO_COMPACT_RECENT_MESSAGES", "4")
    messages = _build_messages(120)

    result = auto_compact_messages(messages, current_summary="", llm=_FakeLLM())

    assert result.compacted is True
    assert result.used_llm is True
    assert result.anchor_count == 1
    assert "事实锚点" in result.summary
    assert "F1" in result.summary


def test_inject_compact_summary_appends_memory():
    prompt = "请继续回答。"
    summary = "用户关注实验结论与局限性。"
    augmented = inject_compact_summary(prompt, summary)
    assert "[会话压缩记忆]" in augmented
    assert "局限性" in augmented


def test_build_context_usage_snapshot_contains_required_keys(monkeypatch):
    monkeypatch.setenv("AGENT_CONTEXT_MAX_INPUT_TOKENS", "10000")
    usage = build_context_usage_snapshot(
        messages=_build_messages(1),
        compact_summary="历史摘要",
        active_skills={"summary"},
    )
    assert usage["model_window_tokens"] == 10000
    assert "breakdown" in usage
    for key in (
        "system_prompt",
        "custom_agents",
        "memory_files",
        "skills",
        "tools",
        "messages",
        "autocompact_buffer",
        "free_space",
    ):
        assert key in usage["breakdown"]


def test_extract_active_skills_supports_python_dict_content():
    trace_payload = [
        {
            "receiver": "use_skill",
            "content": "{'skill_name': 'summary', 'task': 'x'}",
        }
    ]
    assert extract_active_skills_from_trace(trace_payload) == {"summary"}


def test_extract_active_skills_supports_skill_activate_event():
    trace_payload = [
        {
            "performative": "skill_activate",
            "receiver": "skill:mindmap",
            "content": "activate mindmap",
        }
    ]
    assert extract_active_skills_from_trace(trace_payload) == {"mindmap"}


def test_extract_skill_context_texts_from_trace_supports_new_skill_events():
    trace_payload = [
        {
            "performative": "tool_call",
            "receiver": "use_skill",
            "content": "{'skill_name': 'summary', 'task': 'x'}",
        },
        {
            "performative": "skill_activate",
            "receiver": "skill:summary",
            "content": "activate summary",
        },
        {
            "performative": "tool_result",
            "sender": "use_skill",
            "content": "Skill: summary\nDescription: ...",
        },
    ]
    texts = extract_skill_context_texts_from_trace(trace_payload)
    assert len(texts) == 3
    assert any("skill_name" in item for item in texts)
    assert any(item.startswith("activate summary") for item in texts)
    assert any(item.startswith("Skill: summary") for item in texts)


def test_should_trigger_auto_compact(monkeypatch):
    monkeypatch.setenv("AGENT_CONTEXT_MAX_INPUT_TOKENS", "2048")
    monkeypatch.setenv("AGENT_AUTO_COMPACT_TRIGGER_RATIO", "0.30")
    assert should_trigger_auto_compact(_build_messages(120)) is True
    assert should_trigger_auto_compact(_build_messages(1)) is False
