import json
from types import SimpleNamespace

from agent.memory.extraction import extract_memory_candidates


def test_extract_memory_candidates_uses_llm_output_without_keyword_rules(monkeypatch) -> None:
    captured = {}

    class _FakeModel:
        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "candidates": [
                            {
                                "action": "ADD",
                                "memory_type": "knowledge_memory",
                                "title": "实验结论",
                                "content": "方法A在该任务上优于方法B",
                                "canonical_text": "方法A优于方法B",
                                "dedup_key": "knowledge:paper-main-finding",
                                "confidence": 0.82,
                                "evidence": [
                                    {
                                        "episode_uid": "ep-2",
                                        "quote": "方法A在该任务上优于方法B",
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    monkeypatch.setattr("agent.memory.extraction.read_api_key_for_user", lambda uuid: f"k:{uuid}")
    monkeypatch.setattr("agent.memory.extraction.read_model_name_for_user", lambda uuid: "fake-model")
    monkeypatch.setattr("agent.memory.extraction.read_base_url_for_user", lambda uuid: "https://example.test")
    monkeypatch.setattr("agent.memory.extraction.create_chat_model", lambda **_kwargs: _FakeModel())

    candidates = extract_memory_candidates(
        episode={
            "episode_uid": "ep-2",
            "prompt": "帮我沉淀这轮对话里真正值得长期保存的内容",
            "answer": "可以长期保留的事实是方法A在该任务上优于方法B。",
        },
        recent_episodes=[{"episode_uid": "ep-1", "prompt": "上一轮", "answer": "上一轮回答"}],
        active_memories=[{"memory_uid": "m-1", "memory_type": "knowledge_memory", "canonical_text": "旧结论"}],
        user_uuid="local-user",
    )

    assert len(candidates) == 1
    assert candidates[0]["memory_type"] == "knowledge_memory"
    assert candidates[0]["action"] == "ADD"
    assert candidates[0]["source_episode_uid"] == "ep-2"
    assert candidates[0]["evidence"][0]["episode_uid"] == "ep-2"
    assert "长期保存" in str(captured["messages"])
    assert "旧结论" in str(captured["messages"])


def test_extract_memory_candidates_returns_empty_when_llm_returns_no_candidates(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, _messages):
            return SimpleNamespace(content=json.dumps({"candidates": []}, ensure_ascii=False))

    monkeypatch.setattr("agent.memory.extraction.read_api_key_for_user", lambda uuid: f"k:{uuid}")
    monkeypatch.setattr("agent.memory.extraction.read_model_name_for_user", lambda uuid: "fake-model")
    monkeypatch.setattr("agent.memory.extraction.read_base_url_for_user", lambda uuid: "https://example.test")
    monkeypatch.setattr("agent.memory.extraction.create_chat_model", lambda **_kwargs: _FakeModel())

    candidates = extract_memory_candidates(
        episode={
            "episode_uid": "ep-3",
            "prompt": "这一轮只是在寒暄",
            "answer": "好的，继续。",
        },
        recent_episodes=[],
        active_memories=[],
        user_uuid="local-user",
    )

    assert candidates == []
