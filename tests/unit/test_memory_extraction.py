import json
from types import SimpleNamespace
from pathlib import Path

from agent.memory.extraction import extract_memory_candidates
from utils.utils import ensure_local_user, init_database, save_api_key, save_base_url, save_model_name


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

    monkeypatch.setattr("agent.memory.extraction.read_api_key_for_user", lambda uuid, db_name="./database.sqlite": f"k:{uuid}")
    monkeypatch.setattr("agent.memory.extraction.read_model_name_for_user", lambda uuid, db_name="./database.sqlite": "fake-model")
    monkeypatch.setattr("agent.memory.extraction.read_base_url_for_user", lambda uuid, db_name="./database.sqlite": "https://example.test")
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
    assert "fragment statement" in str(captured["messages"]).lower()
    assert "q/a" in str(captured["messages"]).lower()


def test_extract_memory_candidates_returns_empty_when_llm_returns_no_candidates(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, _messages):
            return SimpleNamespace(content=json.dumps({"candidates": []}, ensure_ascii=False))

    monkeypatch.setattr("agent.memory.extraction.read_api_key_for_user", lambda uuid, db_name="./database.sqlite": f"k:{uuid}")
    monkeypatch.setattr("agent.memory.extraction.read_model_name_for_user", lambda uuid, db_name="./database.sqlite": "fake-model")
    monkeypatch.setattr("agent.memory.extraction.read_base_url_for_user", lambda uuid, db_name="./database.sqlite": "https://example.test")
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


def test_extract_memory_candidates_reads_credentials_from_provided_db(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "memory-extraction.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))
    save_api_key("local-user", "test-key", db_name=str(db_path))
    save_model_name("local-user", "fake-model", db_name=str(db_path))
    save_base_url("local-user", "https://example.test", db_name=str(db_path))

    class _FakeModel:
        def invoke(self, _messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "candidates": [
                            {
                                "action": "ADD",
                                "memory_type": "user_memory",
                                "title": "回答偏好",
                                "content": "以后用中文要点回答",
                                "canonical_text": "以后用中文要点回答",
                                "dedup_key": "user:response_preferences",
                                "confidence": 0.91,
                                "evidence": [{"quote": "以后用中文要点回答"}],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    monkeypatch.setattr("agent.memory.extraction.create_chat_model", lambda **_kwargs: _FakeModel())

    candidates = extract_memory_candidates(
        episode={
            "episode_uid": "ep-db",
            "prompt": "之后回复改成中文要点。",
            "answer": "收到，之后会用中文要点回答。",
        },
        recent_episodes=[],
        active_memories=[],
        user_uuid="local-user",
        db_name=str(db_path),
    )

    assert len(candidates) == 1
    assert candidates[0]["memory_type"] == "user_memory"
    assert candidates[0]["source_episode_uid"] == "ep-db"


def test_extract_memory_candidates_accepts_single_candidate_json_object(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, _messages):
            return SimpleNamespace(
                content="""
这里是提取结果：
{
  "action": "ADD",
  "memory_type": "knowledge_memory",
  "title": "项目事实",
  "content": "project Apollo deadline moved to Apr 30",
  "canonical_text": "project Apollo deadline moved to Apr 30",
  "dedup_key": "project:apollo_deadline",
  "confidence": 0.88,
  "evidence": [{"quote": "project Apollo deadline moved to Apr 30"}]
}
"""
            )

    monkeypatch.setattr("agent.memory.extraction.read_api_key_for_user", lambda uuid, db_name="./database.sqlite": f"k:{uuid}")
    monkeypatch.setattr("agent.memory.extraction.read_model_name_for_user", lambda uuid, db_name="./database.sqlite": "fake-model")
    monkeypatch.setattr("agent.memory.extraction.read_base_url_for_user", lambda uuid, db_name="./database.sqlite": "https://example.test")
    monkeypatch.setattr("agent.memory.extraction.create_chat_model", lambda **_kwargs: _FakeModel())

    candidates = extract_memory_candidates(
        episode={
            "episode_uid": "ep-single",
            "prompt": "记录一下 Apollo 项目时间变更。",
            "answer": "project Apollo deadline moved to Apr 30",
        },
        recent_episodes=[],
        active_memories=[],
        user_uuid="local-user",
    )

    assert len(candidates) == 1
    assert candidates[0]["memory_type"] == "knowledge_memory"
    assert candidates[0]["canonical_text"] == "project Apollo deadline moved to Apr 30"


def test_extract_memory_candidates_accepts_string_evidence_items(monkeypatch) -> None:
    class _FakeModel:
        def invoke(self, _messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "candidates": [
                            {
                                "action": "ADD",
                                "memory_type": "user_memory",
                                "title": "Response preference",
                                "content": "user prefers concise answers",
                                "canonical_text": "user prefers concise answers",
                                "dedup_key": "user:concise_answers",
                                "confidence": 0.93,
                                "evidence": [
                                    "Please keep future replies concise.",
                                    "Understood, I will keep future replies concise.",
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    monkeypatch.setattr("agent.memory.extraction.read_api_key_for_user", lambda uuid, db_name="./database.sqlite": f"k:{uuid}")
    monkeypatch.setattr("agent.memory.extraction.read_model_name_for_user", lambda uuid, db_name="./database.sqlite": "fake-model")
    monkeypatch.setattr("agent.memory.extraction.read_base_url_for_user", lambda uuid, db_name="./database.sqlite": "https://example.test")
    monkeypatch.setattr("agent.memory.extraction.create_chat_model", lambda **_kwargs: _FakeModel())

    candidates = extract_memory_candidates(
        episode={
            "episode_uid": "ep-evidence",
            "prompt": "Please keep future replies concise.",
            "answer": "Understood, I will keep future replies concise.",
        },
        recent_episodes=[],
        active_memories=[],
        user_uuid="local-user",
    )

    assert len(candidates) == 1
    assert candidates[0]["evidence"][0]["episode_uid"] == "ep-evidence"
    assert candidates[0]["evidence"][0]["quote"] == "Please keep future replies concise."
