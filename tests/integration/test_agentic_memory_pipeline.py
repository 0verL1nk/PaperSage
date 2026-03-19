import time
from types import SimpleNamespace
import json

from agent.adapters.sqlite.project_repository import ensure_default_project_for_user, ensure_default_project_session
from agent.application.agent_center.controller import build_hinted_prompt
from agent.application.agent_center.memory import persist_turn_memory
from agent.memory.store import (
    list_project_memory_episodes,
    list_project_memory_items,
    query_long_term_memory,
)
from utils.task_queue import get_task_status_by_uid
import utils.utils as legacy_utils
from utils.utils import ensure_local_user, init_database


def _wait_for_memory_items(
    *,
    uuid: str,
    project_uid: str,
    expected_count: int,
    db_name: str,
) -> list[dict[str, object]]:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        items = list_project_memory_items(
            uuid=uuid,
            project_uid=project_uid,
            status="active",
            limit=10,
            db_name=db_name,
        )
        if len(items) >= expected_count:
            return items
        time.sleep(0.05)
    return list_project_memory_items(
        uuid=uuid,
        project_uid=project_uid,
        status="active",
        limit=10,
        db_name=db_name,
    )


def test_persist_turn_memory_runs_agentic_memory_pipeline(monkeypatch, tmp_path) -> None:
    class _FakeModel:
        def invoke(self, messages):
            payload = json.loads(messages[1][1])
            current_prompt = str(payload["episode"]["prompt"])
            if "中文项目符号" in current_prompt:
                body = {
                    "candidates": [
                        {
                            "action": "ADD",
                            "memory_type": "user_memory",
                            "title": "回答偏好",
                            "content": "以后默认用中文项目符号回答",
                            "canonical_text": "默认用中文项目符号回答",
                            "dedup_key": "user:response_preferences",
                            "confidence": 0.9,
                            "evidence": [{"quote": "请记住，以后默认用中文项目符号回答"}],
                        }
                    ]
                }
            else:
                body = {
                    "candidates": [
                        {
                            "action": "ADD",
                            "memory_type": "knowledge_memory",
                            "title": "主要结论",
                            "content": "主要结论是方法A在该任务上优于方法B。",
                            "canonical_text": "方法A在该任务上优于方法B",
                            "dedup_key": "knowledge:main-finding",
                            "confidence": 0.84,
                            "evidence": [{"quote": "主要结论是方法A在该任务上优于方法B。"}],
                        }
                    ]
                }
            return SimpleNamespace(content=json.dumps(body, ensure_ascii=False))

    monkeypatch.setattr("agent.memory.extraction.read_api_key_for_user", lambda uuid, db_name="./database.sqlite": f"k:{uuid}")
    monkeypatch.setattr("agent.memory.extraction.read_model_name_for_user", lambda uuid, db_name="./database.sqlite": "fake-model")
    monkeypatch.setattr("agent.memory.extraction.read_base_url_for_user", lambda uuid, db_name="./database.sqlite": "https://example.test")
    monkeypatch.setattr("agent.memory.extraction.create_chat_model", lambda **_kwargs: _FakeModel())
    monkeypatch.chdir(tmp_path)
    init_database("./database.sqlite")
    ensure_local_user("local-user", db_name="./database.sqlite")
    project_uid = ensure_default_project_for_user(
        "local-user",
        db_name="./database.sqlite",
        sync_existing_files=False,
    )
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name="./database.sqlite",
    )

    persist_turn_memory(
        user_uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="请记住，以后默认用中文项目符号回答",
        answer="收到，后续默认用中文项目符号回答。",
    )
    persist_turn_memory(
        user_uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="这篇论文的主要结论是什么？",
        answer="主要结论是方法A在该任务上优于方法B。",
    )

    episodes = list_project_memory_episodes(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        limit=5,
        db_name="./database.sqlite",
    )
    assert len(episodes) == 2

    task_info = get_task_status_by_uid(
        episodes[0]["episode_uid"],
        "memory_writer",
        db_name="./database.sqlite",
    )
    items = _wait_for_memory_items(
        uuid="local-user",
        project_uid=project_uid,
        db_name="./database.sqlite",
        expected_count=2,
    )
    user_items = query_long_term_memory(
        uuid="local-user",
        project_uid=project_uid,
        query="",
        memory_type="user_memory",
        status="active",
        limit=5,
        db_name="./database.sqlite",
    )
    knowledge_items = query_long_term_memory(
        uuid="local-user",
        project_uid=project_uid,
        query="主要结论",
        memory_type="knowledge_memory",
        status="active",
        limit=5,
        db_name="./database.sqlite",
    )
    hinted_prompt = build_hinted_prompt(
        prompt="请基于已有结论继续分析",
        compact_summary="用户持续关注论文结论",
        user_uuid="local-user",
        project_uid=project_uid,
        detect_language_fn=lambda _text: "zh",
        with_language_hint_fn=lambda text, _detector: f"{text} 中文回答",
        search_project_memory_items_fn=query_long_term_memory,
        inject_long_term_memory_fn=lambda text, _memories: text,
        memory_limit=4,
    )

    assert task_info is not None
    assert task_info["status"] in {"queued", "started", "finished"}
    assert len(items) == 2
    assert {str(item["memory_type"]) for item in items} == {"user_memory", "knowledge_memory"}
    assert len(user_items) == 1
    assert "项目符号" in str(user_items[0]["canonical_text"])
    assert len(knowledge_items) == 1
    assert "方法A" in str(knowledge_items[0]["canonical_text"])
    assert "\nP:user_memory:" in hinted_prompt
    assert "中文项目符号" in hinted_prompt
    assert "\nM:knowledge_memory:" in hinted_prompt
    assert "方法A在该任务上优于方法B" in hinted_prompt
    assert not hasattr(legacy_utils, "search_project_memory_items")
