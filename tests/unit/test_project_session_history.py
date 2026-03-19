import datetime
from pathlib import Path

from agent.adapters.sqlite.project_repository import (
    create_project,
    create_project_session,
    delete_project_session,
    ensure_default_project_session,
    list_project_session_messages,
    list_project_sessions,
    save_project_session_messages,
    update_project_session,
)
from agent.memory.store import (
    get_project_session_compact_memory,
    get_project_memory_episode,
    list_project_memory_items,
    query_long_term_memory,
    save_project_memory_episode,
    save_project_session_compact_memory,
    search_project_memory_items,
    upsert_project_memory_item,
)
from utils.utils import (
    ensure_local_user,
    init_database,
)


def _prepare_project(tmp_path: Path) -> tuple[Path, str]:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))
    project = create_project(
        uuid="local-user",
        project_name="session-demo",
        db_name=str(db_path),
    )
    return db_path, str(project["project_uid"])


def test_project_session_message_roundtrip(tmp_path: Path) -> None:
    db_path, project_uid = _prepare_project(tmp_path)
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )

    messages = [
        {"role": "assistant", "content": "欢迎来到项目会话"},
        {
            "role": "user",
            "content": "请总结这篇论文",
            "workflow_mode": "react",
        },
    ]
    save_project_session_messages(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid="local-user",
        messages=messages,
        db_name=str(db_path),
    )

    loaded = list_project_session_messages(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    assert loaded == messages

    sessions = list_project_sessions(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    assert len(sessions) == 1
    assert sessions[0]["session_uid"] == session_uid
    assert sessions[0]["message_count"] == 2
    assert sessions[0]["last_message"] == "请总结这篇论文"


def test_project_session_update_and_delete(tmp_path: Path) -> None:
    db_path, project_uid = _prepare_project(tmp_path)
    session_a = create_project_session(
        project_uid=project_uid,
        uuid="local-user",
        session_name="会话 A",
        db_name=str(db_path),
    )
    session_b = create_project_session(
        project_uid=project_uid,
        uuid="local-user",
        session_name="会话 B",
        db_name=str(db_path),
    )

    updated = update_project_session(
        session_uid=str(session_b["session_uid"]),
        project_uid=project_uid,
        uuid="local-user",
        session_name="重点会话",
        is_pinned=1,
        db_name=str(db_path),
    )
    assert updated

    sessions = list_project_sessions(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    assert sessions[0]["session_uid"] == str(session_b["session_uid"])
    assert sessions[0]["session_name"] == "重点会话"
    assert sessions[0]["is_pinned"] == 1

    save_project_session_messages(
        session_uid=str(session_b["session_uid"]),
        project_uid=project_uid,
        uuid="local-user",
        messages=[{"role": "assistant", "content": "to be deleted"}],
        db_name=str(db_path),
    )
    save_project_session_compact_memory(
        session_uid=str(session_b["session_uid"]),
        project_uid=project_uid,
        uuid="local-user",
        compact_summary="待删除摘要",
        anchors=[{"id": "F1", "claim": "x", "refs": ["m1"]}],
        db_name=str(db_path),
    )
    deleted = delete_project_session(
        session_uid=str(session_b["session_uid"]),
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    assert deleted

    remaining = list_project_sessions(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    assert len(remaining) == 1
    assert remaining[0]["session_uid"] == str(session_a["session_uid"])

    removed_messages = list_project_session_messages(
        session_uid=str(session_b["session_uid"]),
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    assert removed_messages == []
    removed_summary = get_project_session_compact_memory(
        session_uid=str(session_b["session_uid"]),
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    assert removed_summary["compact_summary"] == ""
    assert removed_summary["anchors"] == []


def test_project_session_compact_memory_roundtrip(tmp_path: Path) -> None:
    db_path, project_uid = _prepare_project(tmp_path)
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )

    save_project_session_compact_memory(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid="local-user",
        compact_summary="用户关注实验结论和局限性",
        anchors=[{"id": "F1", "claim": "关键结论", "refs": ["m2"]}],
        db_name=str(db_path),
    )
    loaded = get_project_session_compact_memory(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    assert loaded["compact_summary"] == "用户关注实验结论和局限性"
    assert loaded["anchors"] == [{"id": "F1", "claim": "关键结论", "refs": ["m2"]}]

    save_project_session_compact_memory(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid="local-user",
        compact_summary="更新后的摘要",
        anchors=[],
        db_name=str(db_path),
    )
    loaded_updated = get_project_session_compact_memory(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    assert loaded_updated["compact_summary"] == "更新后的摘要"
    assert loaded_updated["anchors"] == []


def test_project_memory_item_upsert_and_search(tmp_path: Path) -> None:
    db_path, project_uid = _prepare_project(tmp_path)
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )

    uid_first = upsert_project_memory_item(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type="episodic",
        title="实验结论",
        content="Q: 这篇论文的主要结论是什么 A: 结论是方法A优于方法B",
        source_prompt="主要结论是什么",
        source_answer="方法A优于方法B",
        db_name=str(db_path),
    )
    assert uid_first

    uid_second = upsert_project_memory_item(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type="episodic",
        title="实验结论重复",
        content="Q: 这篇论文的主要结论是什么 A: 结论是方法A优于方法B",
        source_prompt="主要结论是什么",
        source_answer="方法A优于方法B",
        db_name=str(db_path),
    )
    assert uid_second == uid_first

    upsert_project_memory_item(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type="semantic",
        title="数据集",
        content="该实验主要使用CIFAR-10与ImageNet数据集",
        db_name=str(db_path),
    )

    all_items = list_project_memory_items(
        uuid="local-user",
        project_uid=project_uid,
        limit=20,
        db_name=str(db_path),
    )
    assert len(all_items) == 2

    matched = search_project_memory_items(
        uuid="local-user",
        project_uid=project_uid,
        query="主要结论 方法A",
        limit=3,
        db_name=str(db_path),
    )
    assert matched
    assert "方法A优于方法B" in matched[0]["content"]


def test_project_memory_episode_roundtrip(tmp_path: Path) -> None:
    db_path, project_uid = _prepare_project(tmp_path)
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )

    episode_uid = save_project_memory_episode(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="请记住以后默认输出中文摘要",
        answer="好的，后续我会默认输出中文摘要。",
        db_name=str(db_path),
    )
    assert episode_uid

    episode = get_project_memory_episode(
        episode_uid=episode_uid,
        db_name=str(db_path),
    )
    assert episode["episode_uid"] == episode_uid
    assert episode["uuid"] == "local-user"
    assert episode["project_uid"] == project_uid
    assert episode["session_uid"] == session_uid
    assert episode["prompt"] == "请记住以后默认输出中文摘要"
    assert episode["answer"] == "好的，后续我会默认输出中文摘要。"


def test_project_memory_expired_items_are_filtered(tmp_path: Path) -> None:
    db_path, project_uid = _prepare_project(tmp_path)
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    past = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    future = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    upsert_project_memory_item(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type="episodic",
        title="过期记忆",
        content="这个记忆已过期",
        expires_at=past,
        db_name=str(db_path),
    )
    upsert_project_memory_item(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type="episodic",
        title="有效记忆",
        content="这个记忆有效",
        expires_at=future,
        db_name=str(db_path),
    )

    items = list_project_memory_items(
        uuid="local-user",
        project_uid=project_uid,
        limit=20,
        include_expired=False,
        db_name=str(db_path),
    )
    assert len(items) == 1
    assert items[0]["title"] == "有效记忆"


def test_project_memory_item_structured_fields_roundtrip(tmp_path: Path) -> None:
    db_path, project_uid = _prepare_project(tmp_path)
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    episode_uid = save_project_memory_episode(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="用户说以后都用项目符号回答",
        answer="收到，后续默认使用项目符号回答。",
        db_name=str(db_path),
    )

    memory_uid = upsert_project_memory_item(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type="procedural",
        title="回答偏好",
        content="以后默认使用项目符号回答",
        canonical_text="默认使用项目符号回答",
        dedup_key="pref:bullet-style",
        status="active",
        confidence=0.92,
        source_episode_uid=episode_uid,
        evidence=[{"episode_uid": episode_uid, "quote": "以后都用项目符号回答"}],
        db_name=str(db_path),
    )
    assert memory_uid

    items = list_project_memory_items(
        uuid="local-user",
        project_uid=project_uid,
        limit=10,
        db_name=str(db_path),
    )
    assert len(items) == 1
    assert items[0]["canonical_text"] == "默认使用项目符号回答"
    assert items[0]["dedup_key"] == "pref:bullet-style"
    assert items[0]["status"] == "active"
    assert items[0]["confidence"] == 0.92
    assert items[0]["source_episode_uid"] == episode_uid
    assert items[0]["evidence"] == [{"episode_uid": episode_uid, "quote": "以后都用项目符号回答"}]


def test_query_long_term_memory_filters_active_typed_memories(tmp_path: Path) -> None:
    db_path, project_uid = _prepare_project(tmp_path)
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    episode_uid = save_project_memory_episode(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="请记住以后默认用中文回答",
        answer="收到，后续默认用中文回答。",
        db_name=str(db_path),
    )
    upsert_project_memory_item(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type="user_memory",
        title="语言偏好",
        content="默认用中文回答",
        canonical_text="默认用中文回答",
        dedup_key="user:response_preferences",
        status="active",
        source_episode_uid=episode_uid,
        db_name=str(db_path),
    )
    upsert_project_memory_item(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type="knowledge_memory",
        title="旧结论",
        content="方法A不如方法B",
        canonical_text="方法A不如方法B",
        dedup_key="knowledge:main-conclusion-old",
        status="superseded",
        source_episode_uid=episode_uid,
        db_name=str(db_path),
    )
    upsert_project_memory_item(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type="knowledge_memory",
        title="新结论",
        content="方法A优于方法B",
        canonical_text="方法A优于方法B",
        dedup_key="knowledge:main-conclusion",
        status="active",
        source_episode_uid=episode_uid,
        db_name=str(db_path),
    )

    knowledge = query_long_term_memory(
        uuid="local-user",
        project_uid=project_uid,
        query="主要结论",
        memory_type="knowledge_memory",
        status="active",
        limit=5,
        db_name=str(db_path),
    )
    user_items = query_long_term_memory(
        uuid="local-user",
        project_uid=project_uid,
        query="",
        memory_type="user_memory",
        status="active",
        limit=5,
        db_name=str(db_path),
    )

    assert len(knowledge) == 1
    assert knowledge[0]["canonical_text"] == "方法A优于方法B"
    assert len(user_items) == 1
    assert user_items[0]["memory_type"] == "user_memory"
