from agent.memory.extraction import extract_memory_candidates


def test_extract_memory_candidates_user_memory() -> None:
    candidates = extract_memory_candidates(
        episode={
            "episode_uid": "ep-1",
            "prompt": "请记住，以后默认用中文项目符号回答",
            "answer": "收到，后续默认使用中文项目符号回答。",
        },
        recent_episodes=[],
    )

    assert len(candidates) == 1
    assert candidates[0]["memory_type"] == "user_memory"
    assert candidates[0]["action"] == "ADD"
    assert candidates[0]["source_episode_uid"] == "ep-1"
    assert candidates[0]["evidence"][0]["episode_uid"] == "ep-1"
    assert "默认" in candidates[0]["canonical_text"]


def test_extract_memory_candidates_knowledge_memory() -> None:
    candidates = extract_memory_candidates(
        episode={
            "episode_uid": "ep-2",
            "prompt": "这篇论文的主要结论是什么？",
            "answer": "主要结论是方法A在CIFAR-10上优于方法B。",
        },
        recent_episodes=[],
    )

    assert len(candidates) == 1
    assert candidates[0]["memory_type"] == "knowledge_memory"
    assert candidates[0]["action"] == "ADD"
    assert candidates[0]["source_episode_uid"] == "ep-2"
    assert candidates[0]["evidence"][0]["quote"] == "主要结论是方法A在CIFAR-10上优于方法B。"
