from agent.memory.policy import (
    classify_turn_memory_type,
    inject_long_term_memory,
    ttl_for_memory_type,
)


def test_classify_turn_memory_type_procedural():
    memory_type = classify_turn_memory_type(
        "以后请统一用中文回答并给出三点格式",
        "好的，后续将按该格式执行。",
    )
    assert memory_type == "procedural"


def test_classify_turn_memory_type_semantic():
    memory_type = classify_turn_memory_type(
        "这个方法的定义是什么？",
        "该方法定义为对比学习框架。",
    )
    assert memory_type == "semantic"


def test_ttl_for_memory_type():
    assert ttl_for_memory_type("semantic") == ""
    assert ttl_for_memory_type("episodic")
    assert ttl_for_memory_type("procedural")


def test_inject_long_term_memory_appends_section():
    prompt = "请回答问题。"
    augmented = inject_long_term_memory(
        prompt,
        [{"memory_type": "semantic", "content": "该论文数据集是 CIFAR-10"}],
    )
    assert "[长期记忆]" in augmented
    assert "CIFAR-10" in augmented
