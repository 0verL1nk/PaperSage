from agent.output_cleaner import sanitize_public_answer, split_public_answer_and_reasoning


def test_sanitize_extracts_answer_tag_block() -> None:
    raw = "analysis...\n<answer>最终答案：请提供要总结的具体对象。</answer>\nmore"
    assert sanitize_public_answer(raw) == "最终答案：请提供要总结的具体对象。"


def test_sanitize_keeps_valid_json() -> None:
    raw = '{"name":"主题","children":[{"name":"子主题","children":[]}]}'
    assert sanitize_public_answer(raw) == raw


def test_sanitize_removes_reasoning_preface_paragraph() -> None:
    raw = (
        "Okay, let's see. The user asked a vague question.\n"
        "I should clarify scope first.\n\n"
        "请明确你希望我总结的内容：是整篇论文、某一章节，还是某段对话？"
    )
    assert sanitize_public_answer(raw) == "请明确你希望我总结的内容：是整篇论文、某一章节，还是某段对话？"


def test_split_extracts_reasoning_and_final_answer_for_chinese_preface() -> None:
    raw = (
        "好的，用户发来“你好”，需要按要求用中文回答。\n"
        "根据规则，优先调用search_document检索证据。\n\n"
        "你好！请告诉我你想了解论文的哪一部分。"
    )
    answer, reasoning = split_public_answer_and_reasoning(raw)
    assert answer == "你好！请告诉我你想了解论文的哪一部分。"
    assert "用户发来" in reasoning
