from agent.output_cleaner import sanitize_public_answer


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
