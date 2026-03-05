from agent.output_cleaner import (
    replace_evidence_placeholders,
    sanitize_public_answer,
    split_public_answer_and_reasoning,
)


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


def test_replace_evidence_placeholders_with_specific_refs() -> None:
    answer = "该方法提升了召回率[文档证据]，并降低了误报【证据】。"
    evidences = [
        {"chunk_id": "chunk_12", "page_no": 4, "offset_start": 100, "offset_end": 168},
        {"chunk_id": "chunk_33", "page_no": 8},
    ]
    updated = replace_evidence_placeholders(answer, evidences)
    assert "[chunk_12|p4|o100-168]" in updated
    assert "[chunk_33|p8]" in updated
    assert "[文档证据]" not in updated
    assert "【证据】" not in updated


def test_replace_evidence_placeholders_keeps_json_unchanged() -> None:
    answer = '{"name":"主题","children":[]}'
    evidences = [{"chunk_id": "chunk_1", "page_no": 1}]
    assert replace_evidence_placeholders(answer, evidences) == answer


def test_replace_evidence_placeholders_without_evidence_keeps_answer() -> None:
    answer = "结论如下[文档证据]。"
    assert replace_evidence_placeholders(answer, []) == answer
