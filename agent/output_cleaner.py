import json
import re


def _is_json_text(text: str) -> bool:
    try:
        payload = json.loads(text)
    except Exception:
        return False
    return isinstance(payload, (dict, list))


def _looks_like_internal_reasoning(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "okay, let's see",
        "the user asked",
        "planner agent",
        "search_document tool",
        "i should",
        "i need to",
        "用户发来",
        "需要确认",
        "根据规则",
        "检查工具调用条件",
        "系统要求",
        "思考过程",
        "优先调用",
    ]
    return any(marker in lowered for marker in markers)


def split_public_answer_and_reasoning(text: str) -> tuple[str, str]:
    value = text.strip()
    if not value:
        return value, ""

    if _is_json_text(value):
        return value, ""

    tag_match = re.search(r"<answer>(.*?)</answer>", value, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        candidate = tag_match.group(1).strip()
        if candidate:
            prefix = value[: tag_match.start()].strip()
            suffix = value[tag_match.end() :].strip()
            reasoning = "\n\n".join(part for part in (prefix, suffix) if part)
            return candidate, reasoning

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    for marker in ("最终答案：", "Final Answer:", "Answer:"):
        for line in reversed(lines):
            if line.startswith(marker):
                candidate = line[len(marker) :].strip()
                if candidate:
                    line_index = lines.index(line)
                    reasoning = "\n".join(lines[:line_index]).strip()
                    return candidate, reasoning

    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", value) if chunk.strip()]
    if len(paragraphs) >= 2 and _looks_like_internal_reasoning(paragraphs[0]):
        return paragraphs[-1], "\n\n".join(paragraphs[:-1])

    return value, ""


def sanitize_public_answer(text: str) -> str:
    answer, _reasoning = split_public_answer_and_reasoning(text)
    return answer
