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
    ]
    return any(marker in lowered for marker in markers)


def sanitize_public_answer(text: str) -> str:
    value = text.strip()
    if not value:
        return value

    if _is_json_text(value):
        return value

    tag_match = re.search(r"<answer>(.*?)</answer>", value, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        candidate = tag_match.group(1).strip()
        if candidate:
            return candidate

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    for marker in ("最终答案：", "Final Answer:", "Answer:"):
        for line in reversed(lines):
            if line.startswith(marker):
                candidate = line[len(marker) :].strip()
                if candidate:
                    return candidate

    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", value) if chunk.strip()]
    if len(paragraphs) >= 2 and _looks_like_internal_reasoning(paragraphs[0]):
        return paragraphs[-1]

    return value
