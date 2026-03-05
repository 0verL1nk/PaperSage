import json
import re
from typing import Any


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


def _format_evidence_ref(item: dict[str, Any]) -> str:
    chunk_id = str(item.get("chunk_id") or "").strip()
    if not chunk_id:
        return ""
    parts = [chunk_id]
    page_no = item.get("page_no")
    if isinstance(page_no, int):
        parts.append(f"p{page_no}")
    offset_start = item.get("offset_start")
    offset_end = item.get("offset_end")
    if isinstance(offset_start, int) and isinstance(offset_end, int):
        parts.append(f"o{offset_start}-{offset_end}")
    return "|".join(parts)


def replace_evidence_placeholders(
    answer: str,
    evidence_items: list[dict[str, Any]] | None,
    max_refs: int = 3,
) -> str:
    if not isinstance(answer, str):
        return str(answer)
    value = answer.strip()
    if not value or _is_json_text(value):
        return answer
    if not isinstance(evidence_items, list) or not evidence_items:
        return answer

    refs: list[str] = []
    seen: set[str] = set()
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        ref = _format_evidence_ref(item)
        if not ref or ref in seen:
            continue
        refs.append(ref)
        seen.add(ref)
        if len(refs) >= max(1, int(max_refs)):
            break
    if not refs:
        return answer

    placeholder_pattern = re.compile(
        r"(?:\[\s*(?:文档证据|证据|document evidence)(?:\s*#?\d+)?\s*\]|【\s*(?:文档证据|证据)(?:\s*#?\d+)?\s*】)",
        flags=re.IGNORECASE,
    )
    if not placeholder_pattern.search(answer):
        return answer

    rendered_refs = [f"[{ref}]" for ref in refs]
    replacement_index = 0

    def _replace(_match: re.Match[str]) -> str:
        nonlocal replacement_index
        token = rendered_refs[min(replacement_index, len(rendered_refs) - 1)]
        replacement_index += 1
        return token

    return placeholder_pattern.sub(_replace, answer)
