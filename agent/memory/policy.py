import datetime
from typing import Any


def classify_turn_memory_type(prompt: str, answer: str) -> str:
    text = f"{str(prompt or '')}\n{str(answer or '')}".lower()
    procedural_keywords = [
        "以后",
        "请记住",
        "默认",
        "格式",
        "风格",
        "步骤",
        "流程",
        "template",
        "format",
        "always",
        "from now on",
    ]
    if any(keyword in text for keyword in procedural_keywords):
        return "procedural"

    semantic_keywords = [
        "定义",
        "概念",
        "原理",
        "结论",
        "数据集",
        "方法",
        "术语",
        "是什么",
        "definition",
        "concept",
        "principle",
        "dataset",
        "conclusion",
        "method",
    ]
    if any(keyword in text for keyword in semantic_keywords):
        return "semantic"

    return "episodic"


def ttl_for_memory_type(memory_type: str) -> str:
    now = datetime.datetime.now()
    normalized = str(memory_type or "").strip().lower()
    if normalized == "episodic":
        target = now + datetime.timedelta(days=30)
        return target.strftime("%Y-%m-%d %H:%M:%S")
    if normalized == "procedural":
        target = now + datetime.timedelta(days=90)
        return target.strftime("%Y-%m-%d %H:%M:%S")
    return ""


def inject_long_term_memory(
    prompt: str,
    memory_items: list[dict[str, Any]],
    *,
    max_chars: int = 1600,
) -> str:
    if not isinstance(memory_items, list) or not memory_items:
        return prompt
    lines = ["[长期记忆]"]
    current_len = len(prompt)
    for idx, item in enumerate(memory_items, start=1):
        memory_type = str(item.get("memory_type") or "episodic")
        content = str(item.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        line = f"- ({memory_type}) M{idx}: {content}"
        if current_len + len(line) > max_chars:
            break
        lines.append(line)
        current_len += len(line)
    if len(lines) == 1:
        return prompt
    lines.append("[使用要求] 长期记忆若与当前问题或当前证据冲突，以当前问题和当前证据为准。")
    return f"{prompt}\n\n" + "\n".join(lines)
