import json
import math
import os
import re
from ast import literal_eval
from functools import lru_cache
from dataclasses import dataclass
from typing import Any

from .multi_agent_a2a import (
    PLANNER_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
)
from .paper_agent import PAPER_QA_SYSTEM_PROMPT
from .workflow_router import ROUTER_INSTRUCTION


COMPACT_SUMMARY_HEADER = "【自动压缩摘要】"
BOOTSTRAP_PREFIX = "已加载文档《"
LLM_COMPACT_SYSTEM_PROMPT = (
    "你是对话记忆压缩器。目标：在不丢失关键事实的前提下压缩历史对话。"
    "只返回严格 JSON，不要 markdown，不要额外解释。"
    '输出格式: {"summary":"...","anchors":[{"id":"F1","claim":"...","refs":["m12","m19"]}]}。'
    "summary 用中文，控制在 8 条以内。anchors 提供 2-8 条可核验事实锚点。"
    "refs 必须引用消息编号（m数字）。"
)


@dataclass(frozen=True)
class AutoCompactResult:
    messages: list[dict[str, Any]]
    summary: str
    compacted: bool
    source_message_count: int
    source_token_estimate: int
    compacted_token_estimate: int
    used_llm: bool
    anchor_count: int


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value.strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value.strip())
    except Exception:
        return default


def model_context_window_tokens() -> int:
    return max(2048, _env_int("AGENT_CONTEXT_MAX_INPUT_TOKENS", 200_000))


def reserved_output_tokens() -> int:
    return max(512, _env_int("AGENT_CONTEXT_RESERVED_OUTPUT_TOKENS", 16_000))


def compact_trigger_ratio() -> float:
    return min(0.95, max(0.20, _env_float("AGENT_AUTO_COMPACT_TRIGGER_RATIO", 0.55)))


def compact_target_ratio() -> float:
    return min(0.90, max(0.10, _env_float("AGENT_AUTO_COMPACT_TARGET_RATIO", 0.38)))


def compact_recent_messages() -> int:
    return max(4, _env_int("AGENT_AUTO_COMPACT_RECENT_MESSAGES", 12))


def compact_max_summary_tokens() -> int:
    return max(256, _env_int("AGENT_AUTO_COMPACT_MAX_SUMMARY_TOKENS", 2800))


def llm_compact_enabled() -> bool:
    value = os.getenv("AGENT_AUTO_COMPACT_LLM_ENABLED", "").strip().lower()
    if not value:
        return True
    return value in {"1", "true", "yes", "on"}


@lru_cache(maxsize=8)
def _resolve_encoding(model_name: str):
    try:
        import tiktoken
    except Exception:
        return None
    normalized = model_name.strip() if isinstance(model_name, str) else ""
    try:
        if normalized:
            return tiktoken.encoding_for_model(normalized)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    tokenizer_model = os.getenv("AGENT_TOKENIZER_MODEL", "").strip()
    encoding = _resolve_encoding(tokenizer_model)
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception:
            pass
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii_chars = len(text) - ascii_chars
    # Mixed-language heuristic:
    # - Latin text: ~4 chars/token
    # - CJK-heavy text: ~1.6 chars/token
    return max(1, math.ceil(ascii_chars / 4.0) + math.ceil(non_ascii_chars / 1.6))


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        total += estimate_tokens(_message_text(message)) + 8
    return total


def _collapse_text(text: str, limit: int = 140) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _summarize_messages(messages: list[dict[str, Any]], max_lines: int = 12) -> str:
    user_lines: list[str] = []
    assistant_lines: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        text = _collapse_text(_message_text(message))
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        if role == "user":
            user_lines.append(text)
        elif role == "assistant":
            assistant_lines.append(text)
    user_lines = user_lines[-max_lines:]
    assistant_lines = assistant_lines[-max_lines:]
    parts: list[str] = []
    if user_lines:
        parts.append("用户历史诉求:")
        parts.extend(f"- {item}" for item in user_lines)
    if assistant_lines:
        parts.append("助手历史结论:")
        parts.extend(f"- {item}" for item in assistant_lines)
    return "\n".join(parts).strip()


def _cap_summary_tokens(summary: str, token_limit: int) -> str:
    if estimate_tokens(summary) <= token_limit:
        return summary
    lines = [line for line in summary.splitlines() if line.strip()]
    while lines and estimate_tokens("\n".join(lines)) > token_limit:
        lines.pop(0)
    return "\n".join(lines).strip()


def _extract_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _llm_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _normalize_anchor_items(raw_anchors: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_anchors, list):
        return []
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_anchors, start=1):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        anchor_id = str(item.get("id") or f"F{idx}").strip() or f"F{idx}"
        refs_raw = item.get("refs")
        refs: list[str] = []
        if isinstance(refs_raw, list):
            for ref in refs_raw:
                value = str(ref).strip()
                if value.startswith("m"):
                    refs.append(value)
        normalized.append({"id": anchor_id, "claim": claim, "refs": refs})
    return normalized


def _format_anchors(anchors: list[dict[str, Any]]) -> str:
    if not anchors:
        return ""
    lines = ["事实锚点:"]
    for item in anchors:
        refs = item.get("refs") or []
        refs_text = ",".join(refs) if refs else "n/a"
        lines.append(f"- [{item.get('id', 'F')}] {item.get('claim', '')} | refs: {refs_text}")
    return "\n".join(lines)


def _build_compact_input(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown")
        text = _collapse_text(_message_text(message), limit=260)
        if not text:
            continue
        lines.append(f"m{idx} | role={role} | text={text}")
    return "\n".join(lines)


def _summarize_messages_with_llm(
    llm: Any,
    messages: list[dict[str, Any]],
    current_summary: str,
) -> tuple[str, list[dict[str, Any]]] | None:
    if llm is None or not llm_compact_enabled():
        return None
    compact_input = _build_compact_input(messages)
    if not compact_input.strip():
        return None
    prompt = (
        f"{LLM_COMPACT_SYSTEM_PROMPT}\n\n"
        f"已有摘要（可为空）:\n{current_summary.strip() or '(none)'}\n\n"
        f"待压缩历史消息:\n{compact_input}\n"
    )
    try:
        result = llm.invoke(prompt)
    except Exception:
        return None
    text = _llm_content_to_text(getattr(result, "content", result))
    json_block = _extract_json_block(text)
    if not json_block:
        return None
    try:
        payload = json.loads(json_block)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        return None
    anchors = _normalize_anchor_items(payload.get("anchors"))
    return summary, anchors


def _merge_summaries(existing: str, incremental: str, token_limit: int) -> str:
    existing_clean = existing.strip()
    incremental_clean = incremental.strip()
    if not existing_clean:
        return _cap_summary_tokens(incremental_clean, token_limit)
    if not incremental_clean:
        return _cap_summary_tokens(existing_clean, token_limit)
    merged = f"{existing_clean}\n{incremental_clean}"
    return _cap_summary_tokens(merged, token_limit)


def _is_bootstrap_message(message: dict[str, Any]) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") != "assistant":
        return False
    text = _message_text(message)
    return isinstance(text, str) and text.startswith(BOOTSTRAP_PREFIX)


def auto_compact_messages(
    messages: list[dict[str, Any]],
    *,
    current_summary: str = "",
    llm: Any | None = None,
) -> AutoCompactResult:
    source_messages = [msg for msg in messages if isinstance(msg, dict)]
    source_tokens = estimate_message_tokens(source_messages)
    max_input = model_context_window_tokens()
    trigger_tokens = int(max_input * compact_trigger_ratio())
    target_tokens = int(max_input * compact_target_ratio())

    base_messages = [msg for msg in source_messages if not msg.get("auto_compact")]
    bootstrap_message: dict[str, Any] | None = None
    if base_messages and _is_bootstrap_message(base_messages[0]):
        bootstrap_message = base_messages[0]
        base_messages = base_messages[1:]

    if source_tokens <= trigger_tokens:
        return AutoCompactResult(
            messages=source_messages,
            summary=current_summary.strip(),
            compacted=False,
            source_message_count=len(source_messages),
            source_token_estimate=source_tokens,
            compacted_token_estimate=source_tokens,
            used_llm=False,
            anchor_count=0,
        )

    keep_recent = compact_recent_messages()
    if len(base_messages) <= keep_recent:
        compacted_messages = source_messages
        compacted_tokens = estimate_message_tokens(compacted_messages)
        return AutoCompactResult(
            messages=compacted_messages,
            summary=current_summary.strip(),
            compacted=False,
            source_message_count=len(source_messages),
            source_token_estimate=source_tokens,
            compacted_token_estimate=compacted_tokens,
            used_llm=False,
            anchor_count=0,
        )

    compact_source = base_messages[:-keep_recent]
    recent_messages = base_messages[-keep_recent:]
    used_llm = False
    anchors: list[dict[str, Any]] = []
    llm_summary = _summarize_messages_with_llm(llm, compact_source, current_summary)
    if llm_summary is not None:
        incremental, anchors = llm_summary
        used_llm = True
    else:
        incremental = _summarize_messages(compact_source)
    merged_summary = _merge_summaries(
        current_summary,
        incremental,
        token_limit=compact_max_summary_tokens(),
    )
    if anchors:
        merged_summary = _cap_summary_tokens(
            f"{merged_summary}\n\n{_format_anchors(anchors)}",
            compact_max_summary_tokens(),
        )
    compact_summary_message = {
        "role": "assistant",
        "content": f"{COMPACT_SUMMARY_HEADER}\n{merged_summary}" if merged_summary else COMPACT_SUMMARY_HEADER,
        "auto_compact": True,
        "source_messages": len(compact_source),
    }

    compacted_messages: list[dict[str, Any]] = []
    if bootstrap_message is not None:
        compacted_messages.append(bootstrap_message)
    if merged_summary:
        compacted_messages.append(compact_summary_message)
    compacted_messages.extend(recent_messages)

    compacted_tokens = estimate_message_tokens(compacted_messages)
    # If still above target, shrink recent messages conservatively.
    while len(compacted_messages) > 4 and compacted_tokens > target_tokens:
        if compacted_messages and compacted_messages[0].get("auto_compact"):
            break
        compacted_messages.pop(1 if bootstrap_message is not None else 0)
        compacted_tokens = estimate_message_tokens(compacted_messages)

    return AutoCompactResult(
        messages=compacted_messages,
        summary=merged_summary,
        compacted=True,
        source_message_count=len(source_messages),
        source_token_estimate=source_tokens,
        compacted_token_estimate=compacted_tokens,
        used_llm=used_llm,
        anchor_count=len(anchors),
    )


def should_trigger_auto_compact(messages: list[dict[str, Any]]) -> bool:
    source_messages = [msg for msg in messages if isinstance(msg, dict)]
    source_tokens = estimate_message_tokens(source_messages)
    trigger_tokens = int(model_context_window_tokens() * compact_trigger_ratio())
    return source_tokens > trigger_tokens


def inject_compact_summary(prompt: str, summary: str) -> str:
    clean_summary = summary.strip()
    if not clean_summary:
        return prompt
    return (
        f"{prompt}\n\n"
        "[会话压缩记忆]\n"
        f"{clean_summary}\n\n"
        "[使用要求] 若当前问题与上述记忆冲突，请以当前问题和当前文档证据为准。"
    )


def _estimate_skills_tokens(
    skill_context_texts: list[str] | None = None,
) -> int:
    if isinstance(skill_context_texts, list):
        real_total = 0
        for text in skill_context_texts:
            value = str(text or "").strip()
            if not value:
                continue
            real_total += estimate_tokens(value)
        return real_total
    return 0


def _estimate_tools_tokens(tool_specs: list[dict[str, str]] | None = None) -> int:
    if not isinstance(tool_specs, list):
        return 0
    total = 0
    for item in tool_specs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        args_schema = str(item.get("args_schema") or "").strip()
        if not (name or description or args_schema):
            continue
        total += estimate_tokens(f"{name}\n{description}\n{args_schema}")
    return total


def extract_active_skills_from_trace(trace_payload: list[dict[str, Any]]) -> set[str]:
    skills: set[str] = set()
    for item in trace_payload:
        if not isinstance(item, dict):
            continue
        receiver = str(item.get("receiver") or "")
        if receiver != "use_skill":
            continue
        content = str(item.get("content") or "")
        if not content.strip():
            continue
        try:
            payload = json.loads(content)
        except Exception:
            try:
                payload = literal_eval(content)
            except Exception:
                payload = None
        if isinstance(payload, dict):
            skill_name = payload.get("skill_name")
            if isinstance(skill_name, str) and skill_name.strip():
                skills.add(skill_name.strip().lower())
    return skills


def extract_skill_context_texts_from_trace(trace_payload: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for item in trace_payload:
        if not isinstance(item, dict):
            continue
        sender = str(item.get("sender") or "")
        performative = str(item.get("performative") or "")
        if sender != "use_skill" or performative != "tool_result":
            continue
        content = str(item.get("content") or "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        texts.append(content)
    return texts


def build_context_usage_snapshot(
    *,
    messages: list[dict[str, Any]],
    compact_summary: str = "",
    active_skills: set[str] | None = None,
    tool_specs: list[dict[str, str]] | None = None,
    skill_context_texts: list[str] | None = None,
) -> dict[str, Any]:
    max_input = model_context_window_tokens()
    reserve_output = reserved_output_tokens()

    system_tokens = estimate_tokens(PAPER_QA_SYSTEM_PROMPT) + estimate_tokens(ROUTER_INSTRUCTION)
    custom_agents_tokens = (
        estimate_tokens(PLANNER_SYSTEM_PROMPT)
        + estimate_tokens(RESEARCHER_SYSTEM_PROMPT)
        + estimate_tokens(REVIEWER_SYSTEM_PROMPT)
        + estimate_tokens(REACT_SYSTEM_PROMPT)
    )
    memory_tokens = estimate_tokens(compact_summary)
    _ = active_skills  # Deprecated: skills now use progressive runtime-loaded context only.
    skills_tokens = _estimate_skills_tokens(skill_context_texts=skill_context_texts)
    tools_tokens = _estimate_tools_tokens(tool_specs)
    messages_tokens = estimate_message_tokens(messages)
    tools_count = len(tool_specs) if isinstance(tool_specs, list) else 0
    skills_loaded_count = 0
    if isinstance(skill_context_texts, list):
        skills_loaded_count = len({str(item).strip() for item in skill_context_texts if str(item).strip()})

    used_tokens = (
        system_tokens
        + custom_agents_tokens
        + memory_tokens
        + skills_tokens
        + tools_tokens
        + messages_tokens
    )
    autocompact_buffer_tokens = max(
        0,
        int(max_input * compact_trigger_ratio()) - used_tokens,
    )
    free_tokens = max(0, max_input - reserve_output - used_tokens)
    total_visible = used_tokens + autocompact_buffer_tokens + free_tokens
    if total_visible <= 0:
        total_visible = 1

    def _pct(value: int) -> float:
        return round((value / total_visible) * 100.0, 1)

    def _ratio(value: int) -> float:
        return (value / total_visible) * 100.0

    context_order = [
        ("system_prompt", "Primary agent prompt", system_tokens),
        ("custom_agents", "Collaborator prompts", custom_agents_tokens),
        ("memory_files", "Memory files", memory_tokens),
        ("skills", "Skills", skills_tokens),
        ("tools", "Tools", tools_tokens),
        ("messages", "Messages", messages_tokens),
        ("free_space", "Free space", free_tokens),
        ("autocompact_buffer", "Autocompact buffer", autocompact_buffer_tokens),
    ]
    context_segments: list[dict[str, Any]] = []
    cursor = 0.0
    for key, label, tokens in context_order:
        ratio = max(0.0, _ratio(tokens))
        start = cursor
        end = min(100.0, start + ratio)
        context_segments.append(
            {
                "key": key,
                "label": label,
                "tokens": tokens,
                "pct": round(ratio, 2),
                "start_pct": round(start, 2),
                "end_pct": round(end, 2),
            }
        )
        cursor = end

    return {
        "model_window_tokens": max_input,
        "reserved_output_tokens": reserve_output,
        "used_tokens": used_tokens,
        "free_tokens": free_tokens,
        "autocompact_buffer_tokens": autocompact_buffer_tokens,
        "tools_count": tools_count,
        "skills_loaded_count": skills_loaded_count,
        "primary_agent_name": "react_agent",
        "context_view_scope": "primary_agent_first",
        "context_segments": context_segments,
        "breakdown": {
            "system_prompt": {"tokens": system_tokens, "pct": _pct(system_tokens)},
            "custom_agents": {"tokens": custom_agents_tokens, "pct": _pct(custom_agents_tokens)},
            "memory_files": {"tokens": memory_tokens, "pct": _pct(memory_tokens)},
            "skills": {"tokens": skills_tokens, "pct": _pct(skills_tokens)},
            "tools": {"tokens": tools_tokens, "pct": _pct(tools_tokens)},
            "messages": {"tokens": messages_tokens, "pct": _pct(messages_tokens)},
            "free_space": {"tokens": free_tokens, "pct": _pct(free_tokens)},
            "autocompact_buffer": {
                "tokens": autocompact_buffer_tokens,
                "pct": _pct(autocompact_buffer_tokens),
            },
        },
    }
