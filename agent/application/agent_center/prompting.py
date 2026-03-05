from typing import Any

from agent.settings import load_agent_settings


def with_language_hint(prompt: str, detect_language_fn) -> str:
    detected = detect_language_fn(prompt)
    if detected == "en":
        return f"{prompt}\n\n[Response language requirement: answer in English.]"
    if detected == "zh":
        return f"{prompt}\n\n[回答语言要求：请使用中文回答。]"
    return prompt


def build_routing_context(
    messages: list[dict[str, Any]],
    compact_summary: str,
) -> str:
    settings = load_agent_settings()
    recent_limit = max(1, settings.agent_routing_context_recent_limit)
    max_chars = max(256, settings.agent_routing_context_max_chars)
    item_max_chars = max(40, settings.agent_routing_context_item_max_chars)
    reason_max_chars = max(40, settings.agent_routing_context_reason_max_chars)
    roles_preview_count = max(1, settings.agent_routing_context_roles_preview_count)
    parts: list[str] = []
    summary = compact_summary.strip()
    if summary:
        parts.append("[会话压缩摘要]")
        parts.append(summary)

    normalized_messages = [item for item in messages if isinstance(item, dict)]
    recent_messages = normalized_messages[-recent_limit:]
    if recent_messages:
        parts.append("[最近对话片段]")
        for item in recent_messages:
            role = str(item.get("role") or "unknown")
            content = item.get("content")
            if not isinstance(content, str):
                content = str(content or "")
            compact_text = " ".join(content.split())
            if compact_text:
                parts.append(f"- {role}: {compact_text[:item_max_chars]}")

    for item in reversed(normalized_messages):
        if str(item.get("role") or "") != "assistant":
            continue
        policy = item.get("policy_decision")
        team = item.get("team_execution")
        if isinstance(policy, dict):
            parts.append("[上一轮执行策略]")
            parts.append(
                f"plan={bool(policy.get('plan_enabled'))}, "
                f"team={bool(policy.get('team_enabled'))}, "
                f"reason={str(policy.get('reason') or '')[:reason_max_chars]}"
            )
        if isinstance(team, dict):
            parts.append("[上一轮团队执行]")
            parts.append(
                f"enabled={bool(team.get('enabled'))}, "
                f"rounds={int(team.get('rounds') or 0)}, "
                f"roles={','.join(str(x) for x in (team.get('roles') or [])[:roles_preview_count])}"
            )
        break

    merged = "\n".join(parts).strip()
    if len(merged) <= max_chars:
        return merged
    return merged[-max_chars:]
