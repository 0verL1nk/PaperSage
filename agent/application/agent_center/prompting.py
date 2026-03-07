from typing import Any

from agent.settings import load_agent_settings

ROUTING_LAYOUT_VERSION = "RTv1"
ROUTING_SUMMARY_TAG = "S"
ROUTING_POLICY_TAG = "P"
ROUTING_TEAM_TAG = "T"
ROUTING_HISTORY_TAG = "H"


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
    max_chars = max(64, settings.agent_routing_context_max_chars)
    item_max_chars = max(40, settings.agent_routing_context_item_max_chars)
    reason_max_chars = max(40, settings.agent_routing_context_reason_max_chars)
    roles_preview_count = max(1, settings.agent_routing_context_roles_preview_count)
    normalized_messages = [item for item in messages if isinstance(item, dict)]
    summary_text = _truncate_tail(compact_summary.strip(), max(80, int(max_chars * 0.45)))
    policy_line, team_line = _latest_runtime_snapshot(
        messages=normalized_messages,
        reason_max_chars=reason_max_chars,
        roles_preview_count=roles_preview_count,
    )
    recent_lines = _recent_dialog_lines(
        messages=normalized_messages,
        recent_limit=recent_limit,
        item_max_chars=item_max_chars,
    )

    merged = _assemble_routing_context(
        summary_text=summary_text,
        policy_line=policy_line,
        team_line=team_line,
        recent_lines=recent_lines,
    )
    if len(merged) <= max_chars:
        return merged

    while recent_lines and len(merged) > max_chars:
        recent_lines.pop(0)
        merged = _assemble_routing_context(
            summary_text=summary_text,
            policy_line=policy_line,
            team_line=team_line,
            recent_lines=recent_lines,
        )
    if len(merged) <= max_chars:
        return merged

    if team_line and len(merged) > max_chars:
        team_line = ""
        merged = _assemble_routing_context(
            summary_text=summary_text,
            policy_line=policy_line,
            team_line=team_line,
            recent_lines=recent_lines,
        )
    if policy_line and len(merged) > max_chars:
        policy_line = ""
        merged = _assemble_routing_context(
            summary_text=summary_text,
            policy_line=policy_line,
            team_line=team_line,
            recent_lines=recent_lines,
        )
    if len(merged) <= max_chars:
        return merged

    summary_text = _truncate_tail(summary_text, max(40, max_chars // 3))
    merged = _assemble_routing_context(
        summary_text=summary_text,
        policy_line=policy_line,
        team_line=team_line,
        recent_lines=recent_lines,
    )
    return _truncate_tail(merged, max_chars)


def _latest_runtime_snapshot(
    *,
    messages: list[dict[str, Any]],
    reason_max_chars: int,
    roles_preview_count: int,
) -> tuple[str, str]:
    for item in reversed(messages):
        if str(item.get("role") or "") != "assistant":
            continue
        policy_line = ""
        team_line = ""
        policy = item.get("policy_decision")
        team = item.get("team_execution")
        if isinstance(policy, dict):
            reason = _collapse_text(str(policy.get("reason") or ""), reason_max_chars)
            policy_line = (
                f"plan={bool(policy.get('plan_enabled'))},"
                f"team={bool(policy.get('team_enabled'))},"
                f"reason={reason}"
            )
        if isinstance(team, dict):
            roles = ",".join(str(x) for x in (team.get("roles") or [])[:roles_preview_count])
            team_line = (
                f"enabled={bool(team.get('enabled'))},"
                f"rounds={int(team.get('rounds') or 0)},"
                f"roles={roles}"
            )
        return policy_line, team_line
    return "", ""


def _recent_dialog_lines(
    *,
    messages: list[dict[str, Any]],
    recent_limit: int,
    item_max_chars: int,
) -> list[str]:
    recent_messages = messages[-recent_limit:]
    lines: list[str] = []
    for idx, item in enumerate(recent_messages, start=1):
        role = str(item.get("role") or "unknown")
        compact_text = _collapse_text(item.get("content"), item_max_chars)
        if compact_text:
            lines.append(f"{ROUTING_HISTORY_TAG}{idx}:{role}:{compact_text}")
    return lines


def _collapse_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    compact = " ".join(text.split()).strip()
    if not compact:
        return ""
    return _truncate_tail(compact, max_chars)


def _assemble_routing_context(
    *,
    summary_text: str,
    policy_line: str,
    team_line: str,
    recent_lines: list[str],
) -> str:
    parts: list[str] = [
        ROUTING_LAYOUT_VERSION,
        f"{ROUTING_SUMMARY_TAG}:{summary_text or '-'}",
        f"{ROUTING_POLICY_TAG}:{policy_line or '-'}",
        f"{ROUTING_TEAM_TAG}:{team_line or '-'}",
    ]
    for line in recent_lines:
        if line.strip():
            parts.append(line)
    return "\n".join(parts)


def _truncate_tail(text: str, limit: int) -> str:
    value = str(text or "")
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return f"{value[: limit - 3]}..."
