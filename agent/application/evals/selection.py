from __future__ import annotations

from .contracts import AgentEvalCase


def select_eval_cases(
    cases: list[AgentEvalCase],
    *,
    case_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[AgentEvalCase]:
    selected = list(cases)
    if case_ids:
        normalized_ids = {str(item).strip() for item in case_ids if str(item).strip()}
        selected = [item for item in selected if item.case_id in normalized_ids]
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected
