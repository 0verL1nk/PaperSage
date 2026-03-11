from collections.abc import Iterable


def normalize_max_revision_rounds(max_rounds: int, *, minimum: int = 1) -> int:
    return max(int(minimum), int(max_rounds))


def has_revision_budget(
    current_round: int,
    max_rounds: int,
    *,
    minimum: int = 1,
) -> bool:
    normalized_max = normalize_max_revision_rounds(max_rounds, minimum=minimum)
    return int(current_round) < normalized_max


def decision_needs_revision(
    decision: str,
    *,
    pass_tokens: Iterable[str] = ("PASS",),
    revise_tokens: Iterable[str] = ("REVISE",),
    default_needs_revision: bool = True,
) -> bool:
    normalized = str(decision or "").strip().upper()
    normalized_pass = {str(item).strip().upper() for item in pass_tokens if str(item).strip()}
    normalized_revise = {str(item).strip().upper() for item in revise_tokens if str(item).strip()}
    if normalized and normalized in normalized_pass:
        return False
    if normalized and normalized in normalized_revise:
        return True
    return bool(default_needs_revision)


def failure_needs_revision(
    failure_reason: str | None,
    *,
    non_revisable_reasons: Iterable[str] = (
        "plan_cycle_guard_triggered",
        "policy_switched_to_non_team",
        "manual_abort",
    ),
) -> bool:
    normalized = str(failure_reason or "").strip().lower()
    if not normalized:
        return False
    blocked = {str(item).strip().lower() for item in non_revisable_reasons if str(item).strip()}
    return normalized not in blocked
