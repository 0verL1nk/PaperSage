from agent.domain.revision_policy import (
    decision_needs_revision,
    failure_needs_revision,
    has_revision_budget,
    normalize_max_revision_rounds,
)


def test_normalize_max_revision_rounds_clamps_to_minimum() -> None:
    assert normalize_max_revision_rounds(0) == 1
    assert normalize_max_revision_rounds(-2) == 1
    assert normalize_max_revision_rounds(3) == 3
    assert normalize_max_revision_rounds(0, minimum=2) == 2


def test_has_revision_budget_uses_normalized_limits() -> None:
    assert has_revision_budget(0, 0) is True
    assert has_revision_budget(1, 1) is False
    assert has_revision_budget(2, 3) is True
    assert has_revision_budget(1, 1, minimum=2) is True


def test_decision_needs_revision_defaults_to_safe_true() -> None:
    assert decision_needs_revision("pass") is False
    assert decision_needs_revision("REVISE") is True
    assert decision_needs_revision("unknown") is True
    assert decision_needs_revision("unknown", default_needs_revision=False) is False


def test_failure_needs_revision_filters_non_revisable_reasons() -> None:
    assert failure_needs_revision("missing_evidence") is True
    assert failure_needs_revision("plan_cycle_guard_triggered") is False
    assert failure_needs_revision("manual_abort") is False
    assert failure_needs_revision("") is False
