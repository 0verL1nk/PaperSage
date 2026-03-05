from agent.a2a.replan_policy import (
    has_replan_budget,
    normalize_max_review_rounds,
    review_needs_revision,
)


def test_normalize_max_review_rounds_clamps_minimum():
    assert normalize_max_review_rounds(0) == 1
    assert normalize_max_review_rounds(-3) == 1
    assert normalize_max_review_rounds(2) == 2


def test_review_needs_revision_defaults_to_safe_true():
    assert review_needs_revision("Decision: PASS\nFeedback: good") is False
    assert review_needs_revision("Decision: REVISE\nFeedback: add citations") is True
    assert review_needs_revision("looks good") is True


def test_has_replan_budget():
    assert has_replan_budget(1, 2) is True
    assert has_replan_budget(2, 2) is False
