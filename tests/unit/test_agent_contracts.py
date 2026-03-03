from agent.contracts import (
    REVIEW_DECISION_PASS,
    REVIEW_DECISION_REVISE,
    REVIEW_DECISION_UNKNOWN,
    get_review_decision,
    normalize_plan_text,
    normalize_review_text,
)


def test_normalize_plan_text_from_json_steps():
    raw = '{"steps":["collect evidence","draft answer","review"]}'
    result = normalize_plan_text(raw)
    assert result.startswith("1. collect evidence")
    assert "2. draft answer" in result


def test_normalize_plan_text_invalid_json_returns_empty():
    raw = '{"steps":["single"]}'
    assert normalize_plan_text(raw) == ""


def test_normalize_review_text_from_json():
    raw = '{"decision":"REVISE","feedback":"add stronger citations"}'
    result = normalize_review_text(raw)
    assert result == "Decision: REVISE\nFeedback: add stronger citations"


def test_get_review_decision_variants():
    assert get_review_decision("Decision: PASS\nFeedback: ok") == REVIEW_DECISION_PASS
    assert get_review_decision("Decision: REVISE\nFeedback: fix") == REVIEW_DECISION_REVISE
    assert get_review_decision("looks good") == REVIEW_DECISION_UNKNOWN
