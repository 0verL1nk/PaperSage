from agent.schemas import (
    REVIEW_DECISION_PASS,
    REVIEW_DECISION_REVISE,
    parse_plan_payload,
    parse_review_payload,
)


def test_parse_plan_payload_valid_steps():
    payload = {"steps": ["collect evidence", "draft answer", "review"]}
    assert parse_plan_payload(payload) == [
        "collect evidence",
        "draft answer",
        "review",
    ]


def test_parse_plan_payload_invalid_steps():
    payload = {"steps": ["only one"]}
    assert parse_plan_payload(payload) == []


def test_parse_review_payload_normalizes_decision():
    payload = {"decision": "pass", "feedback": "ready to ship"}
    assert parse_review_payload(payload) == (REVIEW_DECISION_PASS, "ready to ship")


def test_parse_review_payload_invalid_decision():
    payload = {"decision": "HOLD", "feedback": "needs more data"}
    assert parse_review_payload(payload) is None


def test_parse_review_payload_revise():
    payload = {"decision": "REVISE", "feedback": "add stronger evidence"}
    assert parse_review_payload(payload) == (
        REVIEW_DECISION_REVISE,
        "add stronger evidence",
    )
