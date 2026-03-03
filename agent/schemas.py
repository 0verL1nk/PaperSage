PLAN_MIN_STEPS = 2
PLAN_MAX_STEPS = 6

REVIEW_DECISION_PASS = "PASS"
REVIEW_DECISION_REVISE = "REVISE"
REVIEW_DECISION_UNKNOWN = "UNKNOWN"
REVIEW_ALLOWED_DECISIONS = {REVIEW_DECISION_PASS, REVIEW_DECISION_REVISE}


def parse_plan_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return []
    normalized_steps = [
        str(step).strip() for step in steps if isinstance(step, str) and step.strip()
    ]
    if len(normalized_steps) < PLAN_MIN_STEPS:
        return []
    return normalized_steps[:PLAN_MAX_STEPS]


def parse_review_payload(payload: object) -> tuple[str, str] | None:
    if not isinstance(payload, dict):
        return None
    decision = payload.get("decision")
    feedback = payload.get("feedback")
    if not isinstance(decision, str) or not isinstance(feedback, str):
        return None
    normalized_decision = decision.strip().upper()
    normalized_feedback = feedback.strip()
    if not normalized_feedback:
        return None
    if normalized_decision not in REVIEW_ALLOWED_DECISIONS:
        return None
    return normalized_decision, normalized_feedback
