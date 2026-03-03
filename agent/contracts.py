import json

from .schemas import (
    PLAN_MAX_STEPS,
    PLAN_MIN_STEPS,
    REVIEW_DECISION_PASS,
    REVIEW_DECISION_REVISE,
    REVIEW_DECISION_UNKNOWN,
    parse_plan_payload,
    parse_review_payload,
)

MIN_PLAN_STEPS = PLAN_MIN_STEPS
MAX_PLAN_STEPS = PLAN_MAX_STEPS


def extract_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    return text[start : end + 1]


def normalize_plan_text(raw_plan: str) -> str:
    json_block = extract_json_block(raw_plan)
    if json_block:
        try:
            payload = json.loads(json_block)
            normalized_steps = parse_plan_payload(payload)
            if normalized_steps:
                return "\n".join(
                    f"{index}. {step}"
                    for index, step in enumerate(normalized_steps, start=1)
                )
        except Exception:
            pass
        if raw_plan.strip().startswith("{") and raw_plan.strip().endswith("}"):
            return ""

    lines = [line.strip("-•* \t") for line in raw_plan.splitlines() if line.strip()]
    if len(lines) >= MIN_PLAN_STEPS:
        normalized = [line for line in lines[:MAX_PLAN_STEPS] if line]
        if normalized:
            return "\n".join(
                f"{index}. {line}" for index, line in enumerate(normalized, start=1)
            )
    return raw_plan.strip()


def normalize_review_text(raw_review: str) -> str:
    json_block = extract_json_block(raw_review)
    if not json_block:
        return raw_review
    try:
        payload = json.loads(json_block)
    except Exception:
        return raw_review
    parsed = parse_review_payload(payload)
    if parsed is None:
        return raw_review
    normalized_decision, cleaned_feedback = parsed
    return f"Decision: {normalized_decision}\nFeedback: {cleaned_feedback}"


def get_review_decision(review_text: str) -> str:
    normalized = review_text.upper()
    if "DECISION: PASS" in normalized:
        return REVIEW_DECISION_PASS
    if "DECISION: REVISE" in normalized or "REVISE" in normalized:
        return REVIEW_DECISION_REVISE
    return REVIEW_DECISION_UNKNOWN
