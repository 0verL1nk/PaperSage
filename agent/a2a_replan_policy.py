from .contracts import REVIEW_DECISION_PASS, REVIEW_DECISION_REVISE, get_review_decision


def normalize_max_review_rounds(max_replan_rounds: int) -> int:
    return max(1, int(max_replan_rounds))


def review_needs_revision(review_text: str) -> bool:
    decision = get_review_decision(review_text)
    if decision == REVIEW_DECISION_PASS:
        return False
    if decision == REVIEW_DECISION_REVISE:
        return True
    # Reviewer output not following contract should be treated as revise to avoid false PASS.
    return True


def has_replan_budget(current_round: int, max_rounds: int) -> bool:
    return current_round < normalize_max_review_rounds(max_rounds)
