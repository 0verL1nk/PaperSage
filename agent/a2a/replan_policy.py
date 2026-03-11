from ..contracts import REVIEW_DECISION_PASS, REVIEW_DECISION_REVISE, get_review_decision
from ..domain.revision_policy import (
    decision_needs_revision,
    has_revision_budget,
    normalize_max_revision_rounds,
)


def normalize_max_review_rounds(max_replan_rounds: int) -> int:
    return normalize_max_revision_rounds(max_replan_rounds, minimum=1)


def review_needs_revision(review_text: str) -> bool:
    decision = get_review_decision(review_text)
    return decision_needs_revision(
        decision,
        pass_tokens=(REVIEW_DECISION_PASS,),
        revise_tokens=(REVIEW_DECISION_REVISE,),
        default_needs_revision=True,
    )


def has_replan_budget(current_round: int, max_rounds: int) -> bool:
    return has_revision_budget(current_round, max_rounds, minimum=1)
