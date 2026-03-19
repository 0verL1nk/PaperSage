from __future__ import annotations

from dataclasses import replace

from agent.domain.orchestration import TeamRunState
from agent.orchestration.state_machine import transition_team_run_state


class LeaderTeammateCoordinator:
    """Apply reviewer decisions to a structured Leader-teammate run."""

    def apply_review_decision(
        self,
        run_state: TeamRunState,
        decision: str,
    ) -> TeamRunState:
        normalized = str(decision or "").strip().lower()
        tagged_state = replace(run_state, review_decision=normalized)
        if normalized in {"pass", "approved", "accept", "accepted"}:
            return transition_team_run_state(
                tagged_state,
                "completed",
                review_decision=normalized,
            )
        if normalized in {"revise", "revision", "replan", "retry"}:
            return transition_team_run_state(
                tagged_state,
                "replanning",
                review_decision=normalized,
            )
        return transition_team_run_state(
            tagged_state,
            "failed",
            review_decision=normalized,
            error=f"review_decision={normalized or 'unknown'}",
        )
