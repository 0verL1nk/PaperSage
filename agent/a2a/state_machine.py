STATE_SUBMITTED = "submitted"
STATE_PLANNING = "planning"
STATE_RESEARCHING = "researching"
STATE_REVIEWING = "reviewing"
STATE_REPLANNING = "replanning"
STATE_FINALIZING = "finalizing"
STATE_COMPLETED = "completed"

ALLOWED_STATE_TRANSITIONS = {
    STATE_SUBMITTED: {STATE_PLANNING, STATE_RESEARCHING},
    STATE_PLANNING: {STATE_RESEARCHING},
    STATE_RESEARCHING: {STATE_REVIEWING, STATE_FINALIZING},
    STATE_REVIEWING: {STATE_REPLANNING, STATE_FINALIZING},
    STATE_REPLANNING: {STATE_RESEARCHING},
    STATE_FINALIZING: {STATE_COMPLETED},
    STATE_COMPLETED: set(),
}


def transition_state(current_state: str, next_state: str) -> str:
    allowed = ALLOWED_STATE_TRANSITIONS.get(current_state, set())
    if next_state not in allowed:
        raise ValueError(f"Invalid internal A2A state transition: {current_state} -> {next_state}")
    return next_state


def is_terminal_state(state: str) -> bool:
    return state == STATE_COMPLETED
