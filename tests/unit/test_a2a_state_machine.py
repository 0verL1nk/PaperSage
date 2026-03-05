import pytest

from agent.a2a.state_machine import (
    STATE_COMPLETED,
    STATE_FINALIZING,
    STATE_PLANNING,
    STATE_RESEARCHING,
    STATE_SUBMITTED,
    is_terminal_state,
    transition_state,
)


def test_transition_state_allows_valid_edge():
    assert transition_state(STATE_SUBMITTED, STATE_PLANNING) == STATE_PLANNING
    assert transition_state(STATE_PLANNING, STATE_RESEARCHING) == STATE_RESEARCHING
    assert transition_state(STATE_RESEARCHING, STATE_FINALIZING) == STATE_FINALIZING
    assert transition_state(STATE_FINALIZING, STATE_COMPLETED) == STATE_COMPLETED


def test_transition_state_rejects_invalid_edge():
    with pytest.raises(ValueError):
        transition_state(STATE_SUBMITTED, STATE_FINALIZING)


def test_transition_state_rejects_any_from_terminal():
    with pytest.raises(ValueError):
        transition_state(STATE_COMPLETED, STATE_SUBMITTED)


def test_is_terminal_state():
    assert is_terminal_state(STATE_COMPLETED) is True
    assert is_terminal_state(STATE_SUBMITTED) is False
