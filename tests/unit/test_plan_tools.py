"""Unit tests for plan management tools."""

import pytest
from agent.tools.plan_tools import create_plan, read_plan, update_plan, delete_plan


def test_create_plan_new():
    """Test creating a new plan."""
    state = {}
    result = create_plan.func(
        goal="Test goal",
        description="Test description",
        tool_call_id="test_id",
        state=state,
    )

    assert result.update["plan"]["goal"] == "Test goal"
    assert result.update["plan"]["description"] == "Test description"
    assert len(result.update["messages"]) == 1
    assert "Plan created successfully" in result.update["messages"][0].content


def test_create_plan_replace_existing():
    """Test replacing an existing plan."""
    state = {"plan": {"goal": "Old goal", "description": "Old description"}}
    result = create_plan.func(
        goal="New goal",
        description="New description",
        tool_call_id="test_id",
        state=state,
    )

    assert result.update["plan"]["goal"] == "New goal"
    assert result.update["plan"]["description"] == "New description"
    assert "Previous plan replaced" in result.update["messages"][0].content


def test_read_plan_exists():
    """Test reading an existing plan."""
    state = {"plan": {"goal": "Test goal", "description": "Test strategy"}}
    result = read_plan.func(state=state)

    assert "Test goal" in result
    assert "Test strategy" in result


def test_read_plan_not_exists():
    """Test reading when no plan exists."""
    state = {}
    result = read_plan.func(state=state)

    assert "No active plan" in result


def test_update_plan_success():
    """Test updating an existing plan."""
    state = {"plan": {"goal": "Test goal", "description": "Old description"}}
    result = update_plan.func(
        description="New description",
        tool_call_id="test_id",
        state=state,
    )

    assert result.update["plan"]["description"] == "New description"
    assert result.update["plan"]["goal"] == "Test goal"
    assert "Plan updated successfully" in result.update["messages"][0].content


def test_update_plan_no_plan():
    """Test updating when no plan exists."""
    state = {}
    result = update_plan.func(
        description="New description",
        tool_call_id="test_id",
        state=state,
    )

    assert "Error: No active plan" in result.update["messages"][0].content


def test_delete_plan_success():
    """Test deleting an existing plan."""
    state = {"plan": {"goal": "Test goal", "description": "Test description"}}
    result = delete_plan.func(tool_call_id="test_id", state=state)

    assert result.update["plan"] is None
    assert "Plan deleted successfully" in result.update["messages"][0].content


def test_delete_plan_no_plan():
    """Test deleting when no plan exists."""
    state = {}
    result = delete_plan.func(tool_call_id="test_id", state=state)

    assert "No plan to delete" in result.update["messages"][0].content
