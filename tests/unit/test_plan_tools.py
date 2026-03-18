"""Unit tests for plan management tools."""

from agent.tools.plan_tools import read_plan, write_plan


def test_write_plan_create_new():
    """Test creating a new plan."""
    state = {}
    result = write_plan.func(
        goal="Test goal",
        description="Test description",
        tool_call_id="test_id",
        state=state,
    )

    assert result.update["plan"]["goal"] == "Test goal"
    assert result.update["plan"]["description"] == "Test description"
    assert len(result.update["messages"]) == 1
    assert "Plan created" in result.update["messages"][0].content


def test_write_plan_create_without_goal():
    """Test creating a new plan without goal fails."""
    state = {}
    result = write_plan.func(
        goal=None,
        description="Test description",
        tool_call_id="test_id",
        state=state,
    )

    assert "Error: goal required" in result.update["messages"][0].content


def test_write_plan_update_existing():
    """Test updating an existing plan."""
    state = {"plan": {"goal": "Old goal", "description": "Old description"}}
    result = write_plan.func(
        goal=None,
        description="New description",
        tool_call_id="test_id",
        state=state,
    )

    assert result.update["plan"]["goal"] == "Old goal"
    assert result.update["plan"]["description"] == "New description"
    assert "Plan updated" in result.update["messages"][0].content


def test_write_plan_replace_with_new_goal():
    """Test replacing plan with new goal."""
    state = {"plan": {"goal": "Old goal", "description": "Old description"}}
    result = write_plan.func(
        goal="New goal",
        description="New description",
        tool_call_id="test_id",
        state=state,
    )

    assert result.update["plan"]["goal"] == "New goal"
    assert result.update["plan"]["description"] == "New description"


def test_write_plan_delete():
    """Test deleting plan with empty description."""
    state = {"plan": {"goal": "Test goal", "description": "Test description"}}
    result = write_plan.func(
        goal=None,
        description="",
        tool_call_id="test_id",
        state=state,
    )

    assert result.update["plan"] is None
    assert "Plan deleted" in result.update["messages"][0].content


def test_write_plan_delete_no_plan():
    """Test deleting when no plan exists."""
    state = {}
    result = write_plan.func(
        goal=None,
        description="",
        tool_call_id="test_id",
        state=state,
    )

    assert "No plan to delete" in result.update["messages"][0].content


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

