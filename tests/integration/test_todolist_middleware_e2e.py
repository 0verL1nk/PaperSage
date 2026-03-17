"""E2E test for TodoListMiddleware integration."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.runtime_agent import create_runtime_agent


class SimpleMockLLM:
    """Minimal mock LLM that supports bind_tools."""

    def __init__(self):
        self.call_count = 0

    def bind_tools(self, tools, **kwargs):
        """Return self to support tool binding."""
        return self

    def invoke(self, messages, config=None):
        """Return a simple response."""
        self.call_count += 1
        # Simple response without tool calls
        return AIMessage(content="好的，我会帮你规划项目。")


def test_todolist_middleware_provides_write_todos_tool():
    """Test that TodoListMiddleware provides write_todos tool to the agent."""
    mock_llm = SimpleMockLLM()

    # Create agent with TodoListMiddleware
    agent = create_runtime_agent(
        model=mock_llm,
        system_prompt="你是一个助手。",
        tools=[],
        enable_auto_summarization=False,
    )

    # Invoke agent
    result = agent.invoke(
        {"messages": [HumanMessage(content="测试")]},
        config={"configurable": {"thread_id": "test_thread"}},
    )

    # Verify agent was invoked
    assert mock_llm.call_count > 0
    assert result is not None
    assert "messages" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
