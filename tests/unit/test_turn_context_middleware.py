from langchain_core.messages import HumanMessage, SystemMessage

from agent.middlewares.turn_context import TurnContextMiddleware


def test_turn_context_middleware_injects_system_context_before_user_message() -> None:
    middleware = TurnContextMiddleware()
    state = {"messages": [HumanMessage(content="真实用户问题")]}

    update = middleware.before_model(
        state,
        runtime=None,
        config={
            "configurable": {
                "turn_context": {
                    "response_language": "zh",
                    "memory_items": [{"memory_type": "semantic", "content": "偏好简洁回答"}],
                }
            }
        },
    )

    assert isinstance(update, dict)
    messages = update["messages"]
    assert isinstance(messages[0], SystemMessage)
    assert "请使用中文回答" in str(messages[0].content)
    assert "Relevant long-term memory" in str(messages[0].content)
    assert messages[1].content == "真实用户问题"


def test_turn_context_middleware_skips_empty_context() -> None:
    middleware = TurnContextMiddleware()
    state = {"messages": [HumanMessage(content="hello")]}

    update = middleware.before_model(
        state,
        runtime=None,
        config={"configurable": {"turn_context": {}}},
    )

    assert update is None
