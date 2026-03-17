"""Tool selector middleware configuration."""

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "Your goal is to select the most relevant tools for answering the user's query."
DEFAULT_MAX_TOOLS = 8  # Default maximum number of tools to select


class SimpleToolSelectorMiddleware(AgentMiddleware):
    """Tool selector that uses JSON text parsing instead of structured output."""

    def __init__(
        self,
        *,
        model: Any,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
    ):
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt
        self.max_tools = max_tools
        self.always_include = always_include or []

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Filter tools based on LLM selection before invoking the model."""
        # Get available tools
        if not request.tools or len(request.tools) == 0:
            return handler(request)

        base_tools = [tool for tool in request.tools if not isinstance(tool, dict)]
        available_tools = [tool for tool in base_tools if tool.name not in self.always_include]

        if not available_tools:
            return handler(request)

        # Get last user message
        last_user_message = None
        for message in reversed(request.messages):
            if hasattr(message, "type") and message.type == "human":
                last_user_message = message
                break

        if not last_user_message:
            return handler(request)

        # Select tools using LLM
        selected_tool_names = self._select_tools(last_user_message, available_tools)

        # Add always_include tools
        for name in self.always_include:
            if name not in selected_tool_names:
                selected_tool_names.append(name)

        # Filter tools
        selected_tools = [tool for tool in base_tools if tool.name in selected_tool_names]
        provider_tools = [tool for tool in request.tools if isinstance(tool, dict)]

        logger.info(f"Tool selector: {len(base_tools)} -> {len(selected_tools)} tools")

        return handler(request.override(tools=[*selected_tools, *provider_tools]))

    def _select_tools(self, user_message: Any, available_tools: list[Any]) -> list[str]:
        """Use LLM to select relevant tools."""
        tool_list = "\n".join(f"- {tool.name}: {tool.description}" for tool in available_tools)

        system_msg = self.system_prompt
        if self.max_tools is not None:
            system_msg += (
                f"\nIMPORTANT: Select up to {self.max_tools} most relevant tools. "
                f"List them in order of relevance."
            )

        prompt = f"""{system_msg}

Available tools:
{tool_list}

User query: {user_message.content}

Output JSON format (no markdown):
{{"tools": ["tool1", "tool2", ...]}}"""

        try:
            response = self.model.invoke([{"role": "user", "content": prompt}])
            response_text = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            selected_names = self._extract_json(response_text, available_tools)

            # Limit to max_tools
            if self.max_tools is not None:
                selected_names = selected_names[:self.max_tools]

            return selected_names
        except Exception as e:
            logger.warning(f"Tool selection failed: {e}, using all tools")
            return [tool.name for tool in available_tools]

    def _extract_json(self, response_text: str, available_tools: list[Any]) -> list[str]:
        """Extract tool names from JSON response."""
        valid_names = {tool.name for tool in available_tools}

        # Remove markdown code blocks
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
            tools = data.get("tools", [])
            return [name for name in tools if name in valid_names]
        except Exception:
            # Fallback: extract tool names from text
            return [name for name in valid_names if name in response_text]


def build_tool_selector_middleware(model: Any) -> SimpleToolSelectorMiddleware:
    """Build simple tool selector middleware.

    Args:
        model: The model to use for tool selection.

    Returns:
        Configured SimpleToolSelectorMiddleware instance.
    """
    return SimpleToolSelectorMiddleware(
        model=model,
        max_tools=DEFAULT_MAX_TOOLS,
        always_include=["ask_human", "search_document"],
    )


__all__ = ["build_tool_selector_middleware", "SimpleToolSelectorMiddleware"]
