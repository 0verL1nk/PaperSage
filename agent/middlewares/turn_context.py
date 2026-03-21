"""Dynamic per-turn system context injection."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from .types import AgentState


class TurnContextMiddleware(AgentMiddleware):
    """Inject structured per-turn context as a system message."""

    def before_model(  # type: ignore[override]
        self,
        state: AgentState,
        runtime: Runtime,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not hasattr(last_msg, "type") or last_msg.type != "human":
            return None

        configurable = config.get("configurable", {}) if config else {}
        turn_context = configurable.get("turn_context")
        system_content = self._build_system_content(turn_context)
        if not system_content:
            return None
        return {"messages": messages[:-1] + [SystemMessage(content=system_content), last_msg]}

    @staticmethod
    def _build_system_content(turn_context: Any) -> str:
        if not isinstance(turn_context, dict):
            return ""

        lines: list[str] = []
        response_language = str(turn_context.get("response_language") or "").strip().lower()
        if response_language == "en":
            lines.append("If the user does not explicitly request another language, answer in English.")
        elif response_language == "zh":
            lines.append("如果用户没有明确要求其他语言，请使用中文回答。")

        memory_items = turn_context.get("memory_items")
        if isinstance(memory_items, list):
            memory_lines: list[str] = []
            for item in memory_items:
                if not isinstance(item, dict):
                    continue
                memory_type = str(item.get("memory_type") or "episodic").strip().lower() or "episodic"
                content = " ".join(str(item.get("content") or "").split()).strip()
                if not content:
                    continue
                memory_lines.append(f"- ({memory_type}) {content}")
            if memory_lines:
                lines.append("Relevant long-term memory:")
                lines.extend(memory_lines)
                lines.append(
                    "If memory conflicts with the current user request or current evidence, prefer the current request and current evidence."
                )

        return "\n".join(lines).strip()


turn_context_middleware = TurnContextMiddleware()

__all__ = ["TurnContextMiddleware", "turn_context_middleware"]
