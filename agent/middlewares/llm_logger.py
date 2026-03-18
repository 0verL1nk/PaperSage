"""LLM 输入输出日志记录 middleware"""
import json
import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class LLMLoggerMiddleware(AgentMiddleware):
    """记录完整的 LLM 输入和输出"""

    def before_model(
        self, state: AgentState, runtime: Runtime, config: RunnableConfig | None = None
    ) -> dict[str, Any] | None:
        """记录 LLM 输入"""
        messages = state.get("messages", [])
        if not messages:
            return None

        try:
            # 提取最后几条消息作为输入上下文
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            input_log = []
            for msg in recent_messages:
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
                input_log.append({"role": role, "content": str(content)[:500]})

            logger.info(f"LLM_INPUT: {json.dumps(input_log, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"Failed to log LLM input: {e}")

        return None

    def after_model(
        self, state: AgentState, runtime: Runtime, config: RunnableConfig | None = None
    ) -> dict[str, Any] | None:
        """记录 LLM 输出"""
        messages = state.get("messages", [])
        if not messages:
            return None

        try:
            last_msg = messages[-1]
            role = getattr(last_msg, "type", "unknown")
            content = getattr(last_msg, "content", "")
            tool_calls = getattr(last_msg, "tool_calls", None)

            output_log = {
                "role": role,
                "content": str(content)[:1000],
                "has_tool_calls": bool(tool_calls),
            }

            if tool_calls:
                output_log["tool_calls"] = [
                    {"name": tc.get("name"), "args": str(tc.get("args", ""))[:200]}
                    for tc in (tool_calls if isinstance(tool_calls, list) else [])
                ]

            logger.info(f"LLM_OUTPUT: {json.dumps(output_log, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"Failed to log LLM output: {e}")

        return None


llm_logger_middleware = LLMLoggerMiddleware()
