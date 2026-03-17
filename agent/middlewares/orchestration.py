"""Orchestration middleware for agent-centric mode activation."""

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from .types import AgentState

logger = logging.getLogger(__name__)


class OrchestrationMiddleware(AgentMiddleware):
    """Middleware that analyzes context and suggests orchestration tools.

    Uses LLM to detect complex tasks and injects guidance prompts
    suggesting the use of plan or team tools, without forcing decisions.
    """

    def __init__(self, llm: Any | None = None):
        """Initialize middleware.

        Args:
            llm: Language model for complexity analysis (optional)
        """
        super().__init__()
        self.llm = llm
        self._last_analysis: dict[str, Any] | None = None

    def before_model(  # type: ignore[override]
        self, state: AgentState, runtime: Runtime, config: RunnableConfig
    ) -> dict[str, Any] | None:
        """Analyze complexity and emit trace events before model invocation."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        is_user_input = hasattr(last_msg, "type") and last_msg.type == "human"

        if not is_user_input:
            return None

        # Get LLM and on_event from config
        llm = self.llm
        if llm is None:
            llm = config.get("configurable", {}).get("llm")

        on_event = config.get("configurable", {}).get("on_event")

        if llm is None:
            return None

        # Analyze complexity
        analysis = self._analyze_complexity(messages, llm, on_event)
        logger.info(f"Task complexity: is_complex={analysis['is_complex']}, reason={analysis['reason']}")

        # Check if plan already exists (from configurable state)
        configurable_state = config.get("configurable", {}).get("state", {})
        existing_plan = configurable_state.get("plan")
        analysis["has_plan"] = existing_plan is not None
        logger.info(f"Plan check: has_plan={analysis['has_plan']}")

        # Store analysis result for wrap_model_call to use
        self._last_analysis = analysis

        # If complex task, inject tool activation message to expose orchestration tools
        if analysis.get("is_complex"):
            activation_msg = self._create_tool_activation_message()
            return {"messages": [activation_msg]}

        return None

    def _analyze_complexity(self, messages: list[Any], llm: Any, on_event: Any = None) -> dict[str, Any]:
        """Use LLM to analyze task complexity with full context.

        Args:
            messages: Full conversation messages
            llm: Language model to use
            on_event: Optional callback for trace events

        Returns:
            Dict with 'is_complex' (bool) and 'reason' (str)
        """
        # Filter out system messages, only keep user and assistant messages
        conversation_msgs = [
            msg for msg in messages
            if hasattr(msg, "type") and msg.type in ("human", "ai")
        ]

        # Format full conversation history
        history_lines = []
        for msg in conversation_msgs:
            role = "用户" if msg.type == "human" else "助手"
            content = str(getattr(msg, "content", ""))
            history_lines.append(f"{role}: {content}")

        history_text = "\n".join(history_lines)

        prompt = f"""分析以下对话的任务复杂度,判断是否需要使用规划工具或团队协作。

对话历史:
{history_text}

判断标准:
- 简单任务: 单一问答、事实查询、简单操作
- 复杂任务: 多步骤分析、需要规划、需要对比研究、需要团队协作

只返回JSON格式,不要其他内容:
{{"is_complex": true/false, "reason": "简短原因"}}"""

        # Emit trace event before analysis
        if callable(on_event):
            on_event({
                "sender": "orchestration",
                "receiver": "evaluator",
                "performative": "complexity_analysis",
                "content": "分析任务复杂度",
            })

        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            result = json.loads(response_text)
            analysis_result = {
                "is_complex": bool(result.get("is_complex", False)),
                "reason": str(result.get("reason", ""))
            }

            # Emit trace event after analysis
            if callable(on_event):
                on_event({
                    "sender": "evaluator",
                    "receiver": "orchestration",
                    "performative": "complexity_result",
                    "content": f"is_complex={analysis_result['is_complex']}, reason={analysis_result['reason']}",
                })

            return analysis_result
        except Exception as e:
            logger.warning(f"Failed to analyze complexity with LLM: {e}, using fallback")
            # Fallback: simple heuristic on last message
            last_content = str(getattr(messages[-1], "content", ""))
            fallback_result = {
                "is_complex": len(last_content) > 200 or any(kw in last_content for kw in ["步骤", "规划", "对比", "分析"]),
                "reason": "fallback heuristic"
            }

            if callable(on_event):
                on_event({
                    "sender": "orchestration",
                    "receiver": "orchestration",
                    "performative": "complexity_fallback",
                    "content": f"使用fallback: is_complex={fallback_result['is_complex']}",
                })

            return fallback_result

    def _create_tool_activation_message(self) -> ToolMessage:
        """Create a tool activation message to expose orchestration tools."""
        orchestration_tools = [
            {"tool_name": "create_plan", "description": "创建执行计划"},
            {"tool_name": "read_plan", "description": "读取执行计划"},
            {"tool_name": "update_plan", "description": "更新执行计划"},
            {"tool_name": "delete_plan", "description": "删除执行计划"},
            {"tool_name": "write_todos", "description": "写入待办任务列表"},
            {"tool_name": "activate_team_mode", "description": "激活团队协作模式"},
        ]

        content = json.dumps(
            {
                "type": "tool_search_result",
                "query": "orchestration planning tools",
                "tools": orchestration_tools,
            },
            ensure_ascii=False,
        )

        return ToolMessage(content=content, name="search_tools", tool_call_id="auto_orchestration")

    def _inject_guidance(self, messages: list[Any]) -> list[Any]:
        """Inject guidance prompt suggesting orchestration tools.

        Args:
            messages: Original messages

        Returns:
            Messages with guidance injected
        """
        has_plan = self._last_analysis and self._last_analysis.get("has_plan", False)

        if has_plan:
            guidance = (
                "【提示】这是一个复杂任务,你已经创建了执行计划。\n\n"
                "建议使用 write_todos 工具跟踪任务进度,或使用 read_plan 查看计划。"
            )
        else:
            guidance = (
                "【重要提示】这是一个复杂的多步骤任务,需要使用规划工具:\n\n"
                "1. 首先使用 create_plan 工具创建执行计划,明确任务步骤和策略\n"
                "2. 使用 write_todos 工具跟踪任务进度\n"
                "3. 如需团队协作,使用 activate_team_mode 工具\n\n"
                "请先调用相应的工具进行规划,不要直接回答。"
            )

        result = list(messages)
        result.insert(-1, SystemMessage(content=guidance))
        return result

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Wrap model call to inject orchestration guidance for complex tasks.

        Args:
            request: Model request
            handler: Next handler in chain

        Returns:
            Model response
        """
        # Check if we have a complexity analysis result
        if self._last_analysis and self._last_analysis.get("is_complex"):
            messages = getattr(request, "messages", [])
            if messages:
                request.messages = self._inject_guidance(messages)
            # Reset analysis after use
            self._last_analysis = None

        return handler(request)
