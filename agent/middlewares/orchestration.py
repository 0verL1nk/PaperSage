"""Orchestration middleware for agent-centric mode activation."""

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from .types import AgentState

logger = logging.getLogger(__name__)


class ComplexityAnalysisResult(BaseModel):
    is_complex: bool = Field(description="Whether the task is complex enough to require planning.")
    needs_team: bool = Field(
        default=False, description="Whether the task likely needs multi-role collaboration."
    )
    reason: str = Field(default="", description="Short explanation for the routing decision.")


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
        self._structured_output_support_cache: dict[str, bool] = {}

    def before_model(  # type: ignore[override]
        self, state: AgentState, runtime: Runtime, config: RunnableConfig | None = None
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
        if llm is None and config:
            llm = config.get("configurable", {}).get("llm")

        on_event = config.get("configurable", {}).get("on_event") if config else None

        if llm is None:
            return None

        # Analyze complexity
        analysis = self._analyze_complexity(messages, llm, on_event)
        logger.info(
            f"Task complexity: is_complex={analysis['is_complex']}, needs_team={analysis.get('needs_team', False)}, reason={analysis['reason']}"
        )

        # Check if plan already exists (from configurable state)
        configurable_state = config.get("configurable", {}).get("state", {}) if config else {}
        existing_plan = configurable_state.get("plan")
        analysis["has_plan"] = existing_plan is not None
        logger.info(f"Plan check: has_plan={analysis['has_plan']}")

        # Store analysis result for wrap_model_call to use
        self._last_analysis = analysis

        update: dict[str, Any] = {"needs_team": analysis.get("needs_team", False)}
        if analysis.get("needs_team", False):
            update["team_handoff"] = {
                "mode": "leader_teammate",
                "reason": analysis.get("reason", ""),
            }
        return update

    @staticmethod
    def _build_complexity_prompt(history_text: str) -> str:
        return f"""分析以下对话的任务复杂度,判断是否需要使用规划工具或团队协作。

对话历史:
{history_text}

判断标准:
- 简单任务: 单一问答、事实查询、简单操作
- 复杂任务: 多步骤分析、需要规划、需要对比研究
- 需要团队: 任务需要多个专业角色协作(如研究+审查、前端+后端、分析+验证等)

只返回JSON格式,不要其他内容:
{{"is_complex": true/false, "needs_team": true/false, "reason": "简短原因"}}"""

    @staticmethod
    def _coerce_analysis_result(result: Any) -> dict[str, Any]:
        if isinstance(result, ComplexityAnalysisResult):
            return result.model_dump()
        if isinstance(result, BaseModel):
            payload = result.model_dump()
        elif isinstance(result, dict):
            payload = result
        else:
            raise TypeError(f"Unexpected complexity analysis result type: {type(result)!r}")

        return ComplexityAnalysisResult.model_validate(payload).model_dump()

    @staticmethod
    def _extract_last_valid_json_object(response_text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        last_valid: dict[str, Any] | None = None
        for idx, char in enumerate(response_text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(response_text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                last_valid = candidate
        if last_valid is None:
            raise ValueError("No valid JSON object found in response")
        return last_valid

    def _invoke_structured_complexity_analysis(
        self, llm: Any, prompt: str
    ) -> dict[str, Any] | None:
        if not hasattr(llm, "with_structured_output"):
            return None
        cache_key = self._llm_capability_cache_key(llm)
        if self._structured_output_support_cache.get(cache_key) is False:
            return None
        structured_llm = llm.with_structured_output(ComplexityAnalysisResult)
        result = structured_llm.invoke(prompt)
        self._structured_output_support_cache[cache_key] = True
        return self._coerce_analysis_result(result)

    @staticmethod
    def _llm_capability_cache_key(llm: Any) -> str:
        class_name = f"{type(llm).__module__}.{type(llm).__qualname__}"
        model_name = str(getattr(llm, "model", "") or getattr(llm, "model_name", "") or "").strip()
        base_url = str(getattr(llm, "base_url", "") or "").strip()
        return f"{class_name}|{model_name}|{base_url}"

    @staticmethod
    def _is_json_mode_unsupported_error(exc: Exception) -> bool:
        message = str(exc or "").lower()
        return "json mode is not supported" in message

    @staticmethod
    def _build_neutral_analysis_result() -> dict[str, Any]:
        return {
            "is_complex": False,
            "needs_team": False,
            "reason": "llm analysis unavailable",
        }

    def _analyze_complexity(
        self, messages: list[Any], llm: Any, on_event: Any = None
    ) -> dict[str, Any]:
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
            msg for msg in messages if hasattr(msg, "type") and msg.type in ("human", "ai")
        ]

        # Format full conversation history
        history_lines = []
        for msg in conversation_msgs:
            role = "用户" if msg.type == "human" else "助手"
            content = str(getattr(msg, "content", ""))
            history_lines.append(f"{role}: {content}")

        history_text = "\n".join(history_lines)
        prompt = self._build_complexity_prompt(history_text)

        # Emit trace event before analysis
        if callable(on_event):
            on_event(
                {
                    "sender": "orchestration",
                    "receiver": "evaluator",
                    "performative": "complexity_analysis",
                    "content": "分析任务复杂度",
                }
            )

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                analysis_result = None
                response_text = ""
                try:
                    analysis_result = self._invoke_structured_complexity_analysis(llm, prompt)
                except Exception as exc:
                    if self._is_json_mode_unsupported_error(exc):
                        self._structured_output_support_cache[
                            self._llm_capability_cache_key(llm)
                        ] = False
                    logger.info(
                        "Structured complexity analysis unavailable, falling back to text JSON: %s",
                        exc,
                    )

                if analysis_result is None:
                    response = llm.invoke(prompt)
                    response_text = (
                        response.content if hasattr(response, "content") else str(response)
                    )
                    parsed = self._extract_last_valid_json_object(response_text.strip())
                    analysis_result = self._coerce_analysis_result(parsed)

                # Emit trace event after analysis
                if callable(on_event):
                    on_event(
                        {
                            "sender": "evaluator",
                            "receiver": "orchestration",
                            "performative": "complexity_result",
                            "content": f"is_complex={analysis_result['is_complex']}, reason={analysis_result['reason']}",
                        }
                    )

                return analysis_result
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Complexity analysis attempt {attempt + 1} failed: {e}, response_text={response_text[:500] if 'response_text' in locals() else 'N/A'}, retrying..."
                    )
                    continue
                logger.warning(
                    f"Failed to analyze complexity with LLM after {max_retries + 1} attempts: {e}, returning neutral result"
                )

        fallback_result = self._build_neutral_analysis_result()

        if callable(on_event):
            on_event(
                {
                    "sender": "orchestration",
                    "receiver": "orchestration",
                    "performative": "complexity_unavailable",
                    "content": "LLM复杂度分析不可用,返回neutral result",
                }
            )

        return fallback_result

    def _inject_guidance(self, messages: list[Any]) -> list[Any]:
        """Inject guidance prompt suggesting orchestration tools.

        Args:
            messages: Original messages

        Returns:
            Messages with guidance injected
        """
        has_plan = self._last_analysis and self._last_analysis.get("has_plan", False)

        needs_team = self._last_analysis and self._last_analysis.get("needs_team", False)

        if needs_team:
            guidance = (
                "【重要提示】这是一个需要多角色协作的复杂任务。\n\n"
                "建议调用 `use_skill('team_mode')` 进入团队协作模式，"
                "系统会引导你完成团队分工、任务派发和结果整合。"
            )
        elif has_plan:
            guidance = (
                "【提示】这是一个复杂任务，你已经创建了执行计划。\n\n"
                "继续使用 write_todos 跟踪进度，系统会返回 scheduler hints 指引下一步。"
            )
        else:
            guidance = (
                "【重要提示】这是一个复杂的多步骤任务。\n\n"
                "建议调用 `use_skill('plan_mode')` 进入规划模式，"
                "系统会引导你完成计划制定、任务拆解和逐步执行。"
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
