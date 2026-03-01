import json
from typing import Any

from .multi_agent_a2a import WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN, WORKFLOW_REACT
from .stream import extract_result_text


WORKFLOW_LABELS = {
    WORKFLOW_REACT: "ReAct（Tool+Memory）",
    WORKFLOW_PLAN_ACT: "Plan-Act（A2A协调）",
    WORKFLOW_PLAN_ACT_REPLAN: "Plan-Act-RePlan（A2A协调）",
}


ROUTER_INSTRUCTION = """
你是工作流路由器。请在以下模式中选择最合适的一种：
- react: 简单问答、单跳检索、快速事实确认
- plan_act: 中等复杂任务，需要先规划再执行
- plan_act_replan: 高复杂度任务，需要规划-执行-复核-重规划

仅返回严格 JSON，不要额外文本：
{"mode":"react|plan_act|plan_act_replan","reason":"简短原因","confidence":0.0}
"""


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _extract_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    return text[start : end + 1]


def _normalize_mode(mode: str) -> str | None:
    normalized = mode.strip().lower()
    if normalized in {WORKFLOW_REACT, WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN}:
        return normalized
    return None


def _route_with_llm(
    prompt: str,
    coordinator: Any,
) -> tuple[str, str] | None:
    try:
        result = coordinator.planner_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{ROUTER_INSTRUCTION}\n用户问题：{prompt}",
                    }
                ]
            },
            config={"configurable": {"thread_id": f"{coordinator.session_id}:router"}},
        )
        text = extract_result_text(result) if isinstance(result, dict) else str(result)
        json_block = _extract_json_block(text)
        if not json_block:
            return None
        payload = json.loads(json_block)
        if not isinstance(payload, dict):
            return None
        mode_value = payload.get("mode")
        reason_value = payload.get("reason")
        if not isinstance(mode_value, str) or not isinstance(reason_value, str):
            return None
        normalized_mode = _normalize_mode(mode_value)
        if not normalized_mode:
            return None
        return normalized_mode, f"LLM路由：{reason_value}"
    except Exception:
        return None


def _heuristic_route(prompt: str) -> tuple[str, str]:
    normalized = prompt.lower().strip()

    if _contains_any(
        normalized,
        ["思维导图", "脑图", "mind map", "mindmap"],
    ):
        return WORKFLOW_PLAN_ACT_REPLAN, "检测到结构化产出请求，启用带复核重规划流程。"

    if _contains_any(
        normalized,
        [
            "对比",
            "比较",
            "优缺点",
            "评估",
            "综述",
            "方案",
            "架构",
            "多步",
            "分析",
            "trade-off",
            "compare",
            "evaluate",
        ],
    ):
        return WORKFLOW_PLAN_ACT_REPLAN, "检测到复杂分析任务，启用 Plan-Act-RePlan。"

    if _contains_any(
        normalized,
        ["总结", "概述", "提纲", "summary", "overview"],
    ):
        return WORKFLOW_PLAN_ACT, "检测到总结任务，启用 Plan-Act。"

    if len(normalized) > 120:
        return WORKFLOW_PLAN_ACT, "输入较长，优先走计划执行流程。"

    return WORKFLOW_REACT, "默认快速问答，启用 ReAct。"


def auto_select_workflow_mode(
    prompt: str,
    coordinator: Any | None = None,
) -> tuple[str, str]:
    if coordinator is not None:
        routed = _route_with_llm(prompt, coordinator)
        if routed is not None:
            return routed
    return _heuristic_route(prompt)
