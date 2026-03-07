from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..settings import load_agent_settings


def _planner_instruction(min_steps: int, max_steps: int) -> str:
    return f"""
你是执行计划器。请把用户问题拆为最小可执行步骤，服务于“基于证据的最终回答”。

要求：
1) 仅输出 {min_steps}-{max_steps} 步，步骤要可执行、可验证、无废话
2) 每一步都应对最终答案有直接贡献
3) 允许引用“检索/校验/综合/输出”等执行动作，但不得写空泛术语
4) 避免重复步骤，禁止输出推理过程和解释性段落
"""


class PlannerOutput(BaseModel):
    goal: str = Field(min_length=1, description="本轮回答目标")
    steps: list[str] = Field(min_length=1, description="可执行步骤")


def _fallback_plan(prompt: str) -> str:
    return (
        "目标：基于文档证据回答用户问题。\n"
        "1. 检索与问题最相关的文档证据。\n"
        "2. 汇总核心观点并校验一致性。\n"
        "3. 输出结构化结论与证据依据。\n"
        f"问题：{prompt}"
    )


def build_execution_plan(prompt: str, llm: Any | None = None) -> str:
    settings = load_agent_settings()
    min_steps = max(1, settings.agent_planner_min_steps)
    max_steps = max(min_steps, settings.agent_planner_max_steps)
    if llm is None:
        return _fallback_plan(prompt)
    try:
        if not hasattr(llm, "with_structured_output"):
            return _fallback_plan(prompt)
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _planner_instruction(min_steps=min_steps, max_steps=max_steps)),
                ("human", "用户问题：\n{prompt}"),
            ]
        )
        chain = planner_prompt | _with_structured_output(llm, PlannerOutput)
        payload = chain.invoke({"prompt": prompt})
        if isinstance(payload, dict):
            payload = PlannerOutput.model_validate(payload)
        if not isinstance(payload, PlannerOutput):
            return _fallback_plan(prompt)
        goal = payload.goal.strip()
        steps = [str(item).strip() for item in payload.steps if str(item).strip()]
        if len(steps) < min_steps:
            return _fallback_plan(prompt)
        lines = [f"目标：{goal or '完成证据充分的回答'}"]
        lines.extend(f"{idx}. {step}" for idx, step in enumerate(steps[:max_steps], start=1))
        return "\n".join(lines)
    except Exception:
        return _fallback_plan(prompt)


def _with_structured_output(llm: Any, schema: type[BaseModel]) -> Any:
    # Prefer function-calling path to avoid provider parse payload warnings.
    try:
        return llm.with_structured_output(schema, method="function_calling")
    except TypeError:
        return llm.with_structured_output(schema)
