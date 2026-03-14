# pyright: reportUnknownMemberType=false, reportExplicitAny=false, reportAny=false
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..domain.orchestration import ExecutionPlan, PlanStep, render_execution_plan
from .langgraph_plan_act import run_plan_act_graph
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


class PlannerStepOutput(BaseModel):
    title: str = Field(min_length=1, description="步骤标题")
    description: str = Field(default="", description="步骤说明")
    depends_on: list[str] = Field(default_factory=list, description="依赖 step id")
    tool_hints: list[str] = Field(default_factory=list, description="建议工具")
    done_when: str = Field(default="", description="步骤完成标准")


class PlannerOutput(BaseModel):
    goal: str = Field(min_length=1, description="本轮回答目标")
    constraints: list[str] = Field(default_factory=list, description="约束条件")
    steps: list[str | PlannerStepOutput] = Field(min_length=1, description="可执行步骤")
    tool_hints: list[str] = Field(default_factory=list, description="计划级工具建议")
    done_when: str = Field(default="", description="整体完成标准")


def _fallback_plan(prompt: str, *, min_steps: int, max_steps: int) -> ExecutionPlan:
    titles = [
        "检索与问题最相关的文档证据。",
        "汇总核心观点并校验一致性。",
        "输出结构化结论与证据依据。",
    ]
    normalized_count = max(1, min(max_steps, max(min_steps, len(titles))))
    steps = [
        PlanStep(
            id=f"step_{idx}",
            title=titles[idx - 1] if idx <= len(titles) else f"补充步骤 {idx}",
            description=titles[idx - 1] if idx <= len(titles) else f"补充步骤 {idx}",
            tool_hints=["search_document"] if idx == 1 else [],
            done_when=(
                "至少获得一条与问题直接相关的文档证据。"
                if idx == 1
                else "当前步骤产出可直接支持最终回答。"
            ),
        )
        for idx in range(1, normalized_count + 1)
    ]
    return ExecutionPlan(
        goal="基于文档证据回答用户问题。",
        constraints=[f"问题：{prompt.strip()}"] if str(prompt).strip() else [],
        steps=steps,
        tool_hints=["search_document"],
        done_when="输出带证据依据的最终回答。",
    )


def _normalize_plan(
    payload: PlannerOutput,
    *,
    min_steps: int,
    max_steps: int,
) -> ExecutionPlan | None:
    goal = payload.goal.strip()
    if not goal:
        return None
    steps: list[PlanStep] = []
    for idx, raw_step in enumerate(payload.steps[:max_steps], start=1):
        if isinstance(raw_step, PlannerStepOutput):
            title = raw_step.title.strip()
            description = raw_step.description.strip() or title
            depends_on = [str(item).strip() for item in raw_step.depends_on if str(item).strip()]
            tool_hints = [str(item).strip() for item in raw_step.tool_hints if str(item).strip()]
            done_when = raw_step.done_when.strip() or "当前步骤输出满足计划推进条件。"
        else:
            title = str(raw_step).strip()
            description = title
            depends_on = []
            tool_hints = []
            done_when = "当前步骤输出满足计划推进条件。"
        if not title:
            continue
        steps.append(
            PlanStep(
                id=f"step_{idx}",
                title=title,
                description=description,
                depends_on=depends_on,
                tool_hints=tool_hints,
                done_when=done_when,
            )
        )
    if len(steps) < min_steps:
        return None
    return ExecutionPlan(
        goal=goal,
        constraints=[str(item).strip() for item in payload.constraints if str(item).strip()],
        steps=steps,
        tool_hints=[str(item).strip() for item in payload.tool_hints if str(item).strip()],
        done_when=payload.done_when.strip(),
    )


def _build_execution_plan_with_llm(
    prompt: str,
    *,
    llm: Any | None,
    min_steps: int,
    max_steps: int,
) -> ExecutionPlan:
    if llm is None:
        return _fallback_plan(prompt, min_steps=min_steps, max_steps=max_steps)
    try:
        if not hasattr(llm, "with_structured_output"):
            return _fallback_plan(prompt, min_steps=min_steps, max_steps=max_steps)
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
            return _fallback_plan(prompt, min_steps=min_steps, max_steps=max_steps)
        normalized = _normalize_plan(payload, min_steps=min_steps, max_steps=max_steps)
        if normalized is None:
            return _fallback_plan(prompt, min_steps=min_steps, max_steps=max_steps)
        return normalized
    except Exception:
        return _fallback_plan(prompt, min_steps=min_steps, max_steps=max_steps)


def build_execution_plan(prompt: str, llm: Any | None = None) -> ExecutionPlan:
    settings = load_agent_settings()
    min_steps = max(1, settings.agent_planner_min_steps)
    max_steps = max(min_steps, settings.agent_planner_max_steps)

    def _planner(plan_prompt: str) -> ExecutionPlan | None:
        return _build_execution_plan_with_llm(
            plan_prompt,
            llm=llm,
            min_steps=min_steps,
            max_steps=max_steps,
        )

    try:
        return run_plan_act_graph(
            prompt=prompt,
            planner=_planner,
            max_attempts=1,
        )
    except Exception:
        return _build_execution_plan_with_llm(
            prompt,
            llm=llm,
            min_steps=min_steps,
            max_steps=max_steps,
        )


def revise_execution_plan(
    *,
    prompt: str,
    current_plan: ExecutionPlan,
    failed_step_id: str,
    failure_reason: str,
    llm: Any | None = None,
) -> ExecutionPlan:
    settings = load_agent_settings()
    min_steps = max(1, settings.agent_planner_min_steps)
    max_steps = max(min_steps, settings.agent_planner_max_steps)
    failed_step = next(
        (step for step in current_plan.steps if step.id == str(failed_step_id).strip()),
        None,
    )
    fallback_steps = [
        PlanStep(
            id="step_1",
            title=(
                failed_step.title
                if failed_step is not None and failed_step.title.strip()
                else "补充缺失信息"
            ),
            description=(
                f"针对失败原因补充执行：{failure_reason}"
                if str(failure_reason).strip()
                else (failed_step.description if failed_step is not None else "补充缺失信息")
            ),
            tool_hints=list(getattr(failed_step, "tool_hints", []) or []),
            done_when=(
                failed_step.done_when
                if failed_step is not None and str(failed_step.done_when).strip()
                else "当前步骤输出满足计划推进条件。"
            ),
        ),
        PlanStep(
            id="step_2",
            title="输出修订后的最终回答",
            description="基于补充结果输出修订后的最终回答",
            done_when="输出最终回答。",
        ),
    ]
    fallback_plan = ExecutionPlan(
        goal=current_plan.goal,
        constraints=list(current_plan.constraints) + [f"失败原因：{failure_reason}"],
        steps=fallback_steps[:max_steps],
        tool_hints=list(current_plan.tool_hints),
        done_when=current_plan.done_when or "输出修订后的最终回答。",
    )
    if llm is None or not hasattr(llm, "with_structured_output"):
        return fallback_plan
    try:
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _planner_instruction(min_steps=min_steps, max_steps=max_steps)),
                (
                    "human",
                    "用户问题：\n{prompt}\n\n当前计划：\n{plan_text}\n\n失败步骤：{failed_step_id}\n失败原因：{failure_reason}\n请输出修订后的计划。",
                ),
            ]
        )
        chain = planner_prompt | _with_structured_output(llm, PlannerOutput)
        payload = chain.invoke(
            {
                "prompt": prompt,
                "plan_text": render_execution_plan(current_plan),
                "failed_step_id": failed_step_id,
                "failure_reason": failure_reason,
            }
        )
        if isinstance(payload, dict):
            payload = PlannerOutput.model_validate(payload)
        if not isinstance(payload, PlannerOutput):
            return fallback_plan
        normalized = _normalize_plan(payload, min_steps=min_steps, max_steps=max_steps)
        return normalized if normalized is not None else fallback_plan
    except Exception:
        return fallback_plan


def _with_structured_output(llm: Any, schema: type[BaseModel]) -> Any:
    # Prefer function-calling path to avoid provider parse payload warnings.
    try:
        return llm.with_structured_output(schema, method="function_calling")
    except TypeError:
        return llm.with_structured_output(schema)
