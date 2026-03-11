PHASE_BY_PERFORMATIVE = {
    "request": "接收请求",
    "dispatch": "调度中",
    "policy": "策略判定",
    "policy_switch": "策略切换",
    "plan": "规划",
    "step_dispatch": "步骤执行",
    "step_result": "步骤结果",
    "step_verify": "步骤校验",
    "step_complete": "步骤完成",
    "step_retry": "步骤重试",
    "plan_todo": "任务规划",
    "plan_todo_reject": "规划修正",
    "tool_load": "工具加载",
    "tool_call": "调用工具",
    "tool_result": "工具返回",
    "tool_activate": "激活工具",
    "skill_activate": "激活技能",
    "draft": "生成草稿",
    "review": "复核",
    "replan": "重规划",
    "fallback": "回退",
    "final": "输出最终答案",
}


def phase_label_from_performative(performative: str) -> str:
    return PHASE_BY_PERFORMATIVE.get(performative, "处理中")


def phase_summary(labels: list[str]) -> str:
    if not labels:
        return "无"
    unique: list[str] = []
    for label in labels:
        if not unique or unique[-1] != label:
            unique.append(label)
    return " -> ".join(unique)
