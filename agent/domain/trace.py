PHASE_BY_PERFORMATIVE = {
    "request": "接收请求",
    "dispatch": "调度中",
    "policy": "策略判定",
    "plan": "规划",
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
