from agent.domain.trace import phase_label_from_performative, phase_summary


def test_phase_label_from_performative_known_and_unknown():
    assert phase_label_from_performative("plan") == "规划"
    assert phase_label_from_performative("policy_switch") == "策略切换"
    assert phase_label_from_performative("step_dispatch") == "步骤执行"
    assert phase_label_from_performative("unknown") == "处理中"


def test_phase_summary_deduplicates_adjacent_labels():
    assert phase_summary([]) == "无"
    assert phase_summary(["接收请求", "接收请求", "规划", "规划", "输出最终答案"]) == (
        "接收请求 -> 规划 -> 输出最终答案"
    )
