from agent.multi_agent_a2a import WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN, WORKFLOW_REACT
from agent.workflow_router import auto_select_workflow_mode


def test_router_uses_replan_for_mindmap_prompt():
    mode, _reason = auto_select_workflow_mode("请根据我们对话生成思维导图")
    assert mode == WORKFLOW_PLAN_ACT_REPLAN


def test_router_uses_replan_for_complex_compare_prompt():
    mode, _reason = auto_select_workflow_mode("请对比这两种方法的优缺点并给出评估")
    assert mode == WORKFLOW_PLAN_ACT_REPLAN


def test_router_uses_plan_act_for_summary_prompt():
    mode, _reason = auto_select_workflow_mode("请总结这篇论文")
    assert mode == WORKFLOW_PLAN_ACT


def test_router_uses_react_for_short_fact_prompt():
    mode, _reason = auto_select_workflow_mode("这篇论文的核心结论是什么")
    assert mode == WORKFLOW_REACT


class _FakePlanner:
    def __init__(self, result):
        self._result = result

    def invoke(self, *_args, **_kwargs):
        return self._result


class _FakeCoordinator:
    def __init__(self, result):
        self.session_id = "test-session"
        self.planner_agent = _FakePlanner(result)


def test_router_prefers_llm_decision_even_without_keywords():
    result = {
        "messages": [
            type("Msg", (), {"type": "ai", "content": '{"mode":"plan_act","reason":"需要先规划","confidence":0.88}'})()
        ]
    }
    coordinator = _FakeCoordinator(result)
    mode, reason = auto_select_workflow_mode("请帮我处理这个问题", coordinator=coordinator)
    assert mode == WORKFLOW_PLAN_ACT
    assert "LLM路由" in reason


def test_router_fallbacks_when_llm_output_is_invalid():
    result = {
        "messages": [
            type("Msg", (), {"type": "ai", "content": "invalid output"})()
        ]
    }
    coordinator = _FakeCoordinator(result)
    mode, _reason = auto_select_workflow_mode("请总结本文", coordinator=coordinator)
    assert mode == WORKFLOW_PLAN_ACT
