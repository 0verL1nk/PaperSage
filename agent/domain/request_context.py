"""请求级上下文，用于策略分析的结构化输入。

当前 Agent Center 主链路由 middleware 负责复杂度分析与模式提示。
该结构保留为策略分析输入的共享载体，供运行时或兼容场景传递请求摘要。
"""
from dataclasses import dataclass


@dataclass
class RequestContext:
    """策略分析输入。

    Attributes:
        prompt: 用户当前输入。
        context_digest: 会话历史压缩摘要（来自 context_governance），
            供策略分析组件了解对话背景。
    """

    prompt: str
    context_digest: str = ""
