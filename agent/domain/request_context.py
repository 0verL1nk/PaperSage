"""请求级上下文，用于路由拦截器的结构化决策输入。

在请求真正发出前，由应用层调用方组装后传入
policy_engine.intercept()，拦截器以此做路由决策。
"""
from dataclasses import dataclass


@dataclass
class RequestContext:
    """请求前拦截器的结构化输入。

    Attributes:
        prompt: 用户当前输入。
        context_digest: 会话历史压缩摘要（来自 context_governance），
            供 LLM 在做路由判断时了解对话背景。
    """

    prompt: str
    context_digest: str = ""
