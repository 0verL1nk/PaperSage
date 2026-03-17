## Context

`agent/capabilities.py` 是一个1000+行的单体模块，包含工具定义、Web搜索提供者、工具构建逻辑等多个职责。这违反了单一职责原则，导致代码难以维护和测试。

当前结构：
- `agent/capabilities.py`: 所有工具定义和构建逻辑
- `agent/tools/`: 仅包含local_ops和基础设施（registry, types）

目标结构：
- `agent/tools/`: 所有工具模块化组织
- 删除 `agent/capabilities.py`

## Goals / Non-Goals

**Goals:**
- 将capabilities.py拆分为独立的工具模块
- 提高代码可测试性和可维护性
- 保持现有功能完全不变
- 更新所有导入路径

**Non-Goals:**
- 不改变工具的行为或接口
- 不重构工具的内部实现
- 不添加新功能
- 不移除渐进式工具加载（已废弃）

## Decisions

### 决策1: 模块拆分策略
按功能域拆分，每个模块负责一类工具：

```
agent/tools/
├── document.py          # search_document, read_document, list_document
├── web_search.py        # Web搜索提供者（Brave, SearXNG, Wikipedia, DDG）
├── paper_search.py      # search_papers
├── skill.py             # use_skill
├── builder.py           # build_agent_tools主函数
├── utils.py             # 共享工具函数（_sanitize_query, _is_dangerous_query等）
├── registry.py          # 工具注册表
└── types.py             # 类型定义
```

**理由**: 按功能域拆分比按技术层拆分更清晰，每个模块职责单一。

**替代方案**: 按工具类型拆分（core_tools, search_tools等），但这会导致模块边界模糊。

### 决策2: 共享代码提取
将工具间共享的函数提取到 `utils.py`：
- `_sanitize_query`
- `_is_dangerous_query`
- `_preview`
- `_env_flag`, `_env_value`, `_load_secret`
- `_format_web_results`

**理由**: 避免代码重复，便于统一维护安全策略。

### 决策3: 导入路径迁移
所有从 `agent.capabilities` 导入的代码需要更新为：
```python
# 旧
from agent.capabilities import build_agent_tools

# 新
from agent.tools.builder import build_agent_tools
```

**理由**: 直接导入，不提供兼容层，强制代码库更新到新结构。

**替代方案**: 在 `agent/__init__.py` 提供兼容导入，但这会延长技术债务。

### 决策4: Web搜索提供者设计
将所有Web搜索提供者封装在 `web_search.py` 中，提供统一的构建函数：
```python
def build_web_search_tool(enabled: bool) -> Tool | None
```

**理由**: Web搜索提供者逻辑紧密相关，放在一个模块中便于管理回退策略。

## Risks / Trade-offs

**[风险] 导入路径破坏性变更** → 通过全局搜索替换所有导入，确保一次性完成迁移

**[风险] 测试覆盖不足** → 在重构前确保现有测试通过，重构后再次验证

**[权衡] 模块粒度** → 选择中等粒度（5-8个模块），避免过度拆分导致导入复杂

**[权衡] 向后兼容** → 不提供兼容层，强制更新，避免长期维护两套导入路径
