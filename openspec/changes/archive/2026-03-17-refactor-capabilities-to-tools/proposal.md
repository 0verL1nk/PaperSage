## Why

`agent/capabilities.py` 是一个超过1000行的单体模块，混合了工具定义、Web搜索提供者、工具构建逻辑和安全机制。这导致代码难以维护、测试和扩展。将其拆分到 `agent/tools` 目录可以提高模块化程度，使每个工具独立可测试。

## What Changes

- 将所有工具定义从 `capabilities.py` 移到 `agent/tools/` 独立模块
- 将Web搜索提供者抽取为独立模块 `agent/tools/web_search.py`
- 将工具构建器移到 `agent/tools/builder.py`
- 保留 `agent/tools/registry.py` 用于工具注册和搜索
- 删除 `agent/capabilities.py` 文件
- 更新所有导入路径

## Capabilities

### New Capabilities
- `document-tools`: 文档相关工具（search_document, read_document, list_document）
- `web-search-providers`: Web搜索提供者抽象层（Brave, SearXNG, Wikipedia, DuckDuckGo）
- `paper-search-tool`: 学术论文搜索工具
- `skill-tool`: 技能模板应用工具
- `tool-builder`: 工具构建和组装逻辑

### Modified Capabilities
<!-- 无需修改现有capabilities -->

## Impact

- **代码结构**: `agent/capabilities.py` 将被删除，功能分散到 `agent/tools/` 多个模块
- **导入路径**: 所有从 `agent.capabilities` 导入的代码需要更新
- **测试**: 需要更新单元测试以匹配新的模块结构
- **向后兼容**: 可以在 `agent/__init__.py` 中提供兼容性导入（可选）
