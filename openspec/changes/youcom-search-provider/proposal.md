## Why

search_web 的提供者链目前依赖 Brave（需 key）、SearXNG 公共池、Firecrawl 免费档与
Wikipedia。You.com 的 MCP 免费档提供 keyless 网页搜索，与 Firecrawl 免费档同类：
不引入任何新依赖（复用核心 httpx）、不需要 API key，可以在不改变默认行为的前提
下为链路增加一个可选的托管搜索提供者。此前被关闭的外部搜索提供者提案（#29/#36）
针对的是已退役的 requirements.txt 工作流，维护者明确表示后续提案应以「新鲜的、
范围收敛的变更」形式针对当前 provider/configuration 边界重新提出——本变更遵循
该边界：默认关闭、零新依赖、按既有提供者注册模式接线。

## What Changes

- 新增可选 You.com 提供者（`AGENT_WEB_YOUCOM_ENABLED=1` 开启，默认关）：
  - keyless 默认走免费档端点 `https://api.you.com/mcp?profile=free`；
    设置 `YOUCOM_API_KEY` 后切换到认证端点 `https://api.you.com/mcp`（Bearer）。
    `AGENT_YOUCOM_MCP_URL` 可覆盖端点。
  - 通过核心依赖 httpx 以 MCP streamable-HTTP JSON-RPC 单次 stateless
    `tools/call you-search` 完成搜索（免 profile 免会话握手，已实测）；
    响应兼容 SSE 与纯 JSON 两种形态，跳过 notification 帧，LF/CRLF 帧切分。
  - 结果映射复用 `_format_web_results`（title/url/description，缺
    description 时回退 contents.highlights 首条）。
- 注册位置：`firecrawl_search` 之后、`wikipedia_api` 之前；失败/无结果按既有
  链路规则落到下一个提供者（含 circuit breaker 与既有回退语义）。
- 实现拆分到独立模块 `agent/tools/web_search_youcom.py`，保持 `web_search.py`
  低于 500 行红线（UA 常量迁至 `agent/tools/utils.py` 单一出处）。

## Capabilities

### New Capabilities

无（仅扩展既有 web-search-providers 能力的提供者清单）。

### Modified Capabilities

- `web-search-providers`：新增 You.com 可选提供者及其 opt-in 开关、keyless/认证
  双模式与回退语义。

## Impact

- 代码：`agent/tools/web_search_youcom.py`（新）、`agent/tools/web_search.py`
  （注册 + UA 常量迁移）、`agent/tools/utils.py`（UA 常量单一出处）。
- 配置：`.env.example` 新增 `AGENT_WEB_YOUCOM_ENABLED` / `YOUCOM_API_KEY`。
- 测试：`tests/unit/test_web_search_youcom.py`（网络无关单测：开关/URL 矩阵、
  请求形状、SSE/JSON 解析、highlight 回退、错误路径、链路注册顺序与默认排除）。
- 风险：外部端点可用性/限额由链路回退与 circuit breaker 兜底；默认关闭，现有
  用户行为不变。