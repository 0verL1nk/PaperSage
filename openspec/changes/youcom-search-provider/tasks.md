## 1. 实现

- [x] 1.1 提供者模块 `agent/tools/web_search_youcom.py`：stateless JSON-RPC
  `tools/call you-search`，SSE/JSON 双形态解析，notification 帧跳过，
  LF/CRLF 帧切分
- [x] 1.2 链路注册：`firecrawl_search` 之后、`wikipedia_api` 之前，默认关闭
- [x] 1.3 配置：`.env.example` 新增 `AGENT_WEB_YOUCOM_ENABLED` / `YOUCOM_API_KEY`；
  `AGENT_YOUCOM_MCP_URL` 端点覆盖
- [x] 1.4 UA 常量迁移 `agent/tools/utils.py` 单一出处

## 2. 测试与门禁

- [x] 2.1 网络无关单测：开关/URL 矩阵（默认关、keyless、认证）、请求形状
  （method/name/arguments、无 Authorization）、SSE+notification 解析、
  纯 JSON 解析、highlight 回退、错误路径（状态码 / isError / 无效载荷 /
  results.web 缺失）
- [x] 2.2 链路注册顺序（firecrawl < youcom < wikipedia）与默认排除
- [x] 2.3 `quality_gate.sh core`（repository_guard / ruff / ty）与
  `tests/unit` 通过