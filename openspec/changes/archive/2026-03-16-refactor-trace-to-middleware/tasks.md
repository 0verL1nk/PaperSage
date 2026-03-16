## 1. 目录结构和基础设置

- [x] 1.1 创建 `agent/middlewares/` 目录
- [x] 1.2 创建 `agent/middlewares/__init__.py` 文件
- [x] 1.3 定义扩展的 AgentState schema（添加 trace_labels 和 trace_summary 字段）

## 2. TraceMiddleware 实现

- [x] 2.1 创建 `agent/middlewares/trace.py` 文件
- [x] 2.2 实现 TraceMiddleware 类，继承 AgentMiddleware
- [x] 2.3 实现 before_model hook，从 state["messages"] 提取 performative 并记录到 trace_labels
- [x] 2.4 实现 after_model hook，从新增的 AIMessage 提取 performative 并记录
- [x] 2.5 实现 after_agent hook，调用 phase_summary 生成摘要并存储到 trace_summary
- [ ] 2.6 编写 TraceMiddleware 的单元测试

## 3. ProgressiveToolDisclosureMiddleware 重构

- [x] 3.1 创建 `agent/middlewares/progressive_tool_disclosure.py` 文件
- [x] 3.2 实现 ProgressiveToolDisclosureMiddleware 类，继承 AgentMiddleware
- [x] 3.3 实现 wrap_model_call hook，使用 request.override(tools=filtered_tools) 过滤工具
- [x] 3.4 实现从消息历史提取激活工具名称的逻辑
- [x] 3.5 保持环境变量 AGENT_PROGRESSIVE_TOOL_DISCLOSURE 控制开关
- [ ] 3.6 编写 ProgressiveToolDisclosureMiddleware 的单元测试

## 4. 更新导入和注册

- [x] 4.1 在 `agent/middlewares/__init__.py` 中导出 TraceMiddleware 和 ProgressiveToolDisclosureMiddleware
- [x] 4.2 在 `agent/capabilities.py` 中添加 deprecated 导入别名，指向新位置
- [x] 4.3 更新 `agent/runtime_agent.py` 中的 middleware 导入路径
- [x] 4.4 更新 middleware 注册逻辑，使用新的实现
- [x] 4.5 更新其他使用方的导入路径（如有）

## 5. 测试和验证

- [x] 5.1 运行所有单元测试，确保通过
- [x] 5.2 运行集成测试，验证 trace 功能正常（跳过e2e测试）
- [x] 5.3 对比重构前后的 trace 输出，确保一致性
- [x] 5.4 验证 progressive tool disclosure 功能正常工作
- [x] 5.5 更新相关测试文件（tests/unit/test_agent_capabilities.py, tests/unit/test_domain_trace.py）

## 6. 清理和文档

- [x] 6.1 移除 `agent/capabilities.py` 中的旧实现代码（保留 deprecated 导入）
- [x] 6.2 更新代码注释和文档字符串
- [x] 6.3 验证所有测试通过后提交代码
