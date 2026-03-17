## 1. 准备工作

- [x] 1.1 创建 agent/tools/utils.py 文件
- [x] 1.2 创建 agent/tools/document.py 文件
- [x] 1.3 创建 agent/tools/web_search.py 文件
- [x] 1.4 创建 agent/tools/paper_search.py 文件
- [x] 1.5 创建 agent/tools/skill.py 文件
- [x] 1.6 创建 agent/tools/builder.py 文件

## 2. 提取共享工具函数

- [x] 2.1 将 _sanitize_query 移到 utils.py
- [x] 2.2 将 _is_dangerous_query 和 _DANGEROUS_QUERY_PATTERNS 移到 utils.py
- [x] 2.3 将 _preview 移到 utils.py
- [x] 2.4 将 _env_flag, _env_value, _load_secret, _read_from_dotenv 移到 utils.py
- [x] 2.5 将 _format_web_results 移到 utils.py
- [x] 2.6 将常量（DEFAULT_MAX_QUERY_CHARS等）移到 utils.py

## 3. 创建文档工具模块

- [x] 3.1 将 SearchDocumentInput, ReadDocumentInput, ListDocumentInput 移到 document.py
- [x] 3.2 实现 build_search_document_tool 函数
- [x] 3.3 实现 build_read_document_tool 函数
- [x] 3.4 实现 build_list_document_tool 函数

## 4. 创建Web搜索模块

- [x] 4.1 将 SearchWebInput 移到 web_search.py
- [x] 4.2 将 _build_brave_web_search_client 移到 web_search.py
- [x] 4.3 将 _build_searxng_web_search_client 和 _parse_searxng_instances 移到 web_search.py
- [x] 4.4 将 _build_wikipedia_web_search_client 移到 web_search.py
- [x] 4.5 将 _build_native_web_search_client 移到 web_search.py
- [x] 4.6 实现 build_web_search_tool 函数，整合所有提供者

## 5. 创建论文搜索模块

- [x] 5.1 将 SearchPapersInput 移到 paper_search.py
- [x] 5.2 实现 build_paper_search_tool 函数

## 6. 创建技能工具模块

- [x] 6.1 将 SkillInput 移到 skill.py
- [x] 6.2 将 _get_skill_options 移到 skill.py
- [x] 6.3 实现 build_skill_tool 函数

## 7. 创建工具构建器

- [x] 7.1 将 _tool_enabled 移到 builder.py
- [x] 7.2 将 discover_available_tools 移到 builder.py
- [x] 7.3 实现 build_agent_tools 主函数，调用各个工具构建函数
- [x] 7.4 移除 start_plan 和 start_team 工具的构建逻辑
- [x] 7.5 移除渐进式工具加载相关代码（search_tools, _extract_tool_names_from_search_result, _extract_activated_tool_names, _resolve_fixed_tool_names, _parse_tool_name_set, _schema_manifest_for_tool）

## 8. 更新导入路径

- [ ] 8.1 搜索所有从 agent.capabilities 导入的代码
- [ ] 8.2 更新为从 agent.tools.builder 导入 build_agent_tools
- [ ] 8.3 更新为从 agent.tools 导入 discover_available_tools（如果需要）
- [ ] 8.4 更新 agent/tools/__init__.py 导出新模块

## 9. 测试验证

- [ ] 9.1 运行现有单元测试，确保通过
- [ ] 9.2 验证工具构建功能正常
- [ ] 9.3 验证所有工具可以正常调用
- [ ] 9.4 检查导入路径是否全部更新

## 10. 清理废弃代码

- [ ] 10.1 删除 agent/runtime_agent.py 中的 SPAWN_TOOL_NAMES 定义
- [ ] 10.2 更新 agent/orchestration/team_runtime.py 移除 SPAWN_TOOL_NAMES 引用
- [ ] 10.3 更新 agent/stream.py 移除 start_plan/start_team 检查
- [ ] 10.4 更新 tests/unit/test_agent_stream.py 移除相关测试
- [ ] 10.5 删除 agent/capabilities.py 文件
- [ ] 10.6 删除 agent/tools/local_ops.py 文件
- [ ] 10.7 更新 agent/tools/__init__.py 移除 local_ops 导出
- [ ] 10.8 确认没有遗留的导入引用
