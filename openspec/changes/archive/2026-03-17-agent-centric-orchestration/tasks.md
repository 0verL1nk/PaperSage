## 1. 创建OrchestrationMiddleware

- [x] 1.1 创建agent/middlewares/orchestration.py文件
- [x] 1.2 实现OrchestrationMiddleware类，继承AgentMiddleware
- [x] 1.3 实现_is_complex_task方法，使用启发式规则判断任务复杂度
- [x] 1.4 实现_inject_guidance方法，插入引导提示到messages
- [x] 1.5 实现wrap_model_call方法，集成复杂度分析和引导注入
- [x] 1.6 在agent/middlewares/__init__.py中导出OrchestrationMiddleware

## 2. 创建Plan管理工具

- [x] 2.1 创建agent/tools/plan_tools.py文件
- [x] 2.2 实现create_plan工具（goal, description参数）
  - [x] 2.2.1 工具只存储Leader提供的内容，不生成内容
  - [x] 2.2.2 不调用planning_service.build_execution_plan
  - [x] 2.2.3 不使用LLM生成plan内容
- [x] 2.3 实现read_plan工具（返回Leader创建的内容）
- [x] 2.4 实现update_plan工具（description参数，存储Leader的更新）
- [x] 2.5 实现delete_plan工具
- [x] 2.6 为所有工具添加_progressive_tool_visibility="lazy"属性
- [x] 2.7 添加工具的详细docstring，说明：
  - [x] 2.7.1 Plan内容由Leader自己撰写
  - [x] 2.7.2 工具只负责存储和检索
  - [x] 2.7.3 建议Leader根据上下文规划策略
- [x] 2.8 Plan数据存储在agent state中（临时，会话级别）

## 3. 创建Team激活工具

- [x] 3.1 创建agent/tools/team_tools.py文件
- [x] 3.2 实现activate_team_mode工具函数
- [x] 3.3 为工具添加_progressive_tool_visibility="lazy"属性
- [x] 3.4 添加工具的详细docstring，说明使用场景

## 4. 集成LangChain TodoListMiddleware

- [x] 4.1 安装或确认langchain包依赖
- [x] 4.2 创建agent/middlewares/todolist.py文件
- [x] 4.3 从langchain.agents.middleware导入TodoListMiddleware
- [x] 4.4 配置TodoListMiddleware并集成到agent middleware链
- [x] 4.5 在agent/middlewares/__init__.py中导出TodoListMiddleware配置

## 5. 移除现有Todo工具

- [x] 5.1 从agent/tools/local_ops.py中移除WriteTodoInput类定义
- [x] 5.2 从agent/tools/local_ops.py中移除EditTodoInput类定义
- [x] 5.3 从agent/tools/local_ops.py中移除write_todo工具函数
- [x] 5.4 从agent/tools/local_ops.py中移除edit_todo工具函数
- [x] 5.5 从agent/tools/local_ops.py中移除todo相关helper函数
- [x] 5.6 从LOCAL_OPS_TOOL_METADATA中移除write_todo和edit_todo条目
- [x] 5.7 更新agent工具注册，移除todo工具引用

## 6. 修改Orchestrator

- [x] 6.1 在orchestrator.py中注释掉intercept_policy调用
- [x] 6.2 移除policy_decision的前置初始化
- [x] 6.3 移除mode检测的while循环（1031-1168行）
- [x] 6.4 移除build_execution_plan调用和plan生成逻辑
- [x] 6.5 移除_run_plan_runtime调用和plan执行逻辑
- [x] 6.6 简化为单次leader_agent.invoke调用
- [x] 6.7 更新policy_decision构建逻辑，基于tool调用结果
- [x] 6.8 确保on_event callback正确传递到runtime_config

## 7. 集成工具和Middleware到Agent

- [x] 7.1 在agent配置中注册OrchestrationMiddleware
- [x] 7.2 在agent配置中注册TodoListMiddleware
- [x] 7.3 在agent工具注册处添加plan_tools
- [x] 7.4 在agent工具注册处添加team_tools
- [x] 7.5 确保plan和team工具被标记为lazy
- [x] 7.6 验证progressive_tool_disclosure正确处理这些工具

## 8. 添加单元测试

- [ ] 8.1 为OrchestrationMiddleware添加测试
- [ ] 8.2 为create_plan工具添加测试
- [ ] 8.3 为read_plan工具添加测试
- [ ] 8.4 为update_plan工具添加测试
- [ ] 8.5 为delete_plan工具添加测试
- [ ] 8.6 为activate_team_mode工具添加测试
- [ ] 8.7 测试简化后的orchestrator流程
- [ ] 8.8 测试TodoListMiddleware集成

## 9. 集成测试

- [ ] 9.1 测试简单任务场景（不触发plan/team tools）
- [ ] 9.2 测试复杂任务场景（agent主动使用plan工具）
- [ ] 9.3 测试团队协作场景（agent主动使用team工具）
- [ ] 9.4 测试todolist功能（agent使用LangChain提供的todo工具）
- [ ] 9.5 验证OrchestrationMiddleware引导效果
- [ ] 9.6 验证agent可以忽略引导建议

## 10. 清理和文档

- [x] 10.1 删除agent/orchestration/policy_engine.py
- [x] 10.2 移除policy_engine的所有import语句
- [x] 10.3 评估是否需要保留planning_service.py
  - [x] 10.3.1 如果planning_service只用于旧的plan模式，考虑删除
  - [x] 10.3.2 如果有其他地方使用，保留但标记为deprecated
- [x] 10.4 清理.agent/todo.json文件（如果存在）
- [x] 10.5 更新相关文档说明新的架构
- [x] 10.6 添加LangChain依赖说明
- [x] 10.7 添加环境变量配置说明（如果需要）
- [x] 10.8 更新README说明Plan由Leader自主生成
