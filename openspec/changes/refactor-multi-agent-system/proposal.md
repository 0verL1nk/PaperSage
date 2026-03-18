# 多 Agent 系统重构

## 概述

重构当前的多 Agent 系统，删除旧的 A2A 协调器和 orchestration 模块，实现基于中间件的新架构。

## 问题

当前系统存在以下问题：

1. **架构混乱**：存在两套不兼容的 team 系统（A2A 协调器和 Team Runtime）
2. **硬编码角色**：A2A 使用固定的 4 个角色，无法动态扩展
3. **编排逻辑分散**：orchestrator.py 包含大量编排逻辑，应该在中间件层实现
4. **缺少统一抽象**：没有类似 LangChain SubagentMiddleware 的统一接口

## 目标

1. **删除旧系统**：移除 `agent/a2a/` 和 `agent/orchestration/` 目录
2. **实现 SubAgent Middleware**：常驻中间件，动态加载预定义的专业 agent
3. **实现 Team 模式**：按需激活的团队协作模式，支持动态创建和管理 agent
4. **增强 TodoList**：支持依赖关系和拓扑排序

## 架构变更

### 删除
- `agent/a2a/` - A2A 协调器
- `agent/orchestration/` - 编排模块

### 新增
- `agent/subagent/` - SubAgent 配置目录
- `agent/team/` - Team 模式实现
- `agent/middlewares/subagent.py` - SubAgent 中间件
- `agent/middlewares/team.py` - Team 中间件
- `agent/middlewares/enhanced_todolist.py` - 增强版 TodoList
- `agent/domain/todo_graph.py` - Todo 依赖图

## 影响范围

- 删除约 20 个文件
- 新增约 15 个文件
- 修改约 5 个核心文件
