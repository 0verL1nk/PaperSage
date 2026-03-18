"""重构后的多 Agent 系统集成测试"""

from agent.domain.todo_graph import TodoGraph
from agent.subagent.loader import load_subagent_configs
from agent.team.runtime import TeamRuntime


def test_subagent_loader_loads_configs():
    """测试 SubAgent 配置加载"""
    configs = load_subagent_configs()

    assert len(configs) >= 3
    assert any(c["name"] == "researcher" for c in configs)
    assert any(c["name"] == "reviewer" for c in configs)
    assert any(c["name"] == "writer" for c in configs)

    for config in configs:
        assert "name" in config
        assert "description" in config
        assert "system_prompt" in config


def test_team_runtime_lifecycle():
    """测试 Team Runtime 基本结构"""
    runtime = TeamRuntime("test-team")

    # 验证初始化
    assert runtime.team_id == "test-team"
    assert len(runtime.agents) == 0
    assert runtime.result_dir.exists()

    # 清理
    runtime.cleanup()


def test_todo_graph_dependency_management():
    """测试 Todo 依赖管理"""
    todos = [
        {"id": "1", "content": "Task 1", "status": "pending", "depends_on": []},
        {"id": "2", "content": "Task 2", "status": "pending", "depends_on": ["1"]},
        {"id": "3", "content": "Task 3", "status": "pending", "depends_on": ["1", "2"]},
    ]

    graph = TodoGraph(todos)

    # 检测无环
    assert not graph.has_cycle()

    # 拓扑排序
    sorted_ids = graph.topological_sort()
    assert sorted_ids == ["1", "2", "3"]

    # 获取可执行 todos
    executable = graph.get_executable_todos()
    assert len(executable) == 1
    assert executable[0]["id"] == "1"

    # 完成 Task 1
    todos[0]["status"] = "completed"
    graph = TodoGraph(todos)
    executable = graph.get_executable_todos()
    assert len(executable) == 1
    assert executable[0]["id"] == "2"


def test_todo_graph_cycle_detection():
    """测试环检测"""
    todos = [
        {"id": "1", "content": "Task 1", "status": "pending", "depends_on": ["2"]},
        {"id": "2", "content": "Task 2", "status": "pending", "depends_on": ["1"]},
    ]

    graph = TodoGraph(todos)
    assert graph.has_cycle()
