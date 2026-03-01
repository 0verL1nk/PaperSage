# 文献阅读助手
[English](README_EN.md) | 简体中文

一个基于 Streamlit + LangChain 的 AI 文献阅读平台，面向科研阅读与写作场景，支持文献提取、总结、问答、改写与思维导图。

## 功能总览

- 用户与文件管理：本地用户、文档上传、文档中心
- 文献分析：
  - 原文提取与关键信息分类
  - 论文总结
  - 文段优化/降重/翻译
  - 论文问答（支持多种 Agent 工作流）
- 可视化：思维导图（pyecharts Tree）

## Agent 工作流（论文问答页）

主入口 `main.py`（Agent 中心）提供三种模式：

1. `ReAct（Tool+Memory）`
2. `Plan-Act（A2A协调）`
3. `Plan-Act-RePlan（A2A协调）`

说明：
- `A2A` 协调包含 Planner / Researcher / Reviewer 角色。
- 代码中保留了 ACP 兼容命名，但主流程按 A2A 思路实现。

## 项目结构

```text
.
├── main.py
├── pages/
│   ├── 1_📁_文件中心.py
│   └── 2_⚙️_设置中心.py
├── agent/
│   ├── paper_agent.py          # ReAct 单 Agent 会话工厂
│   ├── multi_agent_a2a.py      # A2A/ACP 多 Agent 协调
│   ├── capabilities.py         # 工具定义（search_document/search_web/use_skill）
│   ├── local_rag.py            # 本地向量检索
│   └── ...
├── utils/
│   └── ...                     # 通用数据库/任务/页面工具
├── tests/
│   ├── unit/
│   └── integration/
├── pyproject.toml
└── docker-compose.yml
```

## 环境要求

- Python `>=3.10`
- [uv](https://github.com/astral-sh/uv)（推荐）

## 快速开始

1. 克隆仓库

```bash
git clone <仓库地址>
cd <项目目录>
```

2. 安装依赖

```bash
uv sync --no-install-project
```

3. 启动应用

```bash
streamlit run main.py
```

4. 打开浏览器

访问 `http://localhost:8501`

## 配置说明

在侧边栏 `设置` 中配置：

- `API Key`
- `模型名称`
- `OpenAI Compatible Base URL`（可选，不填则使用默认）

## 测试与质量

安装测试依赖：

```bash
uv sync --extra dev --no-install-project
```

运行单元测试：

```bash
uv run --extra dev python -m pytest tests/unit -q
```

运行集成 / E2E 测试：

```bash
uv run --extra dev python -m pytest tests/integration -q
```

运行真实 API E2E（需要 `.env`）：

`.env` 仅支持以下键：

```bash
RUN_LIVE_E2E=1
OPENAI_BASE_URL=...
OPENAI_MODEL_NAME=...
OPENAI_API_KEY=...
AGENT_ENABLE_THINKING=0
AGENT_REASONING_EFFORT=
```

执行命令：

```bash
uv run --extra dev python -m pytest tests/integration/test_live_api_e2e.py -q
```

运行覆盖率统计（建议作为质量门禁）：

```bash
uv run --extra dev python -m pytest \
  --cov=utils \
  --cov=agent \
  --cov=pages \
  --cov-report=term-missing \
  --cov-fail-under=80
```

## Docker（可选）

```bash
docker-compose up --build
```

## 功能截图

### 登录界面
![登录界面](images/登录.png)

### 文件中心
![文件中心](images/%E6%96%87%E4%BB%B6%E4%B8%AD%E5%BF%83.png)

### 原文提取
![原文提取](images/%E5%8E%9F%E6%96%87%E6%8F%90%E5%8F%96.png)

### 文段优化
![文段优化示例](images/文段优化1.png)
![文段优化示例](images/文段优化3.png)
![文段优化结果](images/文段优化4.png)

### 论文问答
![论文问答](images/论文问答.png)
![问答示例](images/论文问答2.png)

### 思维导图
![思维导图](images/思维导图.png)

## 贡献

欢迎提交 Issue / PR。
