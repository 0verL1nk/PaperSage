# Evidence-Grounded QA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为论文问答页实现“证据回链问答”：每次回答都展示可追溯的证据片段与来源定位信息。

**Architecture:** 在本地 RAG 检索层新增结构化证据输出（chunk id、片段、相关度、定位信息），问答页在回答后执行一次同查询证据检索并渲染“证据面板”。先做单文档路径，保持现有 agent 主流程不破坏。所有证据展示均为只读，不改变模型推理链。

**Tech Stack:** Python, Streamlit, LangChain, FastEmbed, InMemoryVectorStore, pytest

---

### Task 1: 定义证据数据结构与格式化函数

**Files:**
- Create: `utils/evidence.py`
- Test: `tests/unit/test_evidence.py`

**Step 1: Write the failing test**

```python
from utils.evidence import EvidenceChunk, format_evidence_markdown


def test_format_evidence_markdown_contains_source_id_and_score():
    chunks = [
        EvidenceChunk(
            chunk_id="c-0001",
            text="Transformer 结构包含自注意力机制。",
            score=0.823,
            locator="chunk:1",
        )
    ]

    rendered = format_evidence_markdown(chunks)

    assert "证据 1" in rendered
    assert "c-0001" in rendered
    assert "0.82" in rendered
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_evidence.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors.

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    text: str
    score: float
    locator: str


def format_evidence_markdown(chunks: list[EvidenceChunk]) -> str:
    if not chunks:
        return "未检索到可展示的证据片段。"
    lines: list[str] = []
    for idx, c in enumerate(chunks, start=1):
        lines.append(f"**证据 {idx}** | 来源ID: `{c.chunk_id}` | 相关度: `{c.score:.2f}` | 定位: `{c.locator}`")
        lines.append(c.text)
    return "\n\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_evidence.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/evidence.py tests/unit/test_evidence.py
git commit -m "feat: add evidence chunk model and markdown formatter"
```

### Task 2: 为本地检索新增结构化证据返回接口

**Files:**
- Modify: `utils/local_rag.py`
- Test: `tests/unit/test_local_rag_evidence.py`

**Step 1: Write the failing test**

```python
from utils.local_rag import build_local_evidence_retriever


def test_build_local_evidence_retriever_returns_structured_chunks():
    retriever = build_local_evidence_retriever("A\nB\nC")
    chunks = retriever("A")

    assert isinstance(chunks, list)
    assert chunks
    first = chunks[0]
    assert first.chunk_id
    assert isinstance(first.score, float)
    assert first.locator.startswith("chunk:")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_local_rag_evidence.py -q`
Expected: FAIL with missing function errors.

**Step 3: Write minimal implementation**

```python
def build_local_evidence_retriever(document_text: str):
    # 复用现有 split + embedding + vectorstore
    # 返回 list[EvidenceChunk]
    # chunk_id 使用稳定前缀 + 序号，如 c-0001
    # locator 先使用 chunk:<index> 形式
    ...
```

Implementation constraints:
- 不移除 `build_local_vector_retriever`，避免破坏现有调用方。
- 证据文本长度做上限截断（例如 400-600 字符），防止 UI 过长。
- 分数缺失时回落为 0.0，但要保持类型稳定。

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_local_rag_evidence.py tests/unit/test_agent_stream.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/local_rag.py tests/unit/test_local_rag_evidence.py
git commit -m "feat: add structured evidence retriever for local rag"
```

### Task 3: 在问答页渲染证据回链面板

**Files:**
- Modify: `pages/4_🤖_论文问答.py`
- Modify: `utils/agent_capabilities.py` (如需新增工具描述约束)
- Test: `tests/integration/test_pages_boot_e2e.py`

**Step 1: Write the failing integration test**

```python
def test_qa_page_renders_evidence_panel_after_answer(monkeypatch):
    # mock: agent 返回固定答案
    # mock: evidence retriever 返回 2 条 EvidenceChunk
    # assert 页面包含 "证据 1" 与来源ID
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_pages_boot_e2e.py -k evidence -q`
Expected: FAIL because evidence panel is not rendered yet.

**Step 3: Write minimal implementation**

```python
# 在 _ensure_agent 阶段同时构建 evidence retriever，并存入 session_state
st.session_state.agent_evidence_retriever = build_local_evidence_retriever(document_text)

# 在回答生成后执行一次证据检索
evidence_chunks = st.session_state.agent_evidence_retriever(prompt)
st.markdown("### 证据回链")
st.markdown(format_evidence_markdown(evidence_chunks))
```

Rendering rules:
- 默认展示 top 3 证据。
- 使用 `st.expander("查看证据详情")` 包裹，减少主回答区噪音。
- 当无证据时显示明确文案，不显示空白区。

**Step 4: Run tests to verify they pass**

Run: `pytest tests/integration/test_pages_boot_e2e.py -k evidence -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add pages/4_🤖_论文问答.py utils/agent_capabilities.py tests/integration/test_pages_boot_e2e.py
git commit -m "feat: render evidence backlink panel in qa page"
```

### Task 4: 回归验证与验收标准

**Files:**
- Test: `tests/unit/test_agent_stream.py`
- Test: `tests/integration/test_streamlit_e2e.py`
- Test: `tests/integration/test_pages_boot_e2e.py`

**Step 1: Run targeted unit tests**

Run: `pytest tests/unit/test_evidence.py tests/unit/test_local_rag_evidence.py tests/unit/test_agent_stream.py -q`
Expected: PASS.

**Step 2: Run integration tests**

Run: `pytest tests/integration/test_pages_boot_e2e.py tests/integration/test_streamlit_e2e.py -q`
Expected: PASS.

**Step 3: Run project type/lint checks (if configured)**

Run: `pytest -q`
Expected: PASS or only pre-existing failures (must be documented).

**Step 4: Manual acceptance check**

Run: `streamlit run 文件中心.py`
Expected:
- 在论文问答页提问后出现“证据回链”区域。
- 每条证据包含来源ID、相关度、定位标识。
- 切换文档后证据来源随文档变化。

**Step 5: Commit**

```bash
git add docs/plans/2026-02-27-evidence-grounded-qa-implementation-plan.md
git commit -m "docs: add implementation plan for evidence-grounded qa"
```

---

## 验收定义（Definition of Done）

1. 论文问答每次回答都可展示结构化证据（至少 top 3）。
2. 证据包含最小可追溯定位信息（chunk id + locator）。
3. 现有问答流式体验不倒退。
4. 单测与集成测试通过，且无新增错误。
5. 无硬编码模型名，无 `as any`、无忽略类型错误行为。

## 非目标（本轮不做）

- 不实现跨多文档全局证据图谱。
- 不实现逐句自动引用对齐（statement-level grounding）。
- 不引入新数据库表（先走内存态 + 会话态）。
