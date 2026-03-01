# RAG 升级技术方案（Chunking + Rerank + 全链路）

**日期**: 2026-02-28  
**适用项目**: `LLM_App_Final`（本地优先的文献阅读 Agent）

## 1. 背景与目标

当前实现在 `agent/local_rag.py` 中采用了：

1. `RecursiveCharacterTextSplitter` 进行固定字符级切分；
2. Dense 向量召回（FastEmbed + InMemoryVectorStore）；
3. Top-K 直接拼接给生成模型。

这一版是可靠 baseline，但在论文场景会遇到：

1. 章节语义边界经常被打断；
2. 候选召回质量不足时，Top-K 直接拼接会放大噪声；
3. 对复杂问题（跨段、多跳）稳定性不足。

**升级目标**：

1. 提升检索相关性和证据覆盖率；
2. 在保持本地部署能力的前提下加入 rerank；
3. 建立可迭代的 RAG 评测闭环（不仅“能跑”，要“可量化改进”）。

## 2. 业界最新实践（与本项目强相关）

### 2.1 文本切分（Chunking）

1. **Recursive splitter 仍是通用基线**  
   LangChain 官方仍将 `RecursiveCharacterTextSplitter` 作为通用文本推荐起点。  
2. **语义切分（Semantic Splitter）成为重要增强**  
   通过句级语义相似度判断断点，能减少“语义断裂 chunk”。  
3. **结构感知切分（Markdown/标题/段落）优先于纯长度切分**  
   在论文/技术文档中，按章节结构先切再细分，通常比纯长度切分更稳。

### 2.2 检索策略（Retrieval）

1. **Hybrid 检索（Dense + Sparse/BM25）** 已成为高质量 RAG 常见配置；  
2. **多阶段检索** 是主流：先召回较大候选集，再 rerank 压缩；  
3. **Query 重写/拆分** 可显著提升复杂问题命中率（OpenAI `file_search` 默认就包含该策略）。

### 2.3 Rerank

1. Rerank 已成为“召回后净化噪声”的标准步骤；  
2. 本地可选小模型路径成熟（如 FlashRank 的 cross-encoder 小模型）；  
3. 若追求中文/多语效果，可考虑更强但更重的 multilingual reranker（如 BGE/Jina 系列）。

### 2.4 评测

RAG 实践从“主观看回答”升级为“指标驱动”，常见指标包括：

1. `faithfulness`（答案是否由上下文支持）
2. `context_precision`（检索排序是否把相关 chunk 放前面）
3. `context_recall`（相关证据是否被检索到）

## 3. 目标架构（建议）

## 3.1 离线预处理

1. 文档解析（保留章节、页码、段落、标题路径元数据）  
2. 分层切分：
   - 一级：结构切分（按章节/标题）
   - 二级：语义切分（句组相似度断点）
   - 三级：长度兜底（token/字符上限）
3. 生成 chunk 元数据：
   - `chunk_id`
   - `section_path`
   - `source_page`
   - `prev_chunk_id` / `next_chunk_id`

## 3.2 索引构建

1. Dense 向量索引（现有 FastEmbed 路径保留）
2. Sparse 索引（建议加 BM25）
3. 可选：离线 contextualized chunk（为 chunk 加简短上下文说明）

## 3.3 在线检索

1. Query 预处理（同义改写/多子问题拆分，可开关）
2. 多路召回：
   - Dense Top-N
   - Sparse Top-N
3. 融合（RRF）
4. Rerank（本地 cross-encoder）
5. 邻域扩展（保留前后 chunk 作为上下文补全）
6. 最终 Top-K 入模生成（带证据引用）

## 3.4 生成与可观测

1. 回答输出必须包含证据来源（chunk id / section）
2. 记录检索轨迹：
   - 候选数
   - rerank 前后排序
   - 最终入模 chunk 列表
3. 记录失败原因：
   - 无命中
   - 命中但低分
   - 证据冲突

## 4. 本项目参数建议（初版）

基于“中文论文 + 本地 CPU 优先”建议以下起步值：

1. 切分：
   - `chunk_size`: 450~700（字符）或 300~450（token）
   - `chunk_overlap`: 60~120
   - 章节优先切分后再做语义细分
2. 检索：
   - Dense 候选：20~40
   - Sparse 候选：20~40
   - RRF 融合后候选：30~60
3. Rerank：
   - rerank 输入候选：30~80
   - rerank 输出 Top-K：6~12
4. 入模上下文：
   - 最终注入 4~8 chunks
   - 每 chunk 附来源标签

## 5. 本地 rerank 模型选型建议

### 5.1 轻量优先（第一阶段）

1. **FlashRank + `ms-marco-MiniLM-L-12-v2`**  
   优点：部署轻、CPU 友好、接入简单。  
   风险：英文优势明显，中文效果需实测。

### 5.2 多语优先（第二阶段）

1. **FlashRank multilingual 模型（如 MultiBERT 路径）**  
2. **BAAI `bge-reranker-v2-m3`（更强但更重）**  
   适合在离线评测显示中文检索瓶颈时再升级。

## 6. 对现有代码的改造建议

## 6.1 短期（1-3 天）

1. 在 `agent/settings.py` 增加 RAG 分阶段参数（候选 K、rerank 开关、rerank model）
2. 在 `agent/local_rag.py` 落地“两阶段检索 + rerank”并保持失败回退（已部分起步）
3. 增加最小单测：
   - rerank 函数排序逻辑
   - rerank 失败 fallback

## 6.2 中期（3-7 天）

1. 引入结构化 chunk（标题/章节路径）
2. 引入 sparse 检索（BM25）并做 RRF
3. 输出证据结构到前端（chunk id + 节点定位）

## 6.3 长期（1-2 周）

1. 语义切分策略（sentence-group semantic breakpoint）
2. contextual retrieval（离线 chunk 上下文增强）
3. 固定评测集 + 回归门禁（PR 必跑）

## 7. 评测方案（必须落地）

建议建立 50~100 条固定问答集，覆盖：

1. 单段事实定位
2. 跨章节推理
3. 术语解释
4. 方法比较
5. 否定/边界条件问题

指标：

1. 检索层：`Recall@K`、`nDCG@K`
2. 生成层：`Faithfulness`、`Context Precision`、`Answer Relevance`
3. 工程层：P95 延迟、平均 token 成本、失败率

## 8. 风险与对策

1. **风险**：rerank 增加时延  
   **对策**：仅 rerank Top-30/Top-50，设置超时与降级
2. **风险**：中文效果不稳定  
   **对策**：多模型 A/B（MiniLM vs multilingual vs bge-reranker）
3. **风险**：复杂化过快影响稳定性  
   **对策**：分阶段开关上线，保留 dense-only fallback

## 9. 结论（建议优先级）

1. **立即做**：两阶段检索 + 本地 rerank + 指标化评测  
2. **随后做**：结构化/语义切分 + hybrid 检索  
3. **最后做**：contextual retrieval 与更重模型

---

## 参考资料（主要来源）

1. LangChain recursive splitter（Python）  
   https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter
2. LangChain RAG 指南（Python）  
   https://docs.langchain.com/oss/python/langchain/rag
3. OpenAI File Search（默认 retrieval / rerank 策略）  
   https://platform.openai.com/docs/assistants/tools/file-search
4. Anthropic Contextual Retrieval（含 hybrid + rerank 实验）  
   https://www.anthropic.com/research/contextual-retrieval
5. Qdrant Hybrid Search with Reranking  
   https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/
6. Qdrant + ColBERT/FastEmbed 说明（多阶段检索建议）  
   https://qdrant.tech/documentation/fastembed/fastembed-colbert/
7. LlamaIndex Semantic Splitter  
   https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/semantic_splitter/
8. LlamaIndex Markdown Parser（结构感知切分）  
   https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/markdown/
9. BAAI bge-reranker-v2-m3  
   https://huggingface.co/BAAI/bge-reranker-v2-m3
10. Jina Reranker v2 multilingual  
    https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual
11. FlashRank（本地轻量 rerank）  
    https://github.com/PrithivirajDamodaran/FlashRank
12. Ragas Faithfulness / Context Precision  
    https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/  
    https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
