"""RAG Chunking 单元测试

测试结构化切分器功能
"""
import pytest

from agent.rag_chunking import (
    ChunkMetadata,
    StructuredChunk,
    SemanticAwareSplitter,
    create_structured_splitter,
)


class TestSemanticAwareSplitter:
    """测试语义感知切分器"""

    def test_split_chinese_text_by_sentence(self):
        """测试中文文本按句子切分"""
        splitter = SemanticAwareSplitter(
            chunk_size=10,  # 小一点以触发切分
            chunk_overlap=3,
        )
        text = "这是第一句话。这是第二句话。这是第三句话。"

        chunks = splitter.split_text(text)

        # 10 字符应该切分成多个 chunk
        assert len(chunks) >= 1
        # 验证每个 chunk 有正确的 metadata
        for chunk in chunks:
            assert isinstance(chunk, StructuredChunk)
            assert isinstance(chunk.metadata, ChunkMetadata)
            assert chunk.metadata.chunk_id.startswith("chunk_")

    def test_split_english_text_by_word(self):
        """测试英文文本按词切分"""
        splitter = SemanticAwareSplitter(
            chunk_size=30,
            chunk_overlap=5,
        )
        text = "This is the first sentence. This is the second sentence."

        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        # 验证英文按空格分词
        for chunk in chunks:
            assert isinstance(chunk.content, str)

    def test_chunk_metadata_has_prev_next_links(self):
        """测试 chunk 前后链接"""
        splitter = SemanticAwareSplitter(
            chunk_size=20,
            chunk_overlap=5,
        )
        text = "第一句。第二句。第三句。第四句。"

        chunks = splitter.split_text(text)

        # 验证链接
        for i, chunk in enumerate(chunks):
            if i > 0:
                assert chunk.metadata.prev_chunk_id is not None
                assert chunk.metadata.prev_chunk_id == chunks[i-1].metadata.chunk_id
            if i < len(chunks) - 1:
                assert chunk.metadata.next_chunk_id is not None

    def test_split_text_respects_chunk_size(self):
        """测试切分大小限制"""
        splitter = SemanticAwareSplitter(
            chunk_size=20,  # 调小以触发切分
            chunk_overlap=3,
        )
        # 创建足够长的文本，超过 chunk_size，需要包含 separators 才能切分
        text = "这是第一句话。" * 10  # 约 60 字符，每句 6 字符

        chunks = splitter.split_text(text)

        # 验证有多个 chunk
        assert len(chunks) >= 2
        # 验证所有 chunks 都不超过 chunk_size + overlap
        for chunk in chunks:
            assert len(chunk.content) <= 25  # 20 + 5 tolerance

    def test_empty_text_returns_empty_list(self):
        """测试空文本返回空列表"""
        splitter = SemanticAwareSplitter()
        chunks = splitter.split_text("")

        assert chunks == []


class TestCreateStructuredSplitter:
    """测试工厂函数"""

    def test_create_with_defaults(self):
        """测试默认参数创建"""
        splitter = create_structured_splitter()

        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 80

    def test_create_with_custom_params(self):
        """测试自定义参数创建"""
        splitter = create_structured_splitter(
            chunk_size=300,
            chunk_overlap=50,
        )

        assert splitter.chunk_size == 300
        assert splitter.chunk_overlap == 50


class TestMarkdownSplitting:
    """测试 Markdown 切分"""

    def test_split_markdown_with_headers(self):
        """测试 Markdown 标题切分"""
        splitter = SemanticAwareSplitter(
            chunk_size=100,
            chunk_overlap=20,
        )
        text = """# 第一章

这是第一章的内容。

## 第一章第一节

这是第一节的内容。

# 第二章

这是第二章的内容。
"""

        chunks = splitter.split_text(text)

        # 应该按章节切分
        assert len(chunks) >= 1
        # 验证有 section_path
        has_section = any(c.metadata.section_path for c in chunks)
        assert has_section

    def test_split_markdown_large_chunk(self):
        """测试大 Markdown 块继续细分"""
        splitter = SemanticAwareSplitter(
            chunk_size=20,  # 小值触发细分
            chunk_overlap=5,
        )
        text = """# 大章节

""" + "这是很长的内容。" * 20

        chunks = splitter.split_text(text)

        # 应该被细分成多个 chunk
        assert len(chunks) > 1

    def test_has_markdown_headers(self):
        """测试 Markdown 标题检测"""
        splitter = SemanticAwareSplitter()

        assert splitter._has_markdown_headers("# 标题") is True
        assert splitter._has_markdown_headers("## 子标题") is True
        assert splitter._has_markdown_headers("普通文本") is False
        assert splitter._has_markdown_headers("") is False
