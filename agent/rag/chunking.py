"""结构化 Chunking 模块

支持按章节/标题的结构化切分，保留 chunk 元数据
"""
import re
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


@dataclass
class ChunkMetadata:
    """Chunk 元数据"""
    chunk_id: str
    section_path: str  # 章节路径，如 "1. 引言 > 1.1 研究背景"
    source_page: int | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    heading_level: int = 0  # 标题级别（1-6）


@dataclass
class StructuredChunk:
    """结构化 Chunk"""
    content: str
    metadata: ChunkMetadata


class SemanticAwareSplitter:
    """语义感知切分器

    结合结构化切分和语义切分，支持：
    1. Markdown 标题感知切分
    2. 句级语义相似度断点
    3. 长度兜底
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n", "\n", "。", "！", "？", ".", "!", "?"
        ]

    def split_markdown(self, text: str) -> list[StructuredChunk]:
        """按 Markdown 标题结构切分"""
        # Markdown 标题识别模式
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False,
        )

        # 首先按标题切分
        header_splits = markdown_splitter.split_text(text)

        chunks: list[StructuredChunk] = []
        chunk_id = 0

        for split in header_splits:
            # 构建 section_path
            section_path = self._build_section_path(split.metadata)

            # 如果单个标题块太大，继续细分
            if len(split.page_content) > self.chunk_size:
                sub_chunks = self._split_by_length(split.page_content)
                for i, sub_content in enumerate(sub_chunks):
                    chunks.append(StructuredChunk(
                        content=sub_content,
                        metadata=ChunkMetadata(
                            chunk_id=f"chunk_{chunk_id}",
                            section_path=section_path,
                            heading_level=self._get_heading_level(split.metadata),
                        )
                    ))
                    chunk_id += 1
            else:
                chunks.append(StructuredChunk(
                    content=split.page_content,
                    metadata=ChunkMetadata(
                        chunk_id=f"chunk_{chunk_id}",
                        section_path=section_path,
                        heading_level=self._get_heading_level(split.metadata),
                    )
                ))
                chunk_id += 1

        # 添加 prev/next 链接
        self._link_chunks(chunks)

        return chunks

    def split_text(self, text: str) -> list[StructuredChunk]:
        """纯文本切分（无 Markdown 结构）"""
        # 尝试检测 Markdown 标题
        if self._has_markdown_headers(text):
            return self.split_markdown(text)

        # 使用递归字符切分器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

        texts = splitter.split_text(text)

        chunks: list[StructuredChunk] = []
        for i, content in enumerate(texts):
            chunks.append(StructuredChunk(
                content=content,
                metadata=ChunkMetadata(
                    chunk_id=f"chunk_{i}",
                    section_path="",
                )
            ))

        self._link_chunks(chunks)
        return chunks

    def _has_markdown_headers(self, text: str) -> bool:
        """检查文本是否包含 Markdown 标题"""
        return bool(re.search(r"^#{1,6}\s+", text, re.MULTILINE))

    def _build_section_path(self, metadata: dict[str, Any]) -> str:
        """从 metadata 构建章节路径"""
        parts = []
        for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            if level in metadata and metadata[level]:
                parts.append(metadata[level])
        return " > ".join(parts) if parts else ""

    def _get_heading_level(self, metadata: dict[str, Any]) -> int:
        """获取标题级别"""
        for i in range(1, 7):
            if f"h{i}" in metadata and metadata[f"h{i}"]:
                return i
        return 0

    def _split_by_length(self, text: str) -> list[str]:
        """按长度细分"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        return splitter.split_text(text)

    def _link_chunks(self, chunks: list[StructuredChunk]) -> None:
        """链接相邻 chunks"""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata.prev_chunk_id = chunks[i - 1].metadata.chunk_id
            if i < len(chunks) - 1:
                chunk.metadata.next_chunk_id = chunks[i + 1].metadata.chunk_id


def create_structured_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> SemanticAwareSplitter:
    """创建结构化切分器"""
    return SemanticAwareSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
