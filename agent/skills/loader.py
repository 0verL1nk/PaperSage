"""Skill 加载器模块

实现 Agent Skills 的动态加载
支持渐进式披露：只加载 name/description 用于发现，完整加载 SKILL.md 用于执行
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    """Skill 元数据"""
    name: str
    description: str
    skill_path: Path


@dataclass
class Skill:
    """完整 Skill"""
    name: str
    description: str
    instructions: str
    skill_path: Path


class SkillLoader:
    """Skill 加载器"""

    def __init__(self, skills_dir: str | None = None):
        if skills_dir is None:
            # 默认使用 agent/skills 目录
            self.skills_dir = Path(__file__).parent
        else:
            self.skills_dir = Path(skills_dir)

        self._cache: dict[str, Skill] = {}
        self._metadata_cache: dict[str, SkillMetadata] = {}

    def discover_skills(self) -> list[SkillMetadata]:
        """发现所有可用的 Skills

        Returns:
            SkillMetadata 列表（只包含 name 和 description，用于发现阶段）
        """
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return []

        skills = []
        for skill_path in self.skills_dir.iterdir():
            if not skill_path.is_dir():
                continue

            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                metadata = self._parse_skill_metadata(skill_md)
                skills.append(metadata)
                self._metadata_cache[metadata.name] = metadata
            except Exception as e:
                logger.warning(f"Failed to parse skill {skill_path.name}: {e}")

        return skills

    def get_skill(self, name: str) -> Skill | None:
        """获取完整的 Skill（用于执行阶段）

        Args:
            name: Skill 名称

        Returns:
            完整的 Skill 对象
        """
        # 检查缓存
        if name in self._cache:
            return self._cache[name]

        # 查找 skill 目录
        skill_path = self.skills_dir / name
        skill_md = skill_path / "SKILL.md"

        if not skill_md.exists():
            logger.warning(f"Skill not found: {name}")
            return None

        try:
            skill = self._parse_skill(skill_md, skill_path)
            self._cache[name] = skill
            return skill
        except Exception as e:
            logger.error(f"Failed to load skill {name}: {e}")
            return None

    def _parse_skill_metadata(self, skill_md: Path) -> SkillMetadata:
        """解析 Skill 元数据（只解析 name 和 description）"""
        content = skill_md.read_text(encoding="utf-8")

        # 解析 YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                for line in frontmatter.split("\n"):
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip('"')
                    elif line.startswith("description:"):
                        description = line.split(":", 1)[1].strip().strip('"')

        return SkillMetadata(
            name=name,
            description=description,
            skill_path=skill_md.parent,
        )

    def _parse_skill(self, skill_md: Path, skill_path: Path) -> Skill:
        """解析完整的 Skill"""
        content = skill_md.read_text(encoding="utf-8")

        # 解析 YAML frontmatter
        name = ""
        description = ""
        instructions = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                instructions = parts[2].strip()

                for line in frontmatter.split("\n"):
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip('"')
                    elif line.startswith("description:"):
                        description = line.split(":", 1)[1].strip().strip('"')

        return Skill(
            name=name,
            description=description,
            instructions=instructions,
            skill_path=skill_path,
        )


# 全局 SkillLoader 实例
_default_loader: SkillLoader | None = None


def get_skill_loader() -> SkillLoader:
    """获取全局 SkillLoader 实例"""
    global _default_loader
    if _default_loader is None:
        _default_loader = SkillLoader()
    return _default_loader


def discover_available_skills() -> list[SkillMetadata]:
    """发现所有可用的 Skills"""
    loader = get_skill_loader()
    return loader.discover_skills()


def get_skill(name: str) -> Skill | None:
    """获取 Skill（完整加载）"""
    loader = get_skill_loader()
    return loader.get_skill(name)
