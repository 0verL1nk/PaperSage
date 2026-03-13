"""Skill loader with progressive disclosure support.

This loader keeps SKILL.md as the primary entry point while indexing optional
resources (`references/`, `scripts/`, `assets/`, `agents/openai.yaml`) so
runtime callers can selectively load only what is needed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TASK_TOKEN_PATTERN = re.compile(r"[a-z0-9_\-\u4e00-\u9fff]+")


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    skill_path: Path
    keywords: str = "" 


@dataclass(frozen=True)
class SkillResources:
    references: list[Path] = field(default_factory=list)
    scripts: list[Path] = field(default_factory=list)
    assets: list[Path] = field(default_factory=list)
    agent_metadata: Path | None = None


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    skill_path: Path
    resources: SkillResources
    keywords: str = "" 


def _extract_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not isinstance(content, str) or not content.startswith("---"):
        return {}, str(content or "").strip()
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, str(content or "").strip()

    frontmatter_raw = parts[1]
    body = str(parts[2] or "").strip()
    payload: dict[str, str] = {}
    for raw_line in frontmatter_raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key not in {"name", "description", "keywords"}:
            continue
        value = raw_value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        payload[normalized_key] = value.strip()
    return payload, body


def _list_files(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    return sorted([item for item in path.rglob("*") if item.is_file()])


def _task_terms(task: str) -> set[str]:
    value = str(task or "").lower()
    return {item for item in _TASK_TOKEN_PATTERN.findall(value) if len(item) >= 2}


def _score_reference(path: Path, task_terms: set[str]) -> int:
    stem_terms = _task_terms(path.stem.replace("-", " ").replace("_", " "))
    if not task_terms:
        return 0
    return len(task_terms & stem_terms)


def _relative_text(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def _read_reference_excerpt(path: Path, char_limit: int) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(content) <= char_limit:
        return content.strip()
    clipped = content[:char_limit].rstrip()
    return f"{clipped}\n...(truncated)"


class SkillLoader:
    def __init__(self, skills_dir: str | None = None):
        self.skills_dir = Path(skills_dir) if skills_dir else Path(__file__).parent
        self._cache: dict[str, Skill] = {}
        self._metadata_cache: dict[str, SkillMetadata] = {}

    def discover_skills(self) -> list[SkillMetadata]:
        if not self.skills_dir.exists():
            logger.warning("Skills directory not found: %s", self.skills_dir)
            return []

        skills: list[SkillMetadata] = []
        for skill_path in self.skills_dir.iterdir():
            if not skill_path.is_dir():
                continue
            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                metadata = self._parse_skill_metadata(skill_md)
            except Exception as exc:
                logger.warning("Failed to parse skill %s: %s", skill_path.name, exc)
                continue
            skills.append(metadata)
            self._metadata_cache[metadata.name] = metadata
        return skills

    def get_skill(self, name: str) -> Skill | None:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            return None
        if normalized_name in self._cache:
            return self._cache[normalized_name]

        skill_path = self.skills_dir / normalized_name
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            logger.warning("Skill not found: %s", normalized_name)
            return None
        try:
            skill = self._parse_skill(skill_md, skill_path)
        except Exception as exc:
            logger.error("Failed to load skill %s: %s", normalized_name, exc)
            return None
        self._cache[normalized_name] = skill
        return skill

    @staticmethod
    def _resource_index(skill_path: Path) -> SkillResources:
        references = _list_files(skill_path / "references")
        scripts = _list_files(skill_path / "scripts")
        assets = _list_files(skill_path / "assets")
        agent_metadata = skill_path / "agents" / "openai.yaml"
        return SkillResources(
            references=references,
            scripts=scripts,
            assets=assets,
            agent_metadata=agent_metadata if agent_metadata.exists() else None,
        )

    def _parse_skill_metadata(self, skill_md: Path) -> SkillMetadata:
        content = skill_md.read_text(encoding="utf-8")
        frontmatter, _body = _extract_frontmatter(content)
        name = str(frontmatter.get("name") or skill_md.parent.name).strip()
        description = str(frontmatter.get("description") or "").strip()
        return SkillMetadata(
            name=name,
            description=description,
            skill_path=skill_md.parent,
        )

    def _parse_skill(self, skill_md: Path, skill_path: Path) -> Skill:
        content = skill_md.read_text(encoding="utf-8")
        frontmatter, body = _extract_frontmatter(content)
        name = str(frontmatter.get("name") or skill_path.name).strip()
        description = str(frontmatter.get("description") or "").strip()
        instructions = body if body else content
        return Skill(
            name=name,
            description=description,
            instructions=instructions,
            skill_path=skill_path,
            resources=self._resource_index(skill_path),
        )


_default_loader: SkillLoader | None = None


def get_skill_loader() -> SkillLoader:
    global _default_loader
    if _default_loader is None:
        _default_loader = SkillLoader()
    return _default_loader


def discover_available_skills() -> list[SkillMetadata]:
    return get_skill_loader().discover_skills()


def get_skill(name: str) -> Skill | None:
    return get_skill_loader().get_skill(name)


def build_skill_runtime_payload(
    name: str,
    *,
    task: str = "",
    max_references: int = 2,
    reference_char_limit: int = 1800,
) -> dict[str, Any] | None:
    skill = get_skill(name)
    if skill is None:
        return None

    references = skill.resources.references
    task_terms = _task_terms(task)
    ranked_references = sorted(
        references,
        key=lambda path: (_score_reference(path, task_terms), path.name),
        reverse=True,
    )
    if ranked_references and _score_reference(ranked_references[0], task_terms) <= 0:
        ranked_references = sorted(references, key=lambda item: item.name)
    selected_references = ranked_references[: max(0, int(max_references))]

    payload: dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
        "instructions": skill.instructions,
        "references": [
            {
                "path": _relative_text(path, skill.skill_path),
                "content": _read_reference_excerpt(path, max(1, int(reference_char_limit))),
            }
            for path in selected_references
        ],
        "scripts": [
            _relative_text(path, skill.skill_path) for path in skill.resources.scripts
        ],
        "assets": [
            _relative_text(path, skill.skill_path) for path in skill.resources.assets
        ],
    }
    if skill.resources.agent_metadata is not None:
        payload["agent_metadata"] = _relative_text(
            skill.resources.agent_metadata,
            skill.skill_path,
        )
    return payload
