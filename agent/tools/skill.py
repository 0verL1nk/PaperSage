"""技能工具模块"""
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..skills.loader import build_skill_runtime_payload, discover_available_skills
from .utils import _is_dangerous_query

logger = logging.getLogger(__name__)


class SkillInput(BaseModel):
    skill_name: str = Field(
        description=(
            "Skill template name. Available skills include: summary, critical_reading, "
            "method_compare, translation, mindmap, agentic_search."
        )
    )
    task: str = Field(
        description="Current user task where the selected skill guidance should be applied."
    )


def _get_skill_options() -> str:
    """获取可用的 skill 列表"""
    skills = discover_available_skills()
    if skills:
        return ", ".join(sorted(s.name for s in skills))
    return "summary, critical_reading, method_compare, translation, mindmap"


def build_skill_tool() -> Any:
    """构建技能工具"""

    @tool(
        "use_skill",
        description="Apply a named skill template to the current task and return operational guidance.",
        args_schema=SkillInput,
    )
    def use_skill(skill_name: str, task: str) -> str:
        normalized_name = skill_name.strip().lower()
        logger.info("tool.use_skill called: skill=%s task_len=%s", normalized_name, len(task))
        if _is_dangerous_query(task):
            logger.warning("tool.use_skill blocked by policy")
            return "Blocked by tool policy: task appears unsafe."

        runtime_payload = build_skill_runtime_payload(
            normalized_name,
            task=task,
            max_references=2,
            reference_char_limit=1800,
        )
        if runtime_payload is not None:
            logger.info("tool.use_skill success: skill=%s", normalized_name)
            parts: list[str] = [
                f"Skill: {runtime_payload['name']}",
                f"Description: {runtime_payload['description']}",
                "",
                "Instructions:",
                str(runtime_payload["instructions"]),
            ]

            references = runtime_payload.get("references", [])
            if isinstance(references, list) and references:
                parts.extend(["", "Selected references:"])
                for item in references:
                    if not isinstance(item, dict):
                        continue
                    path_value = str(item.get("path") or "").strip()
                    content_value = str(item.get("content") or "").strip()
                    if not path_value or not content_value:
                        continue
                    parts.append(f"- {path_value}")
                    parts.append(content_value)

            scripts = runtime_payload.get("scripts", [])
            if isinstance(scripts, list) and scripts:
                parts.extend(["", "Available scripts:"])
                parts.extend(f"- {str(item)}" for item in scripts if str(item).strip())

            agent_metadata = runtime_payload.get("agent_metadata")
            if isinstance(agent_metadata, str) and agent_metadata.strip():
                parts.extend(["", f"Agent metadata: {agent_metadata}"])

            parts.extend(["", f"Task: {task}"])
            return "\n".join(parts)

        options = _get_skill_options()
        logger.warning("tool.use_skill unknown skill: %s", normalized_name)
        return f"Unknown skill '{skill_name}'. Available skills: {options}."

    return use_skill
