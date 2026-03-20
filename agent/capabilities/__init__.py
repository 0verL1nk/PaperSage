from collections.abc import Callable
from typing import Any

from .document import build_document_tools
from .planning import build_planning_tools
from .skill import build_skill_tools
from .team import build_team_tools
from .web import build_web_tools


def build_profile_tools(profile: Any, deps: Any) -> list[Any]:
    builders: dict[str, Callable[[Any], list[Any]]] = {
        "document_pack": build_document_tools,
        "planning_pack": build_planning_tools,
        "skill_pack": build_skill_tools,
        "team_pack": build_team_tools,
        "web_pack": build_web_tools,
    }
    tools: list[Any] = []
    seen_names: set[str] = set()
    for capability_id in getattr(profile, "capability_ids", ()):
        builder = builders.get(str(capability_id))
        if builder is None:
            continue
        for tool_item in builder(deps):
            name = str(getattr(tool_item, "name", "") or "").strip()
            key = name or repr(tool_item)
            if key in seen_names:
                continue
            seen_names.add(key)
            tools.append(tool_item)
    return tools


__all__ = ["build_profile_tools"]
