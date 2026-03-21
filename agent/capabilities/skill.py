from typing import Any

from ..tools.skill import use_skill


def build_skill_tools(_deps: Any) -> list[Any]:
    return [use_skill]
