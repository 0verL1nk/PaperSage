from typing import Any

from ..tools.plan_tools import read_plan, write_plan


def build_planning_tools(_deps: Any) -> list[Any]:
    return [write_plan, read_plan]
