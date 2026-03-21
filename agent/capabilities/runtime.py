from typing import Any

from ..tools.wait import sleep


def build_runtime_tools(_deps: Any) -> list[Any]:
    return [sleep]
