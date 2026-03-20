from typing import Any


def build_team_tools(_deps: Any) -> list[Any]:
    # Team tools are still injected by TeamMiddleware in the current phase.
    # Keep the capability id explicit so session/profile boundaries are ready
    # before middleware filtering is introduced in the next phase.
    return []
