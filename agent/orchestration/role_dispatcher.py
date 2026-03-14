from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent.domain.orchestration import TeamRole


class RoleGenerator(Protocol):
    def __call__(self, prompt: str, llm: Any | None, *, max_members: int) -> list[TeamRole]: ...


class RoleNormalizer(Protocol):
    def __call__(self, raw_roles: list[TeamRole], *, max_members: int) -> list[TeamRole]: ...


@dataclass(frozen=True)
class RoleDispatchPlan:
    roles: list[TeamRole]
    role_map: dict[str, TeamRole]
    role_order: dict[str, int]

    @classmethod
    def from_roles(cls, roles: list[TeamRole]) -> "RoleDispatchPlan":
        role_map = {item.name: item for item in roles}
        role_order = {item.name: idx for idx, item in enumerate(roles)}
        return cls(roles=list(roles), role_map=role_map, role_order=role_order)


def build_role_dispatch_plan(
    *,
    prompt: str,
    llm: Any | None,
    max_members: int,
    role_generator: RoleGenerator,
    role_normalizer: RoleNormalizer,
) -> RoleDispatchPlan:
    bounded_members = max(1, int(max_members))
    generated_roles = role_generator(prompt, llm, max_members=bounded_members)
    normalized_roles = role_normalizer(generated_roles, max_members=bounded_members)
    return RoleDispatchPlan.from_roles(normalized_roles)
