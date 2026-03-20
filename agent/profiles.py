from dataclasses import dataclass
from typing import Callable

from .paper_prompt import (
    build_paper_reviewer_prompt,
    build_paper_system_prompt,
    build_paper_worker_prompt,
)

PromptBuilder = Callable[..., str]


@dataclass(frozen=True)
class AgentProfile:
    name: str
    description: str
    prompt_builder: PromptBuilder
    capability_ids: tuple[str, ...] = ()
    middleware_ids: tuple[str, ...] = ()
    allow_team: bool = False
    allow_global_planning: bool = False
    allow_web: bool = False


paper_leader_profile = AgentProfile(
    name="paper_leader",
    description="Leader agent for paper reading and orchestrated collaboration.",
    prompt_builder=build_paper_system_prompt,
    capability_ids=("document_pack", "planning_pack", "team_pack", "skill_pack", "web_pack"),
    middleware_ids=("trace", "llm_logger", "orchestration", "team", "todolist", "plan"),
    allow_team=True,
    allow_global_planning=True,
    allow_web=True,
)


paper_worker_profile = AgentProfile(
    name="paper_worker",
    description="Worker agent for bounded execution tasks.",
    prompt_builder=build_paper_worker_prompt,
    capability_ids=("document_pack", "skill_pack"),
    middleware_ids=("trace", "llm_logger", "orchestration"),
    allow_team=False,
    allow_global_planning=False,
    allow_web=False,
)


paper_reviewer_profile = AgentProfile(
    name="paper_reviewer",
    description="Reviewer agent for critiquing intermediate outputs.",
    prompt_builder=build_paper_reviewer_prompt,
    capability_ids=("document_pack", "skill_pack"),
    middleware_ids=("trace", "llm_logger", "orchestration"),
    allow_team=False,
    allow_global_planning=False,
    allow_web=False,
)


PROFILE_REGISTRY: dict[str, AgentProfile] = {
    profile.name: profile
    for profile in (
        paper_leader_profile,
        paper_worker_profile,
        paper_reviewer_profile,
    )
}

PROFILE_ALIASES: dict[str, str] = {
    "leader": paper_leader_profile.name,
    "teammate": paper_worker_profile.name,
    "worker": paper_worker_profile.name,
    "reviewer": paper_reviewer_profile.name,
}


def resolve_agent_profile(name: str) -> AgentProfile:
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        raise ValueError("Profile name is required")

    profile_name = PROFILE_ALIASES.get(normalized_name, normalized_name)
    profile = PROFILE_REGISTRY.get(profile_name)
    if profile is None:
        available = ", ".join(sorted(PROFILE_ALIASES))
        raise ValueError(f"Unknown profile/role '{name}'. Available roles: {available}")
    return profile
