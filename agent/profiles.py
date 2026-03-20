from dataclasses import dataclass
from typing import Callable

from .paper_prompt import build_paper_system_prompt

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
    capability_ids=("document_pack", "planning_pack", "team_pack", "skill_pack"),
    middleware_ids=("trace", "llm_logger", "orchestration", "team", "todolist", "plan"),
    allow_team=True,
    allow_global_planning=True,
    allow_web=False,
)
