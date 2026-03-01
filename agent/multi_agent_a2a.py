from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from .capabilities import build_agent_tools
from .stream import extract_result_text


PLANNER_SYSTEM_PROMPT = (
    "You are the Planner agent in an Agent Coordination Protocol (ACP) workflow. "
    "Turn a user question into a compact plan with 2-4 steps. "
    "Focus on evidence-first academic QA. "
    "Do not output chain-of-thought, only concise plan bullets."
)

RESEARCHER_SYSTEM_PROMPT = (
    "You are the Researcher agent in an ACP workflow. "
    "Use tools to gather evidence and answer the question. "
    "Prefer search_document first, use search_web only when document evidence is insufficient. "
    "If the user asks for a mind map, output strict JSON only with this shape: "
    '{"name":"topic","children":[{"name":"subtopic","children":[...]}]}. '
    "Output concise, structured text and mark evidence source. "
    "Use the same language as the user's latest query. "
    "If the user request is ambiguous, ask exactly one short clarification question. "
    "Never expose internal reasoning, planning text, or tool traces."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are the Reviewer agent in an ACP workflow. "
    "Check whether draft answers are grounded and complete. "
    "Output exactly two lines:\n"
    "Decision: PASS or REVISE\n"
    "Feedback: <short actionable feedback>"
)

REACT_SYSTEM_PROMPT = (
    "You are a ReAct-style academic QA agent. "
    "Use tools to gather evidence, prioritize search_document, and answer concisely with sources. "
    "If the user asks for a mind map, output strict JSON only with this shape: "
    '{"name":"topic","children":[{"name":"subtopic","children":[...]}]}. '
    "Use the same language as the user's latest query. "
    "If request scope is unclear, ask one concise clarification question. "
    "Never expose internal reasoning, planning text, or tool traces."
)

WORKFLOW_REACT = "react"
WORKFLOW_PLAN_ACT = "plan_act"
WORKFLOW_PLAN_ACT_REPLAN = "plan_act_replan"


@dataclass(frozen=True)
class A2AMessage:
    sender: str
    receiver: str
    performative: str
    content: str


@dataclass(frozen=True)
class A2AMultiAgentSession:
    coordinator: "A2AMultiAgentCoordinator"
    session_id: str


class A2AMultiAgentCoordinator:
    def __init__(
        self,
        *,
        react_agent: Any,
        planner_agent: Any,
        researcher_agent: Any,
        reviewer_agent: Any,
        session_id: str,
    ) -> None:
        self.react_agent = react_agent
        self.planner_agent = planner_agent
        self.researcher_agent = researcher_agent
        self.reviewer_agent = reviewer_agent
        self.session_id = session_id

    def _invoke(self, agent: Any, role: str, prompt: str) -> str:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": f"{self.session_id}:{role}"}},
        )
        if isinstance(result, dict):
            return extract_result_text(result)
        return str(result)

    @staticmethod
    def _needs_revision(review_text: str) -> bool:
        normalized = review_text.upper()
        if "DECISION: PASS" in normalized:
            return False
        return "REVISE" in normalized

    @staticmethod
    def _record_trace(
        trace: list[A2AMessage],
        message: A2AMessage,
        on_event: Callable[[A2AMessage], None] | None = None,
    ) -> None:
        trace.append(message)
        if on_event is None:
            return
        try:
            on_event(message)
        except Exception:
            # UI callback should never break workflow execution.
            return

    @staticmethod
    def _emit_event(
        message: A2AMessage,
        on_event: Callable[[A2AMessage], None] | None = None,
    ) -> None:
        if on_event is None:
            return
        try:
            on_event(message)
        except Exception:
            return

    def run(
        self,
        question: str,
        workflow_mode: str = WORKFLOW_PLAN_ACT_REPLAN,
        on_event: Callable[[A2AMessage], None] | None = None,
        max_replan_rounds: int = 2,
    ) -> tuple[str, list[A2AMessage]]:
        if workflow_mode == WORKFLOW_REACT:
            self._emit_event(
                A2AMessage(
                    sender="coordinator",
                    receiver="react_agent",
                    performative="dispatch",
                    content=question,
                ),
                on_event,
            )
            answer = self._invoke(
                self.react_agent,
                "react",
                (
                    "Answer the user question with evidence. "
                    "Prefer document retrieval first, then web only if needed.\n"
                    f"Question: {question}"
                ),
            )
            trace: list[A2AMessage] = []
            self._record_trace(
                trace,
                A2AMessage(
                    sender="user",
                    receiver="react_agent",
                    performative="request",
                    content=question,
                ),
                on_event,
            )
            self._record_trace(
                trace,
                A2AMessage(
                    sender="react_agent",
                    receiver="user",
                    performative="final",
                    content=answer,
                ),
                on_event,
            )
            return answer, trace

        trace: list[A2AMessage] = []
        self._record_trace(
            trace,
                A2AMessage(
                    sender="user",
                    receiver="planner",
                    performative="request",
                    content=question,
                ),
                on_event,
            )

        self._emit_event(
            A2AMessage(
                sender="coordinator",
                receiver="planner",
                performative="dispatch",
                content=question,
            ),
            on_event,
        )
        plan = self._invoke(
            self.planner_agent,
            "planner",
            (
                "Generate a compact execution plan for the question below.\n"
                f"Question: {question}"
            ),
        )
        self._record_trace(
            trace,
            A2AMessage(
                sender="planner",
                receiver="researcher",
                performative="plan",
                content=plan,
            ),
            on_event,
        )

        self._emit_event(
            A2AMessage(
                sender="coordinator",
                receiver="researcher",
                performative="dispatch",
                content=f"Question: {question}\nPlan: {plan}",
            ),
            on_event,
        )
        draft = self._invoke(
            self.researcher_agent,
            "researcher",
            (
                "Execute the plan and produce an evidence-grounded answer.\n"
                f"Question: {question}\n"
                f"Plan: {plan}"
            ),
        )
        if workflow_mode == WORKFLOW_PLAN_ACT:
            self._record_trace(
                trace,
                A2AMessage(
                    sender="researcher",
                    receiver="user",
                    performative="final",
                    content=draft,
                ),
                on_event,
            )
            return draft, trace

        bounded_rounds = max(1, max_replan_rounds)
        for round_idx in range(1, bounded_rounds + 1):
            self._record_trace(
                trace,
                A2AMessage(
                    sender="researcher",
                    receiver="reviewer",
                    performative="draft",
                    content=f"[round={round_idx}]\n{draft}",
                ),
                on_event,
            )
            self._emit_event(
                A2AMessage(
                    sender="coordinator",
                    receiver="reviewer",
                    performative="dispatch",
                    content=f"[round={round_idx}]\n{question}",
                ),
                on_event,
            )
            review = self._invoke(
                self.reviewer_agent,
                "reviewer",
                (
                    "Review this draft answer.\n"
                    f"Question: {question}\n"
                    f"Draft: {draft}\n"
                    "Judge grounding and completeness."
                ),
            )
            self._record_trace(
                trace,
                A2AMessage(
                    sender="reviewer",
                    receiver="researcher",
                    performative="review",
                    content=f"[round={round_idx}]\n{review}",
                ),
                on_event,
            )

            if not self._needs_revision(review):
                self._record_trace(
                    trace,
                    A2AMessage(
                        sender="researcher",
                        receiver="user",
                        performative="final",
                        content=draft,
                    ),
                    on_event,
                )
                return draft, trace

            if round_idx >= bounded_rounds:
                break

            self._emit_event(
                A2AMessage(
                    sender="coordinator",
                    receiver="planner",
                    performative="dispatch",
                    content=f"[round={round_idx + 1}]\n{review}",
                ),
                on_event,
            )
            revised_plan = self._invoke(
                self.planner_agent,
                "planner",
                (
                    "Create a revised plan according to reviewer feedback.\n"
                    f"Question: {question}\n"
                    f"Current draft: {draft}\n"
                    f"Reviewer feedback: {review}"
                ),
            )
            self._record_trace(
                trace,
                A2AMessage(
                    sender="planner",
                    receiver="researcher",
                    performative="replan",
                    content=f"[round={round_idx + 1}]\n{revised_plan}",
                ),
                on_event,
            )

            self._emit_event(
                A2AMessage(
                    sender="coordinator",
                    receiver="researcher",
                    performative="dispatch",
                    content=f"[round={round_idx + 1}]\n{revised_plan}",
                ),
                on_event,
            )
            draft = self._invoke(
                self.researcher_agent,
                "researcher",
                (
                    "Execute the revised plan and return an improved final answer.\n"
                    f"Question: {question}\n"
                    f"Previous draft: {draft}\n"
                    f"Revised plan: {revised_plan}\n"
                ),
            )

        self._record_trace(
            trace,
            A2AMessage(
                sender="researcher",
                receiver="user",
                performative="final",
                content=draft,
            ),
            on_event,
        )
        return draft, trace


def create_multi_agent_a2a_session(
    *,
    llm: Any,
    search_document_fn,
    planner_system_prompt: str = PLANNER_SYSTEM_PROMPT,
    react_system_prompt: str = REACT_SYSTEM_PROMPT,
    researcher_system_prompt: str = RESEARCHER_SYSTEM_PROMPT,
    reviewer_system_prompt: str = REVIEWER_SYSTEM_PROMPT,
) -> A2AMultiAgentSession:
    tools = build_agent_tools(search_document_fn)
    session_id = f"a2a-{uuid4().hex}"

    react = create_agent(
        model=llm,
        tools=tools,
        system_prompt=react_system_prompt,
        checkpointer=InMemorySaver(),
    )
    planner = create_agent(
        model=llm,
        tools=[],
        system_prompt=planner_system_prompt,
        checkpointer=InMemorySaver(),
    )
    researcher = create_agent(
        model=llm,
        tools=tools,
        system_prompt=researcher_system_prompt,
        checkpointer=InMemorySaver(),
    )
    reviewer = create_agent(
        model=llm,
        tools=[],
        system_prompt=reviewer_system_prompt,
        checkpointer=InMemorySaver(),
    )
    coordinator = A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id=session_id,
    )
    return A2AMultiAgentSession(coordinator=coordinator, session_id=session_id)


# Backward-compatible ACP names
ACPMessage = A2AMessage
ACPMultiAgentSession = A2AMultiAgentSession
ACPMultiAgentCoordinator = A2AMultiAgentCoordinator
create_multi_agent_acp_session = create_multi_agent_a2a_session
