import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from a2a.types import Message, Part, Role, TextPart

from ..tools.builder import build_agent_tools
from ..contracts import normalize_plan_text, normalize_review_text
from ..runtime_agent import create_runtime_agent
from ..stream import extract_result_text
from .replan_policy import (
    has_replan_budget,
    normalize_max_review_rounds,
    review_needs_revision,
)
from .state_machine import (
    STATE_COMPLETED,
    STATE_FINALIZING,
    STATE_PLANNING,
    STATE_REPLANNING,
    STATE_RESEARCHING,
    STATE_REVIEWING,
    STATE_SUBMITTED,
    transition_state,
)

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = (
    "You are the Planner agent in an Agent Coordination Protocol (ACP) workflow. "
    "Turn a user question into a compact plan with 2-4 steps and return strict JSON: "
    '{"steps":["step 1","step 2"],"goal":"short goal statement"}. '
    "If JSON fails, return concise bullet points. "
    "Focus on evidence-first academic QA. "
    "Do not output chain-of-thought, only concise plan bullets."
)

RESEARCHER_SYSTEM_PROMPT = (
    "You are the Researcher agent in an ACP workflow. "
    "Use tools to gather evidence and answer the question. "
    "Prefer search_document first (JSON evidence), then search_papers for academic discovery, "
    "and use search_web only when evidence is still insufficient. "
    "For summary, critical reading, method comparison, translation, or mind map tasks, you may call use_skill for guidance. "
    "If the user asks for a mind map, output strict JSON only with this shape: "
    '{"name":"topic","children":[{"name":"subtopic","children":[...]}]}. '
    "Output concise, structured text and mark evidence source. "
    "Use the same language as the user's latest query. "
    "If the user request is ambiguous, ask exactly one short clarification question. "
    "Fixed tools can be called directly: search_document/use_skill/ask_human. "
    "For lazy tools, if not exposed, call activate_tool(tool_name) first, then call that tool directly. "
    "Lazy tools include search_web/search_papers/read_file/write_file/edit_file/update_file/bash/write_todo/edit_todo. "
    "Use write_todo/edit_todo to track plan steps, and ask_human before risky actions. "
    "Never expose internal reasoning, planning text, or tool traces."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are the Reviewer agent in an ACP workflow. "
    "Check whether draft answers are grounded and complete. "
    "Prefer strict JSON: "
    '{"decision":"PASS|REVISE","feedback":"short actionable feedback","checklist":["grounded","complete"]}. '
    "If JSON fails, output exactly two lines:\n"
    "Decision: PASS or REVISE\n"
    "Feedback: <short actionable feedback>."
)

REACT_SYSTEM_PROMPT = (
    "You are a ReAct-style academic QA agent. "
    "Use tools to gather evidence, prioritize search_document (JSON evidence), then search_papers, "
    "and answer concisely with sources. "
    "For summary, critical reading, method comparison, translation, or mind map tasks, you may call use_skill for guidance. "
    "If the user asks for a mind map, output strict JSON only with this shape: "
    '{"name":"topic","children":[{"name":"subtopic","children":[...]}]}. '
    "Use the same language as the user's latest query. "
    "If request scope is unclear, ask one concise clarification question. "
    "Fixed tools can be called directly: search_document/use_skill/ask_human. "
    "For lazy tools, if not exposed, call activate_tool(tool_name) first, then call that tool directly. "
    "Lazy tools include search_web/search_papers/read_file/write_file/edit_file/update_file/bash/write_todo/edit_todo. "
    "Use write_todo/edit_todo to track plan steps, and ask_human before risky actions. "
    "Never expose internal reasoning, planning text, or tool traces."
)

WORKFLOW_REACT = "react"
WORKFLOW_PLAN_ACT = "plan_act"
WORKFLOW_PLAN_ACT_REPLAN = "plan_act_replan"
REACT_ALLOWED_TOOLS = {
    "search_document",
    "search_papers",
    "search_web",
    "use_skill",
    "read_file",
    "write_file",
    "edit_file",
    "update_file",
    "bash",
    "write_todo",
    "edit_todo",
    "ask_human",
}
RESEARCHER_ALLOWED_TOOLS = {
    "search_document",
    "search_papers",
    "search_web",
    "use_skill",
    "read_file",
    "write_file",
    "edit_file",
    "update_file",
    "bash",
    "write_todo",
    "edit_todo",
    "ask_human",
}


@dataclass(frozen=True)
class A2AMessage:
    sender: str
    receiver: str
    performative: str
    content: str
    sdk_message: dict[str, Any] | None = None


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

    @staticmethod
    def _preview(text: str, limit: int = 120) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) <= limit:
            return collapsed
        return f"{collapsed[:limit]}..."

    def _invoke(self, agent: Any, role: str, prompt: str) -> str:
        started = time.perf_counter()
        logger.debug(
            "A2A invoke start: role=%s prompt_len=%s prompt_preview=%s",
            role,
            len(prompt),
            self._preview(prompt),
        )
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": f"{self.session_id}:{role}"}},
            )
        except Exception:
            logger.exception("A2A invoke failed: role=%s", role)
            raise
        if isinstance(result, dict):
            text = extract_result_text(result)
        else:
            text = str(result)
        logger.debug(
            "A2A invoke done: role=%s latency_ms=%.2f output_len=%s",
            role,
            (time.perf_counter() - started) * 1000.0,
            len(text),
        )
        return text

    @staticmethod
    def _normalize_plan_text(raw_plan: str) -> str:
        return normalize_plan_text(raw_plan)

    @staticmethod
    def _normalize_review_text(raw_review: str) -> str:
        return normalize_review_text(raw_review)

    @staticmethod
    def _needs_revision(review_text: str) -> bool:
        return review_needs_revision(review_text)

    @staticmethod
    def _transition_state(current_state: str, next_state: str) -> str:
        return transition_state(current_state, next_state)

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
        run_started = time.perf_counter()
        state = STATE_SUBMITTED
        trace_task_id = f"{self.session_id}:run:{uuid4().hex}"
        trace_sequence = 0

        def _msg(
            *,
            sender: str,
            receiver: str,
            performative: str,
            content: str,
            metadata: dict[str, Any] | None = None,
        ) -> A2AMessage:
            nonlocal trace_sequence
            trace_sequence += 1
            sdk_metadata: dict[str, Any] = {
                "sender": sender,
                "receiver": receiver,
                "performative": performative,
                "sequence": trace_sequence,
                "sessionId": self.session_id,
            }
            if isinstance(metadata, dict) and metadata:
                sdk_metadata["traceMeta"] = metadata
            sdk_message = Message(
                role=Role.user if sender == "user" else Role.agent,
                parts=[Part(root=TextPart(text=str(content)))],
                message_id=f"{trace_task_id}:{trace_sequence}",
                task_id=trace_task_id,
                context_id=self.session_id,
                metadata=sdk_metadata,
            ).model_dump(mode="json", by_alias=True, exclude_none=True)
            return A2AMessage(
                sender=sender,
                receiver=receiver,
                performative=performative,
                content=content,
                sdk_message=sdk_message,
            )

        logger.info(
            "A2A run start: mode=%s question_len=%s max_replan_rounds=%s",
            workflow_mode,
            len(question),
            max_replan_rounds,
        )
        if workflow_mode == WORKFLOW_REACT:
            state = self._transition_state(state, STATE_RESEARCHING)
            self._emit_event(
                _msg(
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
                _msg(
                    sender="user",
                    receiver="react_agent",
                    performative="request",
                    content=question,
                ),
                on_event,
            )
            self._record_trace(
                trace,
                _msg(
                    sender="react_agent",
                    receiver="user",
                    performative="final",
                    content=answer,
                ),
                on_event,
            )
            state = self._transition_state(state, STATE_FINALIZING)
            state = self._transition_state(state, STATE_COMPLETED)
            logger.info(
                "A2A run finish: mode=%s trace_events=%s state=%s latency_ms=%.2f",
                workflow_mode,
                len(trace),
                state,
                (time.perf_counter() - run_started) * 1000.0,
            )
            return answer, trace

        trace = []
        self._record_trace(
            trace,
            _msg(
                sender="user",
                receiver="planner",
                performative="request",
                content=question,
            ),
            on_event,
        )

        state = self._transition_state(state, STATE_PLANNING)
        self._emit_event(
            _msg(
                sender="coordinator",
                receiver="planner",
                performative="dispatch",
                content=question,
            ),
            on_event,
        )
        raw_plan = self._invoke(
            self.planner_agent,
            "planner",
            (
                "Generate a compact execution plan for the question below.\n"
                f"Question: {question}"
            ),
        )
        plan = self._normalize_plan_text(raw_plan)
        if not plan.strip():
            plan = "1. Retrieve evidence from document.\n2. Answer with concise cited points."
            logger.debug("Planner returned invalid plan format, fallback plan injected")
        self._record_trace(
            trace,
            _msg(
                sender="planner",
                receiver="researcher",
                performative="plan",
                content=plan,
            ),
            on_event,
        )

        state = self._transition_state(state, STATE_RESEARCHING)
        self._emit_event(
            _msg(
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
                _msg(
                    sender="researcher",
                    receiver="user",
                    performative="final",
                    content=draft,
                ),
                on_event,
            )
            state = self._transition_state(state, STATE_FINALIZING)
            state = self._transition_state(state, STATE_COMPLETED)
            logger.info(
                "A2A run finish: mode=%s trace_events=%s state=%s latency_ms=%.2f",
                workflow_mode,
                len(trace),
                state,
                (time.perf_counter() - run_started) * 1000.0,
            )
            return draft, trace

        bounded_rounds = normalize_max_review_rounds(max_replan_rounds)
        for round_idx in range(1, bounded_rounds + 1):
            logger.debug("A2A review round start: round=%s", round_idx)
            state = self._transition_state(state, STATE_REVIEWING)
            self._record_trace(
                trace,
                _msg(
                    sender="researcher",
                    receiver="reviewer",
                    performative="draft",
                    content=f"[round={round_idx}]\n{draft}",
                    metadata={"round": round_idx},
                ),
                on_event,
            )
            self._emit_event(
                _msg(
                    sender="coordinator",
                    receiver="reviewer",
                    performative="dispatch",
                    content=f"[round={round_idx}]\n{question}",
                    metadata={"round": round_idx},
                ),
                on_event,
            )
            raw_review = self._invoke(
                self.reviewer_agent,
                "reviewer",
                (
                    "Review this draft answer.\n"
                    f"Question: {question}\n"
                    f"Draft: {draft}\n"
                    "Judge grounding and completeness."
                ),
            )
            review = self._normalize_review_text(raw_review)
            self._record_trace(
                trace,
                _msg(
                    sender="reviewer",
                    receiver="researcher",
                    performative="review",
                    content=f"[round={round_idx}]\n{review}",
                    metadata={"round": round_idx},
                ),
                on_event,
            )

            if not self._needs_revision(review):
                logger.debug("Reviewer decision PASS at round=%s", round_idx)
                self._record_trace(
                    trace,
                    _msg(
                        sender="researcher",
                        receiver="user",
                        performative="final",
                        content=draft,
                    ),
                    on_event,
                )
                state = self._transition_state(state, STATE_FINALIZING)
                state = self._transition_state(state, STATE_COMPLETED)
                logger.info(
                    "A2A run finish: mode=%s trace_events=%s rounds=%s state=%s latency_ms=%.2f",
                    workflow_mode,
                    len(trace),
                    round_idx,
                    state,
                    (time.perf_counter() - run_started) * 1000.0,
                )
                return draft, trace

            if not has_replan_budget(round_idx, bounded_rounds):
                logger.debug("Reviewer requested revise but reached max rounds=%s", bounded_rounds)
                break

            state = self._transition_state(state, STATE_REPLANNING)
            self._emit_event(
                _msg(
                    sender="coordinator",
                    receiver="planner",
                    performative="dispatch",
                    content=f"[round={round_idx + 1}]\n{review}",
                    metadata={"round": round_idx + 1},
                ),
                on_event,
            )
            raw_revised_plan = self._invoke(
                self.planner_agent,
                "planner",
                (
                    "Create a revised plan according to reviewer feedback.\n"
                    f"Question: {question}\n"
                    f"Current draft: {draft}\n"
                    f"Reviewer feedback: {review}"
                ),
            )
            revised_plan = self._normalize_plan_text(raw_revised_plan)
            if not revised_plan.strip():
                revised_plan = (
                    "1. Address reviewer feedback with additional evidence.\n"
                    "2. Return a corrected and concise final answer."
                )
                logger.debug("Replan output invalid at round=%s, fallback revised plan used", round_idx + 1)
            self._record_trace(
                trace,
                _msg(
                    sender="planner",
                    receiver="researcher",
                    performative="replan",
                    content=f"[round={round_idx + 1}]\n{revised_plan}",
                    metadata={"round": round_idx + 1},
                ),
                on_event,
            )

            state = self._transition_state(state, STATE_RESEARCHING)
            self._emit_event(
                _msg(
                    sender="coordinator",
                    receiver="researcher",
                    performative="dispatch",
                    content=f"[round={round_idx + 1}]\n{revised_plan}",
                    metadata={"round": round_idx + 1},
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

        state = self._transition_state(state, STATE_FINALIZING)
        self._record_trace(
            trace,
            _msg(
                sender="researcher",
                receiver="user",
                performative="final",
                content=draft,
            ),
            on_event,
        )
        state = self._transition_state(state, STATE_COMPLETED)
        logger.info(
            "A2A run finish: mode=%s trace_events=%s rounds=%s state=%s latency_ms=%.2f",
            workflow_mode,
            len(trace),
            bounded_rounds,
            state,
            (time.perf_counter() - run_started) * 1000.0,
        )
        return draft, trace


def create_multi_agent_a2a_session(
    *,
    llm: Any,
    search_document_fn,
    search_document_evidence_fn=None,
    planner_system_prompt: str = PLANNER_SYSTEM_PROMPT,
    react_system_prompt: str = REACT_SYSTEM_PROMPT,
    researcher_system_prompt: str = RESEARCHER_SYSTEM_PROMPT,
    reviewer_system_prompt: str = REVIEWER_SYSTEM_PROMPT,
    context_hint: str = "",
) -> A2AMultiAgentSession:
    logger.info("Creating multi-agent A2A session")
    react_tools = build_agent_tools(
        search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
        allowed_tools=REACT_ALLOWED_TOOLS,
    )
    researcher_tools = build_agent_tools(
        search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
        allowed_tools=RESEARCHER_ALLOWED_TOOLS,
    )
    session_id = f"a2a-{uuid4().hex}"
    normalized_hint = context_hint.strip()
    react_prompt = (
        f"{react_system_prompt}\n\nContext:\n{normalized_hint}"
        if normalized_hint
        else react_system_prompt
    )
    researcher_prompt = (
        f"{researcher_system_prompt}\n\nContext:\n{normalized_hint}"
        if normalized_hint
        else researcher_system_prompt
    )
    planner_prompt = (
        f"{planner_system_prompt}\n\nContext:\n{normalized_hint}"
        if normalized_hint
        else planner_system_prompt
    )
    reviewer_prompt = (
        f"{reviewer_system_prompt}\n\nContext:\n{normalized_hint}"
        if normalized_hint
        else reviewer_system_prompt
    )

    react = create_runtime_agent(
        model=llm,
        tools=react_tools,
        system_prompt=react_prompt,
    )
    planner = create_runtime_agent(
        model=llm,
        tools=[],
        system_prompt=planner_prompt,
    )
    researcher = create_runtime_agent(
        model=llm,
        tools=researcher_tools,
        system_prompt=researcher_prompt,
    )
    reviewer = create_runtime_agent(
        model=llm,
        tools=[],
        system_prompt=reviewer_prompt,
    )
    coordinator = A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id=session_id,
    )
    logger.info("Created multi-agent A2A session: session_id=%s", session_id)
    return A2AMultiAgentSession(coordinator=coordinator, session_id=session_id)


# Backward-compatible ACP names
ACPMessage = A2AMessage
ACPMultiAgentSession = A2AMultiAgentSession
ACPMultiAgentCoordinator = A2AMultiAgentCoordinator
create_multi_agent_acp_session = create_multi_agent_a2a_session
