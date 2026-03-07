import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..domain.orchestration import PolicyDecision
from ..domain.request_context import RequestContext
from .policy_engine import intercept

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncPolicySnapshot:
    decision: PolicyDecision
    version: int
    updated_at_monotonic: float


class AsyncPolicyPredictor:
    """Periodically refreshes policy decisions in background."""

    def __init__(
        self,
        *,
        prompt: str,
        llm: Any,
        context_digest: str = "",
        force_plan: bool | None = None,
        force_team: bool | None = None,
        refresh_interval_sec: float = 4.0,
    ) -> None:
        self._prompt = str(prompt or "")
        self._llm = llm
        self._force_plan = force_plan
        self._force_team = force_team
        self._refresh_interval = max(0.5, float(refresh_interval_sec))
        self._context_digest = str(context_digest or "")
        self._snapshot: AsyncPolicySnapshot | None = None
        self._version = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._llm is None or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="async-policy-predictor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Async policy predictor started: interval=%.2fs",
            self._refresh_interval,
        )

    def stop(self, *, join_timeout_sec: float = 0.25) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(join_timeout_sec)))
        self._thread = None
        logger.info("Async policy predictor stopped")

    def update_context_digest(self, context_digest: str) -> None:
        with self._lock:
            self._context_digest = str(context_digest or "")

    def latest_snapshot(
        self,
        *,
        max_staleness_seconds: float = 20.0,
    ) -> AsyncPolicySnapshot | None:
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return None
        staleness = time.monotonic() - snapshot.updated_at_monotonic
        if staleness > max(1.0, float(max_staleness_seconds)):
            return None
        return snapshot

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    context_digest = self._context_digest
                decision = intercept(
                    RequestContext(
                        prompt=self._prompt,
                        context_digest=context_digest,
                    ),
                    llm=self._llm,
                    force_plan=self._force_plan,
                    force_team=self._force_team,
                    emit_info_log=False,
                )
                now = time.monotonic()
                with self._lock:
                    self._version += 1
                    version = self._version
                    self._snapshot = AsyncPolicySnapshot(
                        decision=decision,
                        version=version,
                        updated_at_monotonic=now,
                    )
                reason = str(decision.reason or "").strip()
                if len(reason) > 120:
                    reason = f"{reason[:117]}..."
                logger.info(
                    "Async policy refresh: version=%s plan=%s team=%s source=%s confidence=%s reason=%s",
                    version,
                    decision.plan_enabled,
                    decision.team_enabled,
                    decision.source,
                    (
                        f"{float(decision.confidence):.2f}"
                        if decision.confidence is not None
                        else "-"
                    ),
                    reason or "-",
                )
            except Exception:
                logger.exception("Async policy predictor refresh failed")
            self._stop_event.wait(self._refresh_interval)
