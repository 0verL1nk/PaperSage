import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.adapters.llm import create_chat_model
from agent.adapters.rag import create_project_evidence_retriever
from agent.application.evals import (
    AgentEvalCase,
    build_trajectory_llm_as_judge,
    load_eval_cases,
    run_agent_evals,
    select_eval_cases,
)
from agent.application.turn_engine import execute_turn_core
from agent.paper_agent import create_paper_agent_session

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "papers" / "rag_agentic_reasoning"


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line or line.startswith("export "):
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _paper_uid_from_fixture_name(name: str) -> str:
    stem = Path(name).stem
    paper_id = stem.split("-", 1)[0].strip() if "-" in stem else stem.strip()
    return f"arxiv:{paper_id}"


def _paper_title_from_fixture_name(name: str) -> str:
    stem = Path(name).stem
    if "-" not in stem:
        return stem
    return stem.split("-", 1)[1].replace("-", " ").strip()


def _load_project_documents(max_chars_per_doc: int = 30000) -> list[dict[str, str]]:
    cache_dir = FIXTURE_DIR / "_extracted"
    docs: list[dict[str, str]] = []
    for text_path in sorted(cache_dir.glob("*.txt")):
        extracted = text_path.read_text(encoding="utf-8", errors="replace").strip()
        if not extracted:
            continue
        paper_id = _paper_uid_from_fixture_name(text_path.name)
        title = _paper_title_from_fixture_name(text_path.name)
        docs.append(
            {
                "doc_uid": paper_id,
                "doc_name": title,
                "text": (
                    f"[paper_id] {paper_id}\n"
                    f"[title] {title}\n"
                    f"[source_file] {text_path.name}\n\n"
                    f"{extracted[:max_chars_per_doc]}"
                ),
            }
        )
    if not docs:
        raise ValueError("No local paper fixture texts found for live eval smoke run.")
    return docs


def _build_live_llm() -> Any:
    api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
    model_name = str(os.getenv("OPENAI_MODEL_NAME") or "").strip()
    base_url = str(os.getenv("OPENAI_BASE_URL") or "").strip()
    missing = [
        key
        for key, value in {
            "OPENAI_API_KEY": api_key,
            "OPENAI_MODEL_NAME": model_name,
            "OPENAI_BASE_URL": base_url,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing live eval config: {', '.join(missing)}")
    return create_chat_model(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=0.0,
    )


@dataclass(frozen=True)
class LivePaperSageEvalRunner:
    llm: Any
    search_document_fn: Any
    search_document_evidence_fn: Any
    project_name: str
    scope_summary: str

    def __call__(self, case: AgentEvalCase) -> dict[str, Any]:
        session = create_paper_agent_session(
            llm=self.llm,
            search_document_fn=self.search_document_fn,
            search_document_evidence_fn=self.search_document_evidence_fn,
            project_name=self.project_name,
            scope_summary=self.scope_summary,
        )
        return execute_turn_core(
            prompt=case.prompt,
            hinted_prompt=case.prompt,
            leader_agent=session.agent,
            leader_runtime_config=session.runtime_config,
            leader_llm=self.llm,
            search_document_evidence_fn=self.search_document_evidence_fn,
            leader_tool_specs=session.tool_specs,
        )


def _default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("docs/plans/baselines") / f"task-completion-live-smoke-{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small live PaperSage task-completion eval smoke test.")
    parser.add_argument(
        "--fixture",
        default="tests/evals/fixtures/agent_task_eval_set_v1.jsonl",
        help="Path to eval fixture JSONL.",
    )
    parser.add_argument(
        "--env-file",
        default="/home/ling/LLM_App_Final/.env",
        help="Path to env file with live LLM configuration.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. Defaults to docs/plans/baselines/task-completion-live-smoke-<timestamp>.json",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Optional case id to run. Repeat to run multiple cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Optional max number of cases to run after filtering. Defaults to 1.",
    )
    parser.add_argument(
        "--disable-judge",
        action="store_true",
        help="Disable LLM-as-judge and use only stable contract checks.",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))

    fixture_path = Path(args.fixture)
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_eval_cases(fixture_path)
    cases = select_eval_cases(
        cases,
        case_ids=args.case_id or None,
        limit=args.limit if args.limit > 0 else None,
    )
    if not cases:
        raise ValueError("No eval cases selected for live smoke run.")

    documents = _load_project_documents()
    retriever = create_project_evidence_retriever(
        documents=documents,
        project_uid="task-completion-live-smoke",
    )

    def _search_document(query: str) -> str:
        payload = retriever(query)
        evidences = payload.get("evidences") if isinstance(payload, dict) else []
        if not isinstance(evidences, list):
            return ""
        return "\n".join(
            str(item.get("text") or "")
            for item in evidences
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        )

    llm = _build_live_llm()
    judge = None if args.disable_judge else build_trajectory_llm_as_judge(model=llm)
    runner = LivePaperSageEvalRunner(
        llm=llm,
        search_document_fn=_search_document,
        search_document_evidence_fn=retriever,
        project_name="Task Completion Live Smoke",
        scope_summary=f"{len(documents)} local fixture papers",
    )

    report = run_agent_evals(
        cases,
        runner=runner,
        judge=judge,
        fixture_path=str(fixture_path),
    )
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"fixture: {fixture_path}")
    print(f"cases: {report['total_cases']}")
    print(f"selected_case_ids: {[item['case_id'] for item in report['cases']]}")
    print(f"completed_cases: {report['completed_cases']}")
    print(f"completion_rate: {report['completion_rate']:.3f}")
    print(f"remediation_area_counts: {report['remediation_area_counts']}")
    print(f"report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
