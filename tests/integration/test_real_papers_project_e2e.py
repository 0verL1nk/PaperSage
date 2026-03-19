import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest

from agent.adapters.document import extract_document_payload
from agent.adapters.rag import create_project_evidence_retriever
from agent.application.turn_engine import execute_turn_core
from agent.llm_provider import build_openai_compatible_chat_model
from agent.paper_agent import create_paper_agent_session

REAL_PAPER_SOURCES: tuple[dict[str, str], ...] = (
    {
        "paper_id": "arxiv:1706.03762",
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
    },
    {
        "paper_id": "arxiv:2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "url": "https://arxiv.org/pdf/2005.11401.pdf",
    },
    {
        "paper_id": "arxiv:2201.11903",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "url": "https://arxiv.org/pdf/2201.11903.pdf",
    },
)

LOCAL_PAPER_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "papers" / "rag_agentic_reasoning"
)


@dataclass(frozen=True)
class LiveRealScenarioContext:
    llm: Any
    project_uid: str
    documents: list[dict[str, str]]
    search_document_fn: Any
    search_document_evidence_fn: Any


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _build_live_llm(live_config: dict[str, str]):
    return build_openai_compatible_chat_model(
        api_key=live_config["OPENAI_API_KEY"],
        model_name=live_config["OPENAI_MODEL_NAME"],
        base_url=live_config["OPENAI_BASE_URL"],
        temperature=0,
    )


def _download_pdf(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (LLM_App_Final real-papers e2e)"},
    )
    with urlopen(req, timeout=90) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"Downloaded content is not a PDF: {url}")
    target_path.write_bytes(data)


def _extract_pdf_text(pdf_path: Path) -> str:
    payload = extract_document_payload(str(pdf_path))
    if not isinstance(payload, dict) or payload.get("result") != 1:
        raise ValueError(f"Failed to parse PDF: {pdf_path}")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty PDF text: {pdf_path}")
    return text.strip()


def _paper_uid_from_fixture_name(name: str) -> str:
    stem = Path(name).stem
    paper_id = stem.split("-", 1)[0].strip() if "-" in stem else stem.strip()
    return f"arxiv:{paper_id}"


def _paper_title_from_fixture_name(name: str) -> str:
    stem = Path(name).stem
    if "-" not in stem:
        return stem
    return stem.split("-", 1)[1].replace("-", " ").strip()


def _build_project_documents_from_local_fixture() -> list[dict[str, str]]:
    pdf_paths = sorted(LOCAL_PAPER_FIXTURE_DIR.glob("*.pdf"))
    if not pdf_paths:
        return []

    cache_dir = LOCAL_PAPER_FIXTURE_DIR / "_extracted"
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs: list[dict[str, str]] = []
    for pdf_path in pdf_paths:
        cache_path = cache_dir / f"{pdf_path.stem}.txt"
        if cache_path.exists():
            extracted_text = cache_path.read_text(encoding="utf-8", errors="replace")
        else:
            extracted_text = _extract_pdf_text(pdf_path)
            cache_path.write_text(extracted_text, encoding="utf-8")
        paper_id = _paper_uid_from_fixture_name(pdf_path.name)
        title = _paper_title_from_fixture_name(pdf_path.name)
        enriched_text = (
            f"[paper_id] {paper_id}\n"
            f"[title] {title}\n"
            f"[source_file] {pdf_path.name}\n\n"
            f"{extracted_text[:70000]}"
        )
        docs.append(
            {
                "doc_uid": paper_id,
                "doc_name": title,
                "text": enriched_text,
            }
        )
    return docs


def _build_project_documents(download_dir: Path) -> list[dict[str, str]]:
    local_docs = _build_project_documents_from_local_fixture()
    if local_docs:
        return local_docs

    docs: list[dict[str, str]] = []
    for item in REAL_PAPER_SOURCES:
        paper_id = item["paper_id"]
        title = item["title"]
        url = item["url"]
        pdf_name = paper_id.replace(":", "_").replace("/", "_") + ".pdf"
        pdf_path = download_dir / pdf_name
        _download_pdf(url, pdf_path)
        extracted_text = _extract_pdf_text(pdf_path)
        # 给检索增加稳定锚点，方便验证跨论文召回与引用。
        enriched_text = (
            f"[paper_id] {paper_id}\n"
            f"[title] {title}\n"
            f"[source_url] {url}\n\n"
            f"{extracted_text[:70000]}"
        )
        docs.append(
            {
                "doc_uid": paper_id,
                "doc_name": title,
                "text": enriched_text,
            }
        )
    return docs


@pytest.fixture(scope="module")
def live_real_config() -> dict[str, str]:
    _load_env_file(Path(".env"))
    enabled = os.getenv("RUN_LIVE_REAL_PAPERS_E2E", "0").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        pytest.skip("Real papers E2E disabled. Set RUN_LIVE_REAL_PAPERS_E2E=1 in .env")

    required_keys = ["OPENAI_BASE_URL", "OPENAI_MODEL_NAME", "OPENAI_API_KEY"]
    values: dict[str, str] = {}
    missing: list[str] = []
    for key in required_keys:
        value = os.getenv(key, "").strip()
        if not value:
            missing.append(key)
        else:
            values[key] = value
    if missing:
        pytest.skip(f"Real papers E2E config incomplete, missing: {', '.join(missing)}")
    return values


@pytest.fixture(scope="module")
def live_real_scenario(
    live_real_config: dict[str, str],
    tmp_path_factory: pytest.TempPathFactory,
) -> LiveRealScenarioContext:
    download_dir = tmp_path_factory.mktemp("real-papers")
    documents = _build_project_documents(download_dir)
    project_uid = "real-papers-project-e2e"
    search_document_evidence_fn = create_project_evidence_retriever(
        documents=documents,
        project_uid=project_uid,
    )

    def _search_document(query: str) -> str:
        payload = search_document_evidence_fn(query)
        evidences = payload.get("evidences") if isinstance(payload, dict) else []
        if not isinstance(evidences, list):
            return ""
        return "\n".join(
            str(item.get("text", ""))
            for item in evidences
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        )

    return LiveRealScenarioContext(
        llm=_build_live_llm(live_real_config),
        project_uid=project_uid,
        documents=documents,
        search_document_fn=_search_document,
        search_document_evidence_fn=search_document_evidence_fn,
    )


def test_real_papers_project_retrieval_hits_multiple_documents(
    live_real_scenario: LiveRealScenarioContext,
) -> None:
    payload = live_real_scenario.search_document_evidence_fn(
        "请比较 arxiv:1706.03762 与 arxiv:2005.11401 的核心思想差异"
    )
    evidences = payload.get("evidences") if isinstance(payload, dict) else None
    assert isinstance(evidences, list)
    assert len(evidences) >= 2
    hit_doc_uids = {
        str(item.get("doc_uid") or "").strip()
        for item in evidences
        if isinstance(item, dict)
    }
    hit_doc_uids.discard("")
    assert len(hit_doc_uids) >= 2


def test_real_papers_turn_engine_live_end_to_end(
    live_real_scenario: LiveRealScenarioContext,
) -> None:
    leader_session = create_paper_agent_session(
        llm=live_real_scenario.llm,
        search_document_fn=live_real_scenario.search_document_fn,
        search_document_evidence_fn=live_real_scenario.search_document_evidence_fn,
        project_name="真实论文落地测试项目",
        scope_summary=f"{len(live_real_scenario.documents)} 篇论文",
    )

    prompt = (
        "你必须先调用 search_document 工具，再回答。\n"
        "请比较 arxiv:1706.03762 和 arxiv:2005.11401 的方法差异，并补充一条对 arxiv:2201.11903 的关系说明。"
        "输出要求：\n"
        "1) 至少引用两个不同论文的证据\n"
        "2) 最后给出 1 条可执行选型建议"
    )
    result = execute_turn_core(
        prompt=prompt,
        hinted_prompt=prompt,
        leader_agent=leader_session.agent,
        leader_runtime_config=leader_session.runtime_config,
        leader_llm=live_real_scenario.llm,
        search_document_evidence_fn=live_real_scenario.search_document_evidence_fn,
    )

    assert isinstance(result["answer"], str) and result["answer"].strip()
    evidence_items = result["evidence_items"]
    assert isinstance(evidence_items, list) and evidence_items
    hit_doc_uids = {
        str(item.get("doc_uid") or "").strip()
        for item in evidence_items
        if isinstance(item, dict)
    }
    hit_doc_uids.discard("")
    assert len(hit_doc_uids) >= 2

    performatives = [
        str(item.get("performative", ""))
        for item in result["trace_payload"]
        if isinstance(item, dict)
    ]
    assert "complexity_analysis" in performatives or "complexity_result" in performatives
