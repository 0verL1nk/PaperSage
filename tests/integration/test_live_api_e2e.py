import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from agent.archive import list_agent_outputs, save_agent_output
from agent.llm_provider import build_openai_compatible_chat_model
from agent.multi_agent_a2a import (
    WORKFLOW_PLAN_ACT,
    WORKFLOW_PLAN_ACT_REPLAN,
    WORKFLOW_REACT,
    create_multi_agent_a2a_session,
)
from agent.paper_agent import create_paper_agent_session
from agent.stream import iter_agent_response_deltas
from agent.workflow_router import auto_select_workflow_mode
from utils.utils import extract_json_string


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def _build_live_llm(
    live_config: dict[str, str],
    *,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
):
    return build_openai_compatible_chat_model(
        api_key=live_config["OPENAI_API_KEY"],
        model_name=live_config["OPENAI_MODEL_NAME"],
        base_url=live_config["OPENAI_BASE_URL"],
        temperature=0,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )


def _doc_search(_query: str) -> str:
    return (
        "文档记录：代号是 ORBIT-427。"
        "方法A在精度方面更好，方法B在速度方面更快。"
    )


def _collect_answer(session, prompt: str) -> str:
    return "".join(
        iter_agent_response_deltas(
            session.agent,
            [{"role": "user", "content": prompt}],
            config=session.runtime_config,
        )
    ).strip()


def _parse_first_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("No valid JSON object found in model output")


def _result_has_tool_call(result: dict, tool_name: str) -> bool:
    raw_messages = result.get("messages", []) if isinstance(result, dict) else []
    if not isinstance(raw_messages, list):
        return False
    for message in raw_messages:
        tool_calls = getattr(message, "tool_calls", None)
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if isinstance(call, dict):
                name = str(call.get("name") or "").strip()
            else:
                name = str(getattr(call, "name", "") or "").strip()
            if name == tool_name:
                return True
    return False


@pytest.fixture(scope="module")
def live_config() -> dict[str, str]:
    _load_env_file(Path(".env"))
    enabled = os.getenv("RUN_LIVE_E2E", "0").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        pytest.skip("Live E2E disabled. Set RUN_LIVE_E2E=1 in .env")

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
        pytest.skip(f"Live E2E config incomplete, missing: {', '.join(missing)}")
    return values


def test_live_model_roundtrip(live_config: dict[str, str]) -> None:
    llm = _build_live_llm(live_config)
    result = llm.invoke("请只回复 LIVE_OK")
    content = result.content if isinstance(result.content, str) else str(result.content)
    assert "LIVE_OK" in content.upper()


def test_live_model_roundtrip_with_thinking_toggle(
    live_config: dict[str, str],
) -> None:
    llm = _build_live_llm(
        live_config,
        enable_thinking=True,
        reasoning_effort=os.getenv("AGENT_REASONING_EFFORT", "medium"),
    )
    result = llm.invoke("请只回复 THINK_OK")
    content = result.content if isinstance(result.content, str) else str(result.content)
    assert "THINK_OK" in content.upper()


def test_live_agent_roundtrip(live_config: dict[str, str]) -> None:
    session = create_paper_agent_session(
        llm=_build_live_llm(live_config),
        search_document_fn=_doc_search,
    )
    answer = _collect_answer(session, "请告诉我文档里的代号是什么？只返回代号")
    assert answer
    assert "427" in answer


def test_live_router_roundtrip(live_config: dict[str, str]) -> None:
    a2a = create_multi_agent_a2a_session(
        llm=_build_live_llm(live_config),
        search_document_fn=_doc_search,
    )
    mode, reason = auto_select_workflow_mode(
        "请比较两种方法的优缺点并给出 trade-off 建议。",
        coordinator=a2a.coordinator,
    )
    assert mode in {WORKFLOW_REACT, WORKFLOW_PLAN_ACT, WORKFLOW_PLAN_ACT_REPLAN}
    assert reason


def test_live_a2a_react_mode_roundtrip(live_config: dict[str, str]) -> None:
    a2a = create_multi_agent_a2a_session(
        llm=_build_live_llm(live_config),
        search_document_fn=_doc_search,
    )
    answer, trace = a2a.coordinator.run(
        "文档中的代号是什么？只返回代号。",
        workflow_mode=WORKFLOW_REACT,
    )
    assert answer
    assert "427" in answer
    assert [item.performative for item in trace] == ["request", "final"]


def test_live_a2a_plan_act_mode_roundtrip(live_config: dict[str, str]) -> None:
    a2a = create_multi_agent_a2a_session(
        llm=_build_live_llm(live_config),
        search_document_fn=_doc_search,
    )
    answer, trace = a2a.coordinator.run(
        "请先规划再回答：文档中的代号是什么，并补一句说明。",
        workflow_mode=WORKFLOW_PLAN_ACT,
    )
    assert answer
    assert [item.performative for item in trace] == ["request", "plan", "final"]


def test_live_a2a_plan_act_replan_mode_roundtrip(
    live_config: dict[str, str],
) -> None:
    a2a = create_multi_agent_a2a_session(
        llm=_build_live_llm(live_config),
        search_document_fn=_doc_search,
    )
    answer, trace = a2a.coordinator.run(
        "请先规划、执行、复核后回答：对比方法A和方法B并给出代号。",
        workflow_mode=WORKFLOW_PLAN_ACT_REPLAN,
    )
    assert answer

    performatives = [item.performative for item in trace]
    assert performatives[0] == "request"
    assert "plan" in performatives
    assert "draft" in performatives
    assert "review" in performatives
    assert performatives[-1] == "final"


def test_live_mindmap_and_archive_roundtrip(
    live_config: dict[str, str], tmp_path: Path
) -> None:
    session = create_paper_agent_session(
        llm=_build_live_llm(live_config),
        search_document_fn=_doc_search,
    )
    prompt = (
        "请基于文档生成思维导图，严格只输出 JSON 对象。"
        '格式必须为 {"name":"主题","children":[{"name":"子主题","children":[...]}]}，'
        "不要 markdown，不要解释。"
    )
    answer = _collect_answer(session, prompt)
    parsed = _parse_first_json_object(extract_json_string(answer))
    assert isinstance(parsed, dict)
    assert "name" in parsed
    assert isinstance(parsed.get("children"), list)

    db_path = tmp_path / "live_e2e_archive.sqlite"
    user_uuid = f"live-{uuid4().hex}"
    doc_uid = f"doc-{uuid4().hex}"
    save_agent_output(
        uuid=user_uuid,
        doc_uid=doc_uid,
        doc_name="live-e2e-paper.pdf",
        output_type="mindmap",
        content=json.dumps(parsed, ensure_ascii=False),
        db_name=str(db_path),
    )
    records = list_agent_outputs(uuid=user_uuid, doc_uid=doc_uid, db_name=str(db_path))
    assert records
    assert records[0]["output_type"] == "mindmap"


def test_live_agent_auto_calls_mindmap_skill(live_config: dict[str, str]) -> None:
    session = create_paper_agent_session(
        llm=_build_live_llm(live_config),
        search_document_fn=_doc_search,
    )
    result = session.agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请根据当前文档生成思维导图。先自行选择并调用合适的 skill，再输出严格 JSON。"
                    ),
                }
            ]
        },
        config=session.runtime_config,
    )

    assert isinstance(result, dict)
    assert _result_has_tool_call(result, "use_skill")
