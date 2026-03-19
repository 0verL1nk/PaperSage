import json
import logging
import sqlite3
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from agent.llm_provider import build_openai_compatible_chat_model

logger = logging.getLogger(__name__)

VALID_MEMORY_ACTIONS = {"ADD", "UPDATE", "DELETE", "NONE", "SUPERSEDE"}
VALID_MEMORY_TYPES = {"user_memory", "knowledge_memory"}

MEMORY_EXTRACTION_SYSTEM_PROMPT = """
You are a long-term memory extraction agent.

Your job:
1. Read the current conversation episode, recent episodes, and existing active memories.
2. Decide whether this turn contains durable long-term memory worth saving.
3. Return only structured memory candidates in JSON.

Rules:
- Do not use keyword heuristics. Judge from meaning and conversational intent.
- Extract only durable memory. Ignore small talk, transient requests, and one-off acknowledgements.
- `user_memory` is for stable user preferences, response instructions, and long-lived constraints.
- `knowledge_memory` is for durable project or paper facts worth recalling later.
- The stored memory must be a short declarative fragment statement, not a transcript.
- Do not store memory as Q/A, dialogue turns, speaker-prefixed text, or copied conversational exchange.
- Good examples:
  - `user prefers concise answers`
  - `user codes mostly in golang`
  - `project Apollo deadline moved to Apr 30`
- Every candidate must include:
  - `action`: one of ADD, UPDATE, DELETE, NONE, SUPERSEDE
  - `memory_type`: `user_memory` or `knowledge_memory`
  - `title`
  - `content`
  - `canonical_text`
  - `dedup_key`
  - `confidence`: 0 to 1
  - `evidence`: list of short supporting quotes from the current episode
- If nothing should be saved, return `{ "candidates": [] }`.

Output requirements:
- Return strict JSON only.
- Top-level object must be `{ "candidates": [...] }`.
""".strip()


def _read_user_column(*, uuid: str, column: str, db_name: str = "./database.sqlite") -> str | None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"SELECT {column} FROM users WHERE uuid = ?", (uuid,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    value = row[0]
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def read_api_key_for_user(*, uuid: str, db_name: str = "./database.sqlite") -> str:
    return _read_user_column(uuid=uuid, column="api_key", db_name=db_name) or ""


def read_model_name_for_user(*, uuid: str, db_name: str = "./database.sqlite") -> str | None:
    return _read_user_column(uuid=uuid, column="model_name", db_name=db_name)


def read_base_url_for_user(*, uuid: str, db_name: str = "./database.sqlite") -> str | None:
    return _read_user_column(uuid=uuid, column="base_url", db_name=db_name)


def create_chat_model(
    *,
    api_key: str,
    model_name: str,
    base_url: str | None = None,
    temperature: float | None = None,
):
    return build_openai_compatible_chat_model(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
    )


class MemoryCandidate(BaseModel):
    action: str = Field(default="ADD")
    memory_type: str
    title: str
    content: str
    canonical_text: str
    dedup_key: str
    confidence: float = 0.0
    source_episode_uid: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("action")
    @classmethod
    def _validate_action(cls, value: str) -> str:
        normalized = str(value or "").strip().upper() or "ADD"
        if normalized not in VALID_MEMORY_ACTIONS:
            raise ValueError(f"unsupported action: {normalized}")
        return normalized

    @field_validator("memory_type")
    @classmethod
    def _validate_memory_type(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in VALID_MEMORY_TYPES:
            raise ValueError(f"unsupported memory_type: {normalized}")
        return normalized

    @field_validator("title", "content", "canonical_text", "dedup_key")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("required text field is empty")
        return normalized

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        normalized = float(value or 0.0)
        if normalized < 0.0:
            return 0.0
        if normalized > 1.0:
            return 1.0
        return normalized

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence_field(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str) and item.strip():
                normalized.append({"quote": item.strip()})
        return normalized


class MemoryExtractionResult(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content)


def _extract_json_object(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("memory extractor returned empty content")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("memory extractor did not return a JSON object")
    return text[start : end + 1]


def _coerce_extraction_payload(parsed: Any) -> MemoryExtractionResult:
    if isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
        return MemoryExtractionResult.model_validate(parsed)
    if isinstance(parsed, list):
        return MemoryExtractionResult.model_validate({"candidates": parsed})
    if isinstance(parsed, dict) and (
        "memory_type" in parsed or "canonical_text" in parsed or "content" in parsed
    ):
        return MemoryExtractionResult.model_validate({"candidates": [parsed]})
    raise ValueError("memory extractor payload does not contain candidates")


def _parse_extraction_result(raw_text: str) -> MemoryExtractionResult:
    text = str(raw_text or "").strip()
    if not text:
        return MemoryExtractionResult(candidates=[])
    decoder = json.JSONDecoder()
    candidate_positions = [index for index, char in enumerate(text) if char in "{["]
    for index in candidate_positions:
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        try:
            return _coerce_extraction_payload(parsed)
        except Exception:
            continue
    try:
        return MemoryExtractionResult.model_validate_json(_extract_json_object(text))
    except ValidationError as exc:
        raise ValueError(f"memory extractor returned invalid payload: {exc}") from exc


def _normalize_evidence(
    evidence_items: list[dict[str, Any]],
    *,
    episode_uid: str,
) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        if not str(normalized.get("episode_uid") or "").strip():
            normalized["episode_uid"] = episode_uid
        normalized_items.append(normalized)
    return normalized_items


def _serialize_extraction_context(
    *,
    episode: dict[str, Any],
    recent_episodes: list[dict[str, Any]],
    active_memories: list[dict[str, Any]],
) -> str:
    payload = {
        "episode": {
            "episode_uid": str(episode.get("episode_uid") or ""),
            "prompt": str(episode.get("prompt") or ""),
            "answer": str(episode.get("answer") or ""),
        },
        "recent_episodes": [
            {
                "episode_uid": str(item.get("episode_uid") or ""),
                "prompt": str(item.get("prompt") or ""),
                "answer": str(item.get("answer") or ""),
            }
            for item in recent_episodes[:5]
            if isinstance(item, dict)
        ],
        "active_memories": [
            {
                "memory_uid": str(item.get("memory_uid") or ""),
                "memory_type": str(item.get("memory_type") or ""),
                "canonical_text": str(item.get("canonical_text") or ""),
                "content": str(item.get("content") or ""),
                "dedup_key": str(item.get("dedup_key") or ""),
                "status": str(item.get("status") or ""),
            }
            for item in active_memories[:20]
            if isinstance(item, dict)
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def extract_memory_candidates(
    *,
    episode: dict[str, Any],
    recent_episodes: list[dict[str, Any]],
    active_memories: list[dict[str, Any]],
    user_uuid: str,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    prompt = str(episode.get("prompt") or "").strip()
    answer = str(episode.get("answer") or "").strip()
    episode_uid = str(episode.get("episode_uid") or "").strip()
    if not prompt or not answer or not episode_uid:
        return []

    api_key = read_api_key_for_user(uuid=user_uuid, db_name=db_name)
    if not api_key:
        raise ValueError("memory extraction requires a configured API key")
    model_name = read_model_name_for_user(uuid=user_uuid, db_name=db_name)
    if not model_name:
        raise ValueError("memory extraction requires a configured model name")
    base_url = read_base_url_for_user(uuid=user_uuid, db_name=db_name)

    llm = create_chat_model(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=0.0,
    )
    result = llm.invoke(
        [
            ("system", MEMORY_EXTRACTION_SYSTEM_PROMPT),
            (
                "user",
                _serialize_extraction_context(
                    episode=episode,
                    recent_episodes=recent_episodes,
                    active_memories=active_memories,
                ),
            ),
        ]
    )
    result_text = _content_to_text(getattr(result, "content", result))
    try:
        parsed = _parse_extraction_result(result_text)
    except Exception as exc:
        logger.error("memory extractor raw response: %s", result_text)
        raise

    candidates: list[dict[str, Any]] = []
    for candidate in parsed.candidates:
        normalized = candidate.model_dump()
        normalized["source_episode_uid"] = episode_uid
        normalized["evidence"] = _normalize_evidence(
            list(normalized.get("evidence") or []),
            episode_uid=episode_uid,
        )
        candidates.append(normalized)
    return candidates
