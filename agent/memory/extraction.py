from typing import Any

from pydantic import BaseModel, Field


class MemoryCandidate(BaseModel):
    action: str = Field(default="ADD")
    memory_type: str
    title: str
    content: str
    canonical_text: str
    dedup_key: str
    confidence: float = 0.0
    source_episode_uid: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)


def _build_user_memory_candidate(episode: dict[str, Any]) -> MemoryCandidate:
    prompt = str(episode.get("prompt") or "").strip()
    answer = str(episode.get("answer") or "").strip()
    episode_uid = str(episode.get("episode_uid") or "").strip()
    canonical = prompt.replace("请记住", "").strip("，,。. ")
    return MemoryCandidate(
        memory_type="user_memory",
        title=prompt[:80] or "用户偏好",
        content=f"{prompt}\n{answer}".strip(),
        canonical_text=canonical or prompt,
        dedup_key="user:response_preferences",
        confidence=0.85,
        source_episode_uid=episode_uid,
        evidence=[{"episode_uid": episode_uid, "quote": prompt}],
    )


def _build_knowledge_memory_candidate(episode: dict[str, Any]) -> MemoryCandidate:
    prompt = str(episode.get("prompt") or "").strip()
    answer = str(episode.get("answer") or "").strip()
    episode_uid = str(episode.get("episode_uid") or "").strip()
    canonical = answer.strip()
    return MemoryCandidate(
        memory_type="knowledge_memory",
        title=prompt[:80] or "知识事实",
        content=answer,
        canonical_text=canonical,
        dedup_key=f"knowledge:{prompt[:80]}",
        confidence=0.7,
        source_episode_uid=episode_uid,
        evidence=[{"episode_uid": episode_uid, "quote": answer}],
    )


def extract_memory_candidates(
    *,
    episode: dict[str, Any],
    recent_episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del recent_episodes
    prompt = str(episode.get("prompt") or "").lower()
    answer = str(episode.get("answer") or "").strip()
    if not prompt or not answer:
        return []

    user_markers = ["请记住", "以后", "默认", "风格", "格式", "回答"]
    knowledge_markers = ["是什么", "结论", "数据集", "方法", "定义", "概念"]

    candidates: list[MemoryCandidate] = []
    if any(marker in prompt for marker in user_markers):
        candidates.append(_build_user_memory_candidate(episode))
    elif any(marker in prompt for marker in knowledge_markers):
        candidates.append(_build_knowledge_memory_candidate(episode))

    return [candidate.model_dump() for candidate in candidates]
