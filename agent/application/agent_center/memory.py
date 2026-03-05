from agent.memory.policy import classify_turn_memory_type, ttl_for_memory_type
from agent.memory.store import upsert_project_memory_item


def persist_turn_memory(
    *,
    user_uuid: str,
    project_uid: str,
    session_uid: str,
    prompt: str,
    answer: str,
) -> None:
    prompt_text = str(prompt or "").strip()
    answer_text = str(answer or "").strip()
    if not prompt_text or not answer_text:
        return
    max_len = 1200
    memory_content = f"Q: {prompt_text}\nA: {answer_text}"
    if len(memory_content) > max_len:
        memory_content = memory_content[:max_len] + "..."
    memory_type = classify_turn_memory_type(prompt_text, answer_text)
    expires_at = ttl_for_memory_type(memory_type)
    upsert_project_memory_item(
        uuid=user_uuid,
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type=memory_type,
        title=prompt_text[:80],
        content=memory_content,
        source_prompt=prompt_text[:500],
        source_answer=answer_text[:800],
        expires_at=expires_at,
    )
