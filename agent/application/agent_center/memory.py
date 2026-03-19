from uuid import uuid4

from agent.memory.store import save_project_memory_episode
from utils.task_queue import TaskStatus, create_task, enqueue_task, update_task_status
from utils.tasks import task_memory_writer


def _enqueue_memory_writer(*, episode_uid: str, user_uuid: str, db_name: str) -> None:
    normalized_episode_uid = str(episode_uid or "").strip()
    if not normalized_episode_uid:
        return
    task_id = f"memory-{uuid4().hex}"
    create_task(task_id, normalized_episode_uid, "memory_writer", db_name=db_name)
    enqueue_result = enqueue_task(
        task_memory_writer,
        task_id,
        normalized_episode_uid,
        user_uuid,
        db_name=db_name,
    )
    if isinstance(enqueue_result, dict) and enqueue_result.get("mode") == "queued":
        update_task_status(
            task_id,
            TaskStatus.QUEUED,
            job_id=str(enqueue_result.get("job_id") or ""),
            db_name=db_name,
        )


def persist_turn_memory(
    *,
    user_uuid: str,
    project_uid: str,
    session_uid: str,
    prompt: str,
    answer: str,
) -> None:
    db_name = "./database.sqlite"
    prompt_text = str(prompt or "").strip()
    answer_text = str(answer or "").strip()
    if not prompt_text or not answer_text:
        return
    episode_uid = save_project_memory_episode(
        uuid=user_uuid,
        project_uid=project_uid,
        session_uid=session_uid,
        prompt=prompt_text,
        answer=answer_text,
        db_name=db_name,
    )
    _enqueue_memory_writer(episode_uid=episode_uid, user_uuid=user_uuid, db_name=db_name)
