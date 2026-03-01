"""
页面辅助函数 - 用于统一处理任务队列、API key检查等
"""

import uuid
import logging
import streamlit as st
from typing import Any, Optional, Tuple
from .utils import ensure_local_user, get_api_key, get_model_name
from .task_queue import (
    create_task,
    get_task_status_by_uid,
    get_job_status,
    has_active_rq_workers,
    enqueue_task,
    TaskStatus,
)
from .tasks import task_text_extraction, task_file_summary, task_generate_mindmap
from .schemas import EnqueueResult


logger = logging.getLogger("llm_app.page_helpers")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def check_api_key_configured() -> Tuple[bool, Optional[str]]:
    """
    检查API key是否已配置

    Returns:
        (is_configured, error_message)
    """
    if "uuid" not in st.session_state or not st.session_state["uuid"]:
        st.session_state["uuid"] = "local-user"

    ensure_local_user(st.session_state["uuid"])

    api_key = get_api_key(st.session_state["uuid"])
    if not api_key:
        return False, "请先在侧边栏设置中配置您的 API Key"

    model_name = get_model_name(st.session_state["uuid"])
    if not model_name:
        return False, "请先在侧边栏设置中配置模型名称"

    return True, None


def start_async_task(uid: str, content_type: str, task_func, *args) -> Optional[str]:
    """
    启动异步任务

    Args:
        uid: 文件UID
        content_type: 内容类型 ('file_extraction', 'file_summary', 'file_mindmap')
        task_func: 任务函数
        *args: 传递给任务函数的参数

    Returns:
        任务ID，如果失败返回None
    """
    try:
        # 检查API key
        is_configured, error_msg = check_api_key_configured()
        if not is_configured:
            st.warning(f"⚠️ {error_msg}")
            return None

        # 生成任务ID
        task_id = str(uuid.uuid4())
        logger.info(
            "Submitting task: task_id=%s uid=%s type=%s", task_id, uid, content_type
        )

        # 创建任务记录
        create_task(task_id, uid, content_type)

        # 获取用户UUID
        user_uuid = st.session_state["uuid"]

        # 将任务加入队列
        enqueue_result = enqueue_task(task_func, task_id, *args, user_uuid)

        if isinstance(enqueue_result, dict):
            parsed = EnqueueResult.model_validate(enqueue_result)
            mode = parsed.mode
            job_id = parsed.job_id
            if mode == "queued" and job_id:
                from .task_queue import update_task_status

                update_task_status(task_id, TaskStatus.QUEUED, job_id=job_id)
                logger.info("Task queued: task_id=%s job_id=%s", task_id, job_id)
                return task_id
            if mode == "sync":
                logger.info("Task executed synchronously: task_id=%s", task_id)
                return task_id
            logger.warning("Unexpected enqueue mode: task_id=%s mode=%s", task_id, mode)
            return None

        # backward compatibility for legacy return style
        if enqueue_result:
            # 更新任务状态为已入队
            from .task_queue import update_task_status

            update_task_status(task_id, TaskStatus.QUEUED, job_id=enqueue_result)
            logger.info("Task queued: task_id=%s job_id=%s", task_id, enqueue_result)
            return task_id

        logger.warning("Task enqueue returned empty job_id: task_id=%s", task_id)
        return None
    except Exception as e:
        logger.exception("Failed to submit task: uid=%s type=%s", uid, content_type)
        st.error(f"启动任务失败: {str(e)}")
        return None


def check_task_and_content(
    uid: str, content_type: str, auto_start: bool = False
) -> Tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    """
    检查任务状态和内容

    Args:
        uid: 文件UID
        content_type: 内容类型
        auto_start: 如果没有内容且没有任务，是否自动启动

    Returns:
        (content_dict, task_status, task_id)
        content_dict: 如果内容存在则返回内容字典，否则None
        task_status: 任务状态 ('pending', 'started', 'finished', 'failed', 'queued', None)
        task_id: 任务ID
    """
    from .utils import get_content_by_uid
    import json

    # 先检查是否已有内容
    content = get_content_by_uid(uid, content_type)
    if content:
        try:
            if content_type == "file_summary":
                return {"summary": content}, None, None
            elif content_type == "file_mindmap":
                # 思维导图数据是JSON格式
                return json.loads(content), None, None
            else:
                # file_extraction 也是JSON格式
                return json.loads(content), None, None
        except:
            return {"raw": content}, None, None

    # 检查是否有进行中的任务
    task_info = get_task_status_by_uid(uid, content_type)
    if task_info:
        task_status = task_info["status"]
        task_id = task_info["task_id"]

        # 如果任务已完成，再次检查内容（可能任务刚完成但还没刷新）
        if task_status == TaskStatus.FINISHED.value:
            content = get_content_by_uid(uid, content_type)
            if content:
                try:
                    if content_type == "file_summary":
                        return {"summary": content}, None, None
                    else:
                        return json.loads(content), None, None
                except:
                    return {"raw": content}, None, None

        # 检查RQ任务状态
        if task_info.get("job_id"):
            rq_status = get_job_status(task_info["job_id"])
            if rq_status:
                # 同步状态
                if rq_status == "finished":
                    task_status = TaskStatus.FINISHED.value
                elif rq_status == "failed":
                    task_status = TaskStatus.FAILED.value
                elif rq_status == "started":
                    task_status = TaskStatus.STARTED.value
            elif task_status == TaskStatus.QUEUED.value and not has_active_rq_workers():
                from .task_queue import update_task_status

                update_task_status(
                    task_id,
                    TaskStatus.FAILED,
                    error_message="队列无可用 worker，已自动标记失败并可重试",
                )
                return None, None, None

        return None, task_status, task_id

    # 如果没有内容也没有任务，且允许自动启动
    if auto_start:
        return None, None, None

    return None, None, None


def display_task_status(
    task_status: str, error_message: Optional[str] = None, auto_refresh: bool = True
):
    """
    显示任务状态

    Args:
        task_status: 任务状态
        error_message: 错误信息（如果有）
        auto_refresh: 是否自动刷新页面
    """
    status_messages = {
        TaskStatus.PENDING.value: ("⏳", "任务等待中..."),
        TaskStatus.QUEUED.value: ("📋", "任务已加入队列，等待处理..."),
        TaskStatus.STARTED.value: ("🔄", "正在处理中，请稍候..."),
        TaskStatus.FINISHED.value: ("✅", "处理完成"),
        TaskStatus.FAILED.value: ("❌", f"处理失败: {error_message or '未知错误'}"),
    }

    icon, message = status_messages.get(task_status, ("❓", "未知状态"))

    if task_status == TaskStatus.FAILED.value:
        st.error(f"{icon} {message}")
    elif task_status in [
        TaskStatus.PENDING.value,
        TaskStatus.QUEUED.value,
        TaskStatus.STARTED.value,
    ]:
        st.info(f"{icon} {message}")
        # 自动刷新页面以检查任务状态
        if auto_refresh:
            import time

            time.sleep(2)
            st.rerun()
    else:
        st.success(f"{icon} {message}")
