"""
任务队列模块 - 优先使用 RQ，Windows/不可用场景自动切换到本地线程队列
"""

import importlib
import logging
import os
import sqlite3
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from multiprocessing import get_all_start_methods
from threading import Lock
from typing import Any, Dict, Optional

from redis import Redis

from .schemas import EnqueueResult

logger = logging.getLogger("llm_app.task_queue")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

rq_module: Any | None = None
rq_job_module: Any | None = None
try:
    rq_module = importlib.import_module("rq")
    rq_job_module = importlib.import_module("rq.job")
except Exception as exc:
    rq_module = None
    rq_job_module = None
    logger.warning("RQ unavailable, falling back to local queue backend: %s", exc)


def _supports_fork_start_method() -> bool:
    try:
        return "fork" in get_all_start_methods()
    except Exception:
        return False

# Redis 连接配置（可通过环境变量配置）
# 默认使用 localhost，因为 Redis 和应用在同一容器中
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
REDIS_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "0.3"))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "0.3"))

# 创建 Redis 连接
try:
    redis_conn: Redis | None = Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
    )
    logger.info("Redis client initialized for task queue")
except Exception as e:
    logger.warning("Redis connection failed, fallback to sync execution: %s", e)
    redis_conn = None

# 创建任务队列（Windows 不支持 fork，自动回退到本地线程队列）
QUEUE_BACKEND_MODE = os.getenv("TASK_QUEUE_BACKEND", "auto").strip().lower()
if QUEUE_BACKEND_MODE not in {"auto", "rq", "local"}:
    logger.warning("Unknown TASK_QUEUE_BACKEND=%s, fallback to auto", QUEUE_BACKEND_MODE)
    QUEUE_BACKEND_MODE = "auto"
task_queue: Any | None = None
_rq_available = bool(rq_module is not None and redis_conn is not None and _supports_fork_start_method())

if QUEUE_BACKEND_MODE == "rq" and not _rq_available:
    logger.warning("TASK_QUEUE_BACKEND=rq but RQ backend is unavailable, fallback to local")

if (QUEUE_BACKEND_MODE == "rq" and _rq_available) or (
    QUEUE_BACKEND_MODE == "auto" and _rq_available
):
    try:
        task_queue = rq_module.Queue("tasks", connection=redis_conn) if rq_module is not None else None
    except Exception as e:
        logger.warning("RQ queue initialization failed, use local queue backend: %s", e)

QUEUE_BACKEND = "rq" if task_queue else "local"
LOCAL_TASK_MAX_WORKERS = max(1, int(os.getenv("LOCAL_TASK_MAX_WORKERS", "2")))
_local_executor: ThreadPoolExecutor | None = None
_local_jobs: Dict[str, Dict[str, Any]] = {}
_local_jobs_lock = Lock()

if QUEUE_BACKEND == "local":
    _local_executor = ThreadPoolExecutor(
        max_workers=LOCAL_TASK_MAX_WORKERS,
        thread_name_prefix="taskq",
    )
    logger.info("Task queue backend: local_thread (workers=%s)", LOCAL_TASK_MAX_WORKERS)
else:
    logger.info("Task queue backend: rq")


class TaskStatus(Enum):
    """任务状态枚举"""

    PENDING = "pending"  # 等待中
    STARTED = "started"  # 已开始
    FINISHED = "finished"  # 已完成
    FAILED = "failed"  # 失败
    QUEUED = "queued"  # 已入队


def init_task_table(db_name="./database.sqlite"):
    """初始化任务状态表"""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS task_status (
            task_id TEXT PRIMARY KEY,
            uid TEXT NOT NULL,
            content_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error_message TEXT,
            job_id TEXT
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_task_status_uid ON task_status(uid, content_type)
    """)
    conn.commit()
    conn.close()


def create_task(task_id: str, uid: str, content_type: str, db_name="./database.sqlite"):
    """创建任务记录"""
    import datetime

    conn = sqlite3.connect(db_name, timeout=10)
    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout = 5000")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT OR REPLACE INTO task_status
        (task_id, uid, content_type, status, created_at, updated_at, job_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            task_id,
            uid,
            content_type,
            TaskStatus.PENDING.value,
            current_time,
            current_time,
            None,
        ),
    )
    conn.commit()
    conn.close()
    logger.info("Task created: task_id=%s uid=%s type=%s", task_id, uid, content_type)


def update_task_status(
    task_id: str,
    status: TaskStatus,
    job_id: Optional[str] = None,
    error_message: Optional[str] = None,
    db_name="./database.sqlite",
):
    """更新任务状态"""
    import datetime

    conn = sqlite3.connect(db_name, timeout=10)
    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout = 5000")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if error_message:
        cursor.execute(
            """
            UPDATE task_status
            SET status = ?, updated_at = ?, error_message = ?, job_id = ?
            WHERE task_id = ?
        """,
            (status.value, current_time, error_message, job_id, task_id),
        )
    else:
        cursor.execute(
            """
            UPDATE task_status
            SET status = ?, updated_at = ?, job_id = ?
            WHERE task_id = ?
        """,
            (status.value, current_time, job_id, task_id),
        )
    conn.commit()
    conn.close()
    logger.info(
        "Task status updated: task_id=%s status=%s job_id=%s error=%s",
        task_id,
        status.value,
        job_id,
        error_message,
    )


def get_task_status(
    task_id: str, db_name="./database.sqlite"
) -> Optional[Dict[str, Any]]:
    """获取任务状态"""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT task_id, uid, content_type, status, created_at, updated_at, error_message, job_id
        FROM task_status
        WHERE task_id = ?
    """,
        (task_id,),
    )
    result = cursor.fetchone()
    conn.close()

    if not result:
        return None

    return {
        "task_id": result[0],
        "uid": result[1],
        "content_type": result[2],
        "status": result[3],
        "created_at": result[4],
        "updated_at": result[5],
        "error_message": result[6],
        "job_id": result[7],
    }


def get_task_status_by_uid(
    uid: str, content_type: str, db_name="./database.sqlite"
) -> Optional[Dict[str, Any]]:
    """根据 uid 和 content_type 获取任务状态"""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT task_id, uid, content_type, status, created_at, updated_at, error_message, job_id
        FROM task_status
        WHERE uid = ? AND content_type = ?
        ORDER BY created_at DESC
        LIMIT 1
    """,
        (uid, content_type),
    )
    result = cursor.fetchone()
    conn.close()

    if not result:
        return None

    return {
        "task_id": result[0],
        "uid": result[1],
        "content_type": result[2],
        "status": result[3],
        "created_at": result[4],
        "updated_at": result[5],
        "error_message": result[6],
        "job_id": result[7],
    }


def get_job_status(job_id: str) -> Optional[str]:
    """获取任务状态（RQ 或本地线程队列）"""
    if not job_id:
        return None

    if QUEUE_BACKEND == "local":
        return _get_local_job_status(job_id)

    if redis_conn is None or rq_job_module is None:
        return None

    try:
        job = rq_job_module.Job.fetch(job_id, connection=redis_conn)
        return job.get_status()
    except Exception:
        return None


def _has_active_rq_workers() -> bool:
    if not redis_conn:
        return False
    try:
        worker_count_raw = redis_conn.scard("rq:workers")
        if not isinstance(worker_count_raw, int):
            return False
        worker_count = worker_count_raw
        return worker_count > 0
    except Exception:
        return False


def has_active_rq_workers() -> bool:
    if QUEUE_BACKEND == "local":
        return True
    return _has_active_rq_workers()


def _set_local_job_status(job_id: str, status: str) -> None:
    with _local_jobs_lock:
        state = _local_jobs.setdefault(job_id, {})
        state["status"] = status


def _get_local_job_status(job_id: str) -> Optional[str]:
    with _local_jobs_lock:
        state = _local_jobs.get(job_id)
    if not state:
        return None

    status = state.get("status")
    future = state.get("future")
    if isinstance(future, Future):
        if future.cancelled():
            status = "failed"
        elif future.done() and status not in {"finished", "failed"}:
            status = "failed" if future.exception() else "finished"
        if isinstance(status, str):
            _set_local_job_status(job_id, status)
    return status if isinstance(status, str) else None


def _run_local_job(job_id: str, task_func, *args, **kwargs) -> None:
    _set_local_job_status(job_id, "started")
    try:
        task_func(*args, **kwargs)
    except Exception:
        _set_local_job_status(job_id, "failed")
        logger.exception("Local queued task failed: job_id=%s", job_id)
        raise
    else:
        _set_local_job_status(job_id, "finished")


def _enqueue_local_task(task_func, *args, **kwargs) -> Dict[str, Any]:
    if not _local_executor:
        logger.warning("Local queue unavailable, running task synchronously")
        task_func(*args, **kwargs)
        return EnqueueResult(mode="sync", job_id=None).model_dump()

    job_id = f"local-{uuid.uuid4()}"
    _set_local_job_status(job_id, "queued")
    future = _local_executor.submit(_run_local_job, job_id, task_func, *args, **kwargs)
    with _local_jobs_lock:
        _local_jobs[job_id]["future"] = future
    logger.info("Task enqueued locally: job_id=%s", job_id)
    return EnqueueResult(mode="queued", job_id=job_id).model_dump()


def enqueue_task(task_func, *args, **kwargs) -> Dict[str, Any]:
    """将任务加入队列"""
    if QUEUE_BACKEND == "local":
        try:
            return _enqueue_local_task(task_func, *args, **kwargs)
        except Exception:
            logger.exception("Local queue failed, fallback to sync execution")
            task_func(*args, **kwargs)
            return EnqueueResult(mode="sync", job_id=None).model_dump()

    if not task_queue:
        # 如果没有 Redis，直接同步执行
        try:
            logger.info("Queue unavailable, running task synchronously")
            task_func(*args, **kwargs)
            return EnqueueResult(mode="sync", job_id=None).model_dump()
        except Exception as e:
            logger.exception("Synchronous task execution failed")
            raise e

    if not _has_active_rq_workers():
        try:
            logger.warning("No active RQ workers, fallback to synchronous execution")
            task_func(*args, **kwargs)
            return EnqueueResult(mode="sync", job_id=None).model_dump()
        except Exception as e:
            logger.exception("Synchronous fallback failed without active workers")
            raise e

    try:
        job = task_queue.enqueue(task_func, *args, **kwargs, job_timeout="10m")
        logger.info("Task enqueued: job_id=%s", job.id)
        return EnqueueResult(mode="queued", job_id=job.id).model_dump()
    except Exception:
        logger.exception("Task enqueue failed, fallback to sync execution")
        # 如果入队失败，回退到同步执行
        try:
            task_func(*args, **kwargs)
            return EnqueueResult(mode="sync", job_id=None).model_dump()
        except Exception as e:
            logger.exception("Fallback synchronous task execution failed")
            raise e
