import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

LOG_CONTEXT_FIELDS = ("run_id", "uid", "project_uid", "doc_uid", "workflow", "session_id")
APP_LOG_PREFIXES = ("llm_app", "agent", "main", "utils")
_LOG_CONTEXT: ContextVar[dict[str, str]] = ContextVar("agent_log_context", default={})
_HANDLER_MARKER = "_llm_app_logging_handler"


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = _LOG_CONTEXT.get({})
        for key in LOG_CONTEXT_FIELDS:
            setattr(record, key, context.get(key, "-"))
        return True


class _SafeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        for key in LOG_CONTEXT_FIELDS:
            if not hasattr(record, key):
                setattr(record, key, "-")
        return super().format(record)


class _NamespaceFilter(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__()
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefixes)


def _normalize_level(raw_level: str | int | None) -> int:
    if isinstance(raw_level, int):
        return raw_level
    if isinstance(raw_level, str) and raw_level.strip():
        maybe_level = logging.getLevelName(raw_level.strip().upper())
        if isinstance(maybe_level, int):
            return maybe_level
    return logging.INFO


def _context_value(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "-"
    return text.replace("\n", " ")


@contextmanager
def logging_context(**fields: Any) -> Iterator[None]:
    current = dict(_LOG_CONTEXT.get({}))
    for key, value in fields.items():
        if key not in LOG_CONTEXT_FIELDS or value is None:
            continue
        current[key] = _context_value(value)
    token = _LOG_CONTEXT.set(current)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def configure_application_logging(
    *,
    debug_mode: bool = False,
    default_level: str = "INFO",
    logger_name: str = "llm_app",
) -> None:
    level_from_env = os.getenv("APP_LOG_LEVEL")
    final_level = _normalize_level(level_from_env or default_level)
    if debug_mode:
        final_level = logging.DEBUG

    root = logging.getLogger()
    root.setLevel(final_level)

    def _is_app_handler(handler: logging.Handler) -> bool:
        if getattr(handler, _HANDLER_MARKER, False):
            return True
        formatter = getattr(handler, "formatter", None)
        fmt = getattr(formatter, "_fmt", "")
        has_context_fmt = isinstance(fmt, str) and "run=%(run_id)s" in fmt and "project=%(project_uid)s" in fmt
        has_namespace_filter = any(
            isinstance(filter_obj, _NamespaceFilter)
            for filter_obj in getattr(handler, "filters", [])
        )
        return has_context_fmt and has_namespace_filter

    # Streamlit rerun / hot-reload may leave handlers from previous runs.
    # Remove all app-owned handlers, including legacy unmarked ones.
    existing_app_handlers = [handler for handler in list(root.handlers) if _is_app_handler(handler)]
    for handler in existing_app_handlers:
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    log_path_env = os.getenv("APP_LOG_FILE", "./logs/agent_center.log")
    log_file = Path(log_path_env)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = _SafeFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | "
        "run=%(run_id)s uid=%(uid)s project=%(project_uid)s doc=%(doc_uid)s workflow=%(workflow)s session=%(session_id)s | "
        "%(message)s"
    )
    context_filter = _ContextFilter()
    namespace_filter = _NamespaceFilter(APP_LOG_PREFIXES)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(final_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    console_handler.addFilter(namespace_filter)
    setattr(console_handler, _HANDLER_MARKER, True)
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(final_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)
    file_handler.addFilter(namespace_filter)
    setattr(file_handler, _HANDLER_MARKER, True)
    root.addHandler(file_handler)

    logging.getLogger(logger_name).setLevel(final_level)
