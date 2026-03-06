import io
import logging

from agent.logging_utils import _NamespaceFilter, configure_application_logging


def test_configure_application_logging_removes_legacy_unmarked_handlers(monkeypatch, tmp_path):
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    legacy_handler = logging.StreamHandler(io.StringIO())
    legacy_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "run=%(run_id)s uid=%(uid)s project=%(project_uid)s doc=%(doc_uid)s workflow=%(workflow)s session=%(session_id)s | "
            "%(message)s"
        )
    )
    legacy_handler.addFilter(_NamespaceFilter(("llm_app",)))
    root.addHandler(legacy_handler)

    monkeypatch.setenv("APP_LOG_FILE", str(tmp_path / "agent_center.log"))

    try:
        configure_application_logging(default_level="INFO")
        assert legacy_handler not in root.handlers
        marked_handlers = [
            handler
            for handler in root.handlers
            if getattr(handler, "_llm_app_logging_handler", False)
        ]
        assert len(marked_handlers) == 2
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        for handler in previous_handlers:
            root.addHandler(handler)
