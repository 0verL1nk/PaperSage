from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "python_cleanup.py"
    spec = importlib.util.spec_from_file_location("python_cleanup", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_vulture_cmd_supports_whitelist_flag():
    module = _load_module()

    command = module.build_vulture_cmd(
        ("agent", "ui"),
        min_confidence=80,
        make_whitelist=True,
    )

    assert command[:3] == [module.sys.executable, "-m", "vulture"]
    assert "--config" in command
    assert "--make-whitelist" in command
    assert command[-2:] == ["agent", "ui"]


def test_run_fix_safe_runs_ruff_before_autoflake(monkeypatch):
    module = _load_module()
    calls: list[list[str]] = []

    def _fake_run(command, *, dry_run):
        assert dry_run is False
        calls.append(list(command))
        return 0

    monkeypatch.setattr(module, "run_command", _fake_run)
    config = module.CleanupConfig(paths=("agent",), dry_run=False)

    status = module.run_fix_safe(config)

    assert status == 0
    assert len(calls) == 2
    assert calls[0][:4] == [module.sys.executable, "-m", "ruff", "check"]
    assert "--fix" in calls[0]
    assert calls[1][:3] == [module.sys.executable, "-m", "autoflake"]


def test_main_deadcode_uses_defaults(monkeypatch):
    module = _load_module()
    captured = {}

    def _fake_deadcode(config):
        captured["paths"] = config.paths
        captured["min_confidence"] = config.min_confidence
        captured["make_whitelist"] = config.make_whitelist
        return 0

    monkeypatch.setattr(module, "run_deadcode", _fake_deadcode)
    monkeypatch.setattr(module, "_validate_paths", lambda paths: None)

    status = module.main(["deadcode"])

    assert status == 0
    assert captured["paths"] == module.DEFAULT_PATHS
    assert captured["min_confidence"] == 60
    assert captured["make_whitelist"] is False
