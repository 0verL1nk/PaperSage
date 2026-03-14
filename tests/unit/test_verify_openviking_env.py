from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import cast

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch


class VerifyOpenVikingModule(ModuleType):
    def main(self, _argv: Sequence[str] | None = None) -> int: ...

    def check_openviking_sdk(self) -> tuple[bool, str]: ...

    def check_openviking_health(
        self,
        *,
        _base_url: str,
        _timeout_seconds: float,
    ) -> tuple[bool, str]: ...


def _load_module() -> VerifyOpenVikingModule:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "verify_openviking_env.py"
    spec = importlib.util.spec_from_file_location("verify_openviking_env", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(VerifyOpenVikingModule, module)


def test_main_fails_when_required_env_is_missing(capsys: CaptureFixture[str]):
    module = _load_module()

    exit_code = module.main(
        [
            "--base-url",
            "http://localhost:8080",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.out
    assert "OPENAI_API_KEY" in captured.out


def test_main_passes_with_stubbed_dependencies(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
):
    module = _load_module()

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    monkeypatch.setattr(module, "check_openviking_sdk", lambda: (True, "sdk ok"))

    def _fake_health(*, base_url: str, timeout_seconds: float) -> tuple[bool, str]:
        return True, f"{base_url} healthy in {timeout_seconds}"

    monkeypatch.setattr(
        module,
        "check_openviking_health",
        _fake_health,
    )

    exit_code = module.main(
        [
            "--base-url",
            "http://localhost:8080",
            "--timeout-seconds",
            "1.5",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out
    assert "sdk ok" in captured.out
    assert "healthy" in captured.out
