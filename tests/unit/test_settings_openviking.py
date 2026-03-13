from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from agent.settings import (
    DEFAULT_VIKING_BASE_URL,
    DEFAULT_VIKING_MAX_RETRIES,
    DEFAULT_VIKING_RETRY_BACKOFF_SECONDS,
    DEFAULT_VIKING_TIMEOUT_SECONDS,
    load_agent_settings,
)


def test_load_agent_settings_uses_openviking_defaults(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("VIKING_BASE_URL", raising=False)
    monkeypatch.delenv("VIKING_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("VIKING_MAX_RETRIES", raising=False)
    monkeypatch.delenv("VIKING_RETRY_BACKOFF_SECONDS", raising=False)

    settings = load_agent_settings()

    assert settings.viking_base_url == DEFAULT_VIKING_BASE_URL
    assert settings.viking_timeout_seconds == DEFAULT_VIKING_TIMEOUT_SECONDS
    assert settings.viking_max_retries == DEFAULT_VIKING_MAX_RETRIES
    assert settings.viking_retry_backoff_seconds == DEFAULT_VIKING_RETRY_BACKOFF_SECONDS


def test_load_agent_settings_parses_and_clamps_openviking_values(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("VIKING_BASE_URL", "http://openviking:8081")
    monkeypatch.setenv("VIKING_TIMEOUT_SECONDS", "-5")
    monkeypatch.setenv("VIKING_MAX_RETRIES", "-2")
    monkeypatch.setenv("VIKING_RETRY_BACKOFF_SECONDS", "-3")

    settings = load_agent_settings()

    assert settings.viking_base_url == "http://openviking:8081"
    assert settings.viking_timeout_seconds == 0.1
    assert settings.viking_max_retries == 0
    assert settings.viking_retry_backoff_seconds == 0.0
