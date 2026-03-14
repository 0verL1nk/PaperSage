from __future__ import annotations

# pyright: reportPrivateUsage=false
import signal

import pytest

import main


class _FakeProcess:
    def __init__(self) -> None:
        self.terminated: bool = False
        self.killed: bool = False
        self.wait_timeout: float | None = None

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeout = timeout
        return 0

    def kill(self) -> None:
        self.killed = True


def _fail_if_called() -> None:
    raise AssertionError("unexpected call")


def test_cli_entry_skips_spawn_when_openviking_is_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned = False
    waited = False
    streamlit_runs = 0

    def _probe(_base_url: str) -> bool:
        return True

    def _spawn(_base_url: str) -> _FakeProcess:
        nonlocal spawned
        spawned = True
        return _FakeProcess()

    def _wait(_base_url: str) -> bool:
        nonlocal waited
        waited = True
        return True

    def _run_streamlit() -> None:
        nonlocal streamlit_runs
        streamlit_runs = 1

    monkeypatch.setattr(main, "_probe_openviking_health", _probe)
    monkeypatch.setattr(main, "_is_openviking_runnable", lambda: True)
    monkeypatch.setattr(main, "_spawn_openviking_server", _spawn)
    monkeypatch.setattr(main, "_wait_for_openviking_ready", _wait)
    monkeypatch.setattr(main, "_run_streamlit_main", _run_streamlit)

    main._cli_entry()

    assert streamlit_runs == 1
    assert spawned is False
    assert waited is False


def test_cli_entry_fails_fast_when_openviking_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _probe(_base_url: str) -> bool:
        return False

    def _runtime_available() -> bool:
        return False

    def _print_error(message: str) -> None:
        pass

    monkeypatch.setattr(main, "_probe_openviking_health", _probe)
    monkeypatch.setattr(main, "_is_openviking_runnable", _runtime_available)
    monkeypatch.setattr(main, "_print_startup_error", _print_error)
    monkeypatch.setattr(main, "_spawn_openviking_server", _fail_if_called)
    monkeypatch.setattr(main, "_run_streamlit_main", _fail_if_called)

    with pytest.raises(SystemExit, match="1"):
        main._cli_entry()


def test_cli_entry_spawns_and_cleans_up_on_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _FakeProcess()
    streamlit_runs = 0

    def _probe(_base_url: str) -> bool:
        return False

    def _spawn(_base_url: str) -> _FakeProcess:
        return fake_proc

    def _wait(_base_url: str) -> bool:
        return True

    def _run_streamlit() -> None:
        nonlocal streamlit_runs
        streamlit_runs = 1

    monkeypatch.setattr(main, "_probe_openviking_health", _probe)
    monkeypatch.setattr(main, "_is_openviking_runnable", lambda: True)
    monkeypatch.setattr(main, "_spawn_openviking_server", _spawn)
    monkeypatch.setattr(main, "_wait_for_openviking_ready", _wait)
    monkeypatch.setattr(main, "_run_streamlit_main", _run_streamlit)

    main._cli_entry()

    assert streamlit_runs == 1
    assert fake_proc.terminated is True


def test_cli_entry_fails_when_spawn_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _probe(_base_url: str) -> bool:
        return False

    def _spawn(_base_url: str) -> _FakeProcess:
        raise FileNotFoundError("openviking-server binary not found")

    monkeypatch.setattr(main, "_probe_openviking_health", _probe)
    monkeypatch.setattr(main, "_is_openviking_runnable", lambda: True)
    monkeypatch.setattr(main, "_spawn_openviking_server", _spawn)
    monkeypatch.setattr(main, "_run_streamlit_main", _fail_if_called)

    with pytest.raises(SystemExit, match="1"):
        main._cli_entry()


def test_cli_entry_fails_when_not_ready_after_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_proc = _FakeProcess()

    def _probe(_base_url: str) -> bool:
        return False

    def _spawn(_base_url: str) -> _FakeProcess:
        return fake_proc

    def _wait(_base_url: str) -> bool:
        return False

    monkeypatch.setattr(main, "_probe_openviking_health", _probe)
    monkeypatch.setattr(main, "_is_openviking_runnable", lambda: True)
    monkeypatch.setattr(main, "_spawn_openviking_server", _spawn)
    monkeypatch.setattr(main, "_wait_for_openviking_ready", _wait)
    monkeypatch.setattr(main, "_run_streamlit_main", _fail_if_called)

    with pytest.raises(SystemExit, match="1"):
        main._cli_entry()

    assert fake_proc.terminated is True


def test_cli_entry_spawn_success_starts_streamlit_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _FakeProcess()
    streamlit_runs = 0

    def _probe(_base_url: str) -> bool:
        return False

    def _spawn(_base_url: str) -> _FakeProcess:
        return fake_proc

    def _wait(_base_url: str) -> bool:
        return True

    def _run_streamlit() -> None:
        nonlocal streamlit_runs
        streamlit_runs = 1

    monkeypatch.setattr(main, "_probe_openviking_health", _probe)
    monkeypatch.setattr(main, "_is_openviking_runnable", lambda: True)
    monkeypatch.setattr(main, "_spawn_openviking_server", _spawn)
    monkeypatch.setattr(main, "_wait_for_openviking_ready", _wait)
    monkeypatch.setattr(main, "_run_streamlit_main", _run_streamlit)

    main._cli_entry()

    assert streamlit_runs == 1
    assert fake_proc.terminated is True


def test_cli_entry_spawn_failure_cleans_up_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_proc = _FakeProcess()

    def _probe(_base_url: str) -> bool:
        return False

    def _spawn(_base_url: str) -> _FakeProcess:
        return fake_proc

    def _wait(_base_url: str) -> bool:
        return False

    monkeypatch.setattr(main, "_probe_openviking_health", _probe)
    monkeypatch.setattr(main, "_is_openviking_runnable", lambda: True)
    monkeypatch.setattr(main, "_spawn_openviking_server", _spawn)
    monkeypatch.setattr(main, "_wait_for_openviking_ready", _wait)
    monkeypatch.setattr(main, "_run_streamlit_main", _fail_if_called)

    with pytest.raises(SystemExit, match="1"):
        main._cli_entry()

    assert fake_proc.terminated is True


def test_openviking_signal_handler_cleans_up_spawned_process() -> None:
    fake_proc = _FakeProcess()
    handler = main._build_openviking_signal_handler(fake_proc)

    with pytest.raises(SystemExit, match="130"):
        handler(signal.SIGINT, None)

    assert fake_proc.terminated is True
