from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import util as importlib_util
from pathlib import Path
from types import FrameType
from typing import Protocol
from urllib import error, request

import streamlit as st

from agent.settings import DEFAULT_VIKING_BASE_URL

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VIKING_HEALTH_TIMEOUT_SECONDS = 2.0
DEFAULT_VIKING_HEALTH_RETRIES = 10
DEFAULT_VIKING_HEALTH_RETRY_INTERVAL_SECONDS = 0.5
OPENVIKING_SERVER_BINARY = "openviking-server"
OPENVIKING_PACKAGE_NAME = "openviking"
OPENVIKING_SHUTDOWN_TIMEOUT_SECONDS = 3.0
SignalHandlerType = Callable[[int, FrameType | None], object] | int | None | signal.Handlers


class _ManagedProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


def run_app() -> None:
    navigation = st.navigation(
        [
            st.Page(
                str(REPO_ROOT / "pages/0_agent_center.py"),
                title="Agent中心",
                icon="🤖",
                default=True,
            ),
            st.Page(
                str(REPO_ROOT / "pages/1_file_center.py"),
                title="文件中心",
                icon="📁",
            ),
            st.Page(
                str(REPO_ROOT / "pages/2_settings.py"),
                title="设置中心",
                icon="⚙️",
            ),
            st.Page(
                str(REPO_ROOT / "pages/3_project_center.py"),
                title="项目中心",
                icon="🗂️",
            ),
        ],
        position="sidebar",
    )

    navigation.run()


def _cli_entry() -> None:  # pyright: ignore[reportUnusedFunction]
    """CLI entry point: `paper-sage` launches the Streamlit app."""
    viking_base_url = os.getenv("VIKING_BASE_URL", DEFAULT_VIKING_BASE_URL).strip()
    if not viking_base_url:
        viking_base_url = DEFAULT_VIKING_BASE_URL

    openviking_proc: _ManagedProcess | None = None
    previous_signal_handlers: dict[int, SignalHandlerType] = {}

    if not _probe_openviking_health(viking_base_url):
        if not _is_openviking_runnable():
            _print_startup_error(
                "OpenViking (`openviking` package) is not installed or not runnable. "
                + f"Please run `{sys.executable} -m pip install {OPENVIKING_PACKAGE_NAME}` "
                + "and ensure the `openviking-server` binary is available in PATH."
            )
            raise SystemExit(1)

        try:
            openviking_proc = _spawn_openviking_server(viking_base_url)
        except FileNotFoundError as exc:
            _print_startup_error(str(exc))
            raise SystemExit(1) from exc

        if not _wait_for_openviking_ready(viking_base_url):
            _cleanup_openviking_process(openviking_proc)
            _print_startup_error(
                "OpenViking failed to become healthy after retries. "
                + "Please verify the `openviking-server` binary can start normally "
                + "and that `VIKING_BASE_URL` matches its listening address."
            )
            raise SystemExit(1)

        signal_handler = _build_openviking_signal_handler(openviking_proc)
        previous_signal_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
        previous_signal_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
        _ = signal.signal(signal.SIGINT, signal_handler)
        _ = signal.signal(signal.SIGTERM, signal_handler)

    try:
        _run_streamlit_main()
    finally:
        if signal.SIGINT in previous_signal_handlers:
            _ = signal.signal(signal.SIGINT, previous_signal_handlers[signal.SIGINT])
        if signal.SIGTERM in previous_signal_handlers:
            _ = signal.signal(signal.SIGTERM, previous_signal_handlers[signal.SIGTERM])
        if openviking_proc is not None:
            _cleanup_openviking_process(openviking_proc)


def _run_streamlit_main() -> None:
    _ = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__],
        check=False,
    )


def _probe_openviking_health(viking_base_url: str) -> bool:
    health_url = f"{viking_base_url.rstrip('/')}/health"
    try:
        with request.urlopen(health_url, timeout=DEFAULT_VIKING_HEALTH_TIMEOUT_SECONDS):  # nosec B310
            return True
    except (error.URLError, TimeoutError):
        return False


def _wait_for_openviking_ready(viking_base_url: str) -> bool:
    for _ in range(DEFAULT_VIKING_HEALTH_RETRIES):
        if _probe_openviking_health(viking_base_url):
            return True
        time.sleep(DEFAULT_VIKING_HEALTH_RETRY_INTERVAL_SECONDS)
    return False


def _spawn_openviking_server(viking_base_url: str) -> subprocess.Popen[str]:
    del viking_base_url

    try:
        return subprocess.Popen(
            [OPENVIKING_SERVER_BINARY],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing required binary `openviking-server`. "
            + "Install OpenViking server and ensure it is available in PATH."
        ) from exc


def _is_openviking_runnable() -> bool:
    package_available = importlib_util.find_spec(OPENVIKING_PACKAGE_NAME) is not None
    binary_available = shutil.which(OPENVIKING_SERVER_BINARY) is not None
    return package_available and binary_available


def _cleanup_openviking_process(process: _ManagedProcess) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        _ = process.wait(timeout=OPENVIKING_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()


def _build_openviking_signal_handler(
    process: _ManagedProcess,
) -> Callable[[int, FrameType | None], None]:
    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        _cleanup_openviking_process(process)
        if signum == signal.SIGINT:
            raise SystemExit(130)
        raise SystemExit(143)

    return _handle_signal


def _print_startup_error(message: str) -> None:
    print(f"[paper-sage] startup error: {message}", file=sys.stderr)


if __name__ == "__main__":
    run_app()
