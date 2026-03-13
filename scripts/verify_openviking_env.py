from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from http.client import HTTPResponse
from typing import cast
from urllib import error, request

DEFAULT_VIKING_BASE_URL = "http://localhost:8080"
DEFAULT_TIMEOUT_SECONDS = 3.0
REQUIRED_ENV_VARS = ("OPENAI_API_KEY",)


@dataclass(frozen=True)
class VerificationResult:
    name: str
    ok: bool
    detail: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify OpenViking runtime environment.")
    _ = parser.add_argument(
        "--base-url",
        default=os.getenv("VIKING_BASE_URL", DEFAULT_VIKING_BASE_URL),
        help="OpenViking base URL used for health checks.",
    )
    _ = parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds for health checks.",
    )
    return parser.parse_args(argv)


def check_required_env() -> tuple[bool, str]:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name, "").strip()]
    if missing:
        return False, f"Missing required environment variable(s): {', '.join(missing)}"
    return True, "Required environment variables are present"


def check_openviking_sdk() -> tuple[bool, str]:
    try:
        _ = importlib.import_module("openviking")
    except ModuleNotFoundError as exc:
        return False, f"OpenViking SDK import failed: {exc}"
    return True, "OpenViking SDK import succeeded"


def check_openviking_health(*, base_url: str, timeout_seconds: float) -> tuple[bool, str]:
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        typed_response = cast(
            HTTPResponse,
            request.urlopen(health_url, timeout=timeout_seconds),  # nosec B310
        )
        with typed_response:
            status_code = int(typed_response.status)
            if status_code != 200:
                return False, f"Health endpoint returned status {status_code}"
    except error.URLError as exc:
        return False, f"Health endpoint request failed: {exc}"
    except TimeoutError as exc:
        return False, f"Health endpoint timeout: {exc}"
    return True, f"Health endpoint reachable: {health_url}"


def _run_verification(base_url: str, timeout_seconds: float) -> list[VerificationResult]:
    env_ok, env_detail = check_required_env()
    sdk_ok, sdk_detail = check_openviking_sdk()
    health_ok, health_detail = check_openviking_health(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    return [
        VerificationResult(name="env", ok=env_ok, detail=env_detail),
        VerificationResult(name="sdk", ok=sdk_ok, detail=sdk_detail),
        VerificationResult(name="health", ok=health_ok, detail=health_detail),
    ]


def _print_results(results: list[VerificationResult]) -> None:
    for result in results:
        prefix = "PASS" if result.ok else "FAIL"
        print(f"[{prefix}] {result.name}: {result.detail}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    base_url = str(getattr(args, "base_url", DEFAULT_VIKING_BASE_URL))
    timeout_seconds_raw = float(getattr(args, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    timeout_seconds = max(0.1, timeout_seconds_raw)
    results = _run_verification(base_url, timeout_seconds)
    _print_results(results)
    if all(result.ok for result in results):
        print("OpenViking environment verification PASS")
        return 0
    print("OpenViking environment verification FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
