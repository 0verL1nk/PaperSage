from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_PATHS = (
    "main.py",
    "agent",
    "ui",
    "pages",
    "utils",
    "scripts",
    "tests",
)
DEFAULT_RUFF_SELECT = ("F401", "F841")
SAFE_RUFF_FIX_SELECT = ("F401",)


@dataclass(frozen=True)
class CleanupConfig:
    paths: tuple[str, ...]
    min_confidence: int = 60
    make_whitelist: bool = False
    dry_run: bool = False


def build_ruff_check_cmd(paths: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--config",
        "pyproject.toml",
        "--select",
        ",".join(DEFAULT_RUFF_SELECT),
        *paths,
    ]


def build_ruff_fix_cmd(paths: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--config",
        "pyproject.toml",
        "--fix",
        "--select",
        ",".join(SAFE_RUFF_FIX_SELECT),
        *paths,
    ]


def build_autoflake_fix_cmd(paths: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "autoflake",
        "--config",
        "pyproject.toml",
        *paths,
    ]


def build_vulture_cmd(
    paths: Sequence[str],
    *,
    min_confidence: int,
    make_whitelist: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vulture",
        "--config",
        "pyproject.toml",
        "--min-confidence",
        str(min_confidence),
    ]
    if make_whitelist:
        command.append("--make-whitelist")
    command.extend(paths)
    return command


def run_command(command: Sequence[str], *, dry_run: bool) -> int:
    printable = shlex.join(command)
    print(f"[cleanup] {printable}")
    if dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def run_check(config: CleanupConfig) -> int:
    return run_command(build_ruff_check_cmd(config.paths), dry_run=config.dry_run)


def run_fix_safe(config: CleanupConfig) -> int:
    ruff_status = run_command(build_ruff_fix_cmd(config.paths), dry_run=config.dry_run)
    if ruff_status != 0:
        return ruff_status
    return run_command(build_autoflake_fix_cmd(config.paths), dry_run=config.dry_run)


def run_deadcode(config: CleanupConfig) -> int:
    return run_command(
        build_vulture_cmd(
            config.paths,
            min_confidence=config.min_confidence,
            make_whitelist=config.make_whitelist,
        ),
        dry_run=config.dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe cleanup helper for Python unused imports/variables and dead-code audits."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("check", "fix-safe", "deadcode"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument(
            "paths",
            nargs="*",
            default=list(DEFAULT_PATHS),
            help="Target files or directories. Defaults to the project Python paths.",
        )
        if name == "deadcode":
            subparser.add_argument(
                "--min-confidence",
                type=int,
                default=60,
                help="Vulture minimum confidence threshold.",
            )
            subparser.add_argument(
                "--make-whitelist",
                action="store_true",
                help="Print a Vulture whitelist instead of a dead-code report.",
            )

    return parser


def _normalize_paths(raw_paths: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in raw_paths:
        candidate = str(raw).strip()
        if not candidate:
            continue
        normalized.append(candidate)
    return tuple(normalized or DEFAULT_PATHS)


def _validate_paths(paths: Sequence[str]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        joined = ", ".join(sorted(missing))
        raise SystemExit(f"Paths do not exist: {joined}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = _normalize_paths(args.paths)
    _validate_paths(paths)

    config = CleanupConfig(
        paths=paths,
        min_confidence=int(getattr(args, "min_confidence", 60)),
        make_whitelist=bool(getattr(args, "make_whitelist", False)),
        dry_run=bool(args.dry_run),
    )

    if args.command == "check":
        return run_check(config)
    if args.command == "fix-safe":
        return run_fix_safe(config)
    return run_deadcode(config)


if __name__ == "__main__":
    raise SystemExit(main())
