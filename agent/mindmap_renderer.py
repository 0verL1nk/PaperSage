import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _default_cli_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "bin" / "mindmap-cli"


def resolve_mindmap_cli_path() -> Path:
    override = os.getenv("MINDMAP_CLI_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_cli_path()


def render_mindmap_html_with_cli(
    mindmap_data: dict[str, Any],
    *,
    title: str = "思维导图",
    timeout_sec: float = 8.0,
) -> tuple[str | None, str | None]:
    """Render mindmap JSON into interactive HTML via external CLI binary.

    Returns:
        (html, error_message). Exactly one of them is expected to be non-None.
    """
    if not isinstance(mindmap_data, dict) or not mindmap_data:
        return None, "mindmap data is empty"

    cli_path = resolve_mindmap_cli_path()
    if not cli_path.exists():
        return (
            None,
            (
                f"mindmap-cli 不存在: {cli_path}. "
                "请先构建二进制: cd tools/mindmap-cli && go build -o ../../bin/mindmap-cli ."
            ),
        )

    payload = json.dumps(mindmap_data, ensure_ascii=False)
    try:
        result = subprocess.run(
            [
                str(cli_path),
                "-in",
                "-",
                "-out",
                "-",
                "-title",
                title,
            ],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception as exc:
        return None, f"mindmap-cli 执行异常: {exc}"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or "unknown error"
        return None, f"mindmap-cli 执行失败 (code={result.returncode}): {details}"

    html = (result.stdout or "").strip()
    if not html:
        return None, "mindmap-cli 未输出 HTML 内容"
    lowered = html.lower()
    if "<html" not in lowered and "<!doctype html" not in lowered:
        return None, "mindmap-cli 输出内容不是合法 HTML"
    return html, None
