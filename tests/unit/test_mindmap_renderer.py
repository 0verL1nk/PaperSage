from pathlib import Path
from types import SimpleNamespace

from agent.mindmap_renderer import render_mindmap_html_with_cli


def test_render_mindmap_html_with_cli_returns_error_when_binary_missing(monkeypatch):
    missing_path = Path("/tmp/does-not-exist/mindmap-cli")
    monkeypatch.setattr(
        "agent.mindmap_renderer.resolve_mindmap_cli_path",
        lambda: missing_path,
    )

    html, err = render_mindmap_html_with_cli({"name": "root", "children": []})

    assert html is None
    assert "mindmap-cli 不存在" in str(err)


def test_render_mindmap_html_with_cli_returns_error_when_cli_fails(monkeypatch, tmp_path):
    fake_binary = tmp_path / "mindmap-cli"
    fake_binary.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "agent.mindmap_renderer.resolve_mindmap_cli_path",
        lambda: fake_binary,
    )

    def _fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=2, stdout="", stderr="bad input")

    monkeypatch.setattr("agent.mindmap_renderer.subprocess.run", _fake_run)

    html, err = render_mindmap_html_with_cli({"name": "root", "children": []})

    assert html is None
    assert "code=2" in str(err)
    assert "bad input" in str(err)


def test_render_mindmap_html_with_cli_success(monkeypatch, tmp_path):
    fake_binary = tmp_path / "mindmap-cli"
    fake_binary.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "agent.mindmap_renderer.resolve_mindmap_cli_path",
        lambda: fake_binary,
    )

    captured = {}

    def _fake_run(*args, **kwargs):
        captured["cmd"] = list(args[0]) if args else []
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="<!doctype html><html><body>ok</body></html>",
            stderr="",
        )

    monkeypatch.setattr("agent.mindmap_renderer.subprocess.run", _fake_run)

    html, err = render_mindmap_html_with_cli({"name": "root", "children": []})

    assert err is None
    assert "<html>" in str(html)
    assert "-width" in captured["cmd"]
    assert "1200" in captured["cmd"]
    assert "-height" in captured["cmd"]
    assert "520" in captured["cmd"]
