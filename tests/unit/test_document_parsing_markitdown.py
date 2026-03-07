from pathlib import Path

from utils.utils import extract_files


def test_extract_files_prefers_markitdown(monkeypatch, tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_text("dummy", encoding="utf-8")

    monkeypatch.delenv("DOC_PARSE_BACKEND", raising=False)
    monkeypatch.setattr(
        "utils.utils._extract_text_with_markitdown",
        lambda _path: "# Title\nbody",
    )

    def _raise_legacy(_file_path: str, _file_type: str):
        raise AssertionError("legacy parser should not be called")

    monkeypatch.setattr("utils.utils._extract_text_with_legacy", _raise_legacy)

    result = extract_files(str(sample))
    assert result["result"] == 1
    assert result["parser"] == "markitdown"
    assert result["format"] == "markdown"
    assert "# Title" in result["text"]


def test_extract_files_fallbacks_to_legacy(monkeypatch, tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_text("dummy", encoding="utf-8")

    monkeypatch.delenv("DOC_PARSE_BACKEND", raising=False)

    def _raise_markitdown(_path: str) -> str:
        raise RuntimeError("markitdown unavailable")

    monkeypatch.setattr("utils.utils._extract_text_with_markitdown", _raise_markitdown)
    def _legacy(*, file_path: str, file_type: str):
        _ = (file_path, file_type)
        return ("legacy {content}", "pymupdf", "plain")

    monkeypatch.setattr("utils.utils._extract_text_with_legacy", _legacy)

    result = extract_files(str(sample))
    assert result["result"] == 1
    assert result["parser"] == "pymupdf"
    assert result["format"] == "plain"
    assert result["text"] == "legacy {{content}}"


def test_extract_files_respects_legacy_backend(monkeypatch, tmp_path: Path) -> None:
    sample = tmp_path / "sample.docx"
    sample.write_text("dummy", encoding="utf-8")

    monkeypatch.setenv("DOC_PARSE_BACKEND", "legacy")

    def _raise_markitdown(_path: str) -> str:
        raise AssertionError("markitdown parser should not be called")

    monkeypatch.setattr("utils.utils._extract_text_with_markitdown", _raise_markitdown)
    def _legacy(*, file_path: str, file_type: str):
        _ = (file_path, file_type)
        return ("legacy", "pymupdf", "plain")

    monkeypatch.setattr("utils.utils._extract_text_with_legacy", _legacy)

    result = extract_files(str(sample))
    assert result["result"] == 1
    assert result["parser"] == "pymupdf"
    assert result["format"] == "plain"
    assert result["text"] == "legacy"


def test_extract_files_respects_mineru_backend_for_pdf(
    monkeypatch, tmp_path: Path
) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_text("dummy", encoding="utf-8")

    monkeypatch.setenv("DOC_PARSE_BACKEND", "mineru")

    def _raise_markitdown(_path: str) -> str:
        raise AssertionError("markitdown parser should not be called")

    def _raise_legacy(*_args, **_kwargs):
        raise AssertionError("legacy parser should not be called for PDF")

    monkeypatch.setattr("utils.utils._extract_text_with_markitdown", _raise_markitdown)
    monkeypatch.setattr("utils.utils._extract_text_with_legacy", _raise_legacy)
    monkeypatch.setattr(
        "utils.utils._extract_text_with_mineru_api",
        lambda _path: "# MinerU\nparsed {content}",
    )

    result = extract_files(str(sample))
    assert result["result"] == 1
    assert result["parser"] == "mineru-api"
    assert result["format"] == "markdown"
    assert result["text"] == "# MinerU\nparsed {{content}}"


def test_extract_files_respects_mineru_backend_fallback_for_docx(
    monkeypatch, tmp_path: Path
) -> None:
    sample = tmp_path / "sample.docx"
    sample.write_text("dummy", encoding="utf-8")

    monkeypatch.setenv("DOC_PARSE_BACKEND", "mineru")

    def _raise_mineru(_path: str) -> str:
        raise AssertionError("mineru parser should not be called for non-pdf")

    monkeypatch.setattr("utils.utils._extract_text_with_mineru_api", _raise_mineru)
    monkeypatch.setattr(
        "utils.utils._extract_text_with_legacy",
        lambda *, file_path, file_type: (
            "legacy-docx",
            "pymupdf",
            "plain",
        ),
    )

    result = extract_files(str(sample))
    assert result["result"] == 1
    assert result["parser"] == "pymupdf"
    assert result["format"] == "plain"
    assert result["text"] == "legacy-docx"


def test_extract_files_fallbacks_to_legacy_when_mineru_pdf_fails(
    monkeypatch, tmp_path: Path
) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_text("dummy", encoding="utf-8")

    monkeypatch.setenv("DOC_PARSE_BACKEND", "mineru")
    monkeypatch.setattr(
        "utils.utils._extract_text_with_mineru_api",
        lambda _path: (_ for _ in ()).throw(RuntimeError("mineru down")),
    )
    monkeypatch.setattr(
        "utils.utils._extract_text_with_legacy",
        lambda *, file_path, file_type: (
            "legacy-pdf",
            "pymupdf",
            "plain",
        ),
    )

    result = extract_files(str(sample))
    assert result["result"] == 1
    assert result["parser"] == "pymupdf"
    assert result["format"] == "plain"
    assert result["text"] == "legacy-pdf"
