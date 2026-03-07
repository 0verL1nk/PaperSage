from pathlib import Path

from streamlit.testing.v1 import AppTest

from utils.utils import ensure_local_user, init_database, save_api_key, save_model_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepare_ready_user(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))
    save_api_key("local-user", "test-api-key", db_name=str(db_path))
    save_model_name("local-user", "gpt-4o-mini", db_name=str(db_path))


def test_pages_boot_without_exceptions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_ready_user(tmp_path)

    page_files = [
        "pages/1_file_center.py",
        "pages/2_settings.py",
        "pages/3_project_center.py",
    ]

    for page_file in page_files:
        at = AppTest.from_file(str(REPO_ROOT / page_file))
        at.run()
        assert len(at.exception) == 0, page_file
