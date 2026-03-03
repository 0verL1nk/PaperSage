from pathlib import Path

from utils.utils import (
    create_project,
    ensure_default_project_for_user,
    ensure_local_user,
    init_database,
    list_projects,
    update_project,
)


def test_default_project_created_for_local_user(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))

    projects = list_projects("local-user", db_name=str(db_path))
    assert len(projects) >= 1
    assert projects[0]["project_uid"]


def test_create_and_archive_project(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))

    created = create_project(
        uuid="local-user",
        project_name="my project",
        description="demo",
        db_name=str(db_path),
    )
    assert created["project_name"] == "my project"

    updated = update_project(
        project_uid=created["project_uid"],
        uuid="local-user",
        archived=1,
        db_name=str(db_path),
    )
    assert updated

    active_projects = list_projects("local-user", db_name=str(db_path))
    assert all(item["project_uid"] != created["project_uid"] for item in active_projects)


def test_ensure_default_project_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))

    uid_first = ensure_default_project_for_user("local-user", db_name=str(db_path))
    uid_second = ensure_default_project_for_user("local-user", db_name=str(db_path))

    assert uid_first == uid_second
