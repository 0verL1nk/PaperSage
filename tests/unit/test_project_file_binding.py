import os
from pathlib import Path
import sqlite3

from utils.utils import (
    add_file_to_project,
    create_project,
    ensure_default_project_for_user,
    ensure_local_user,
    init_database,
    list_project_files,
    remove_file_from_project,
    save_file_to_database,
)


def test_file_can_bind_multiple_projects(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))

    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello", encoding="utf-8")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        save_file_to_database(
            original_file_name="doc.txt",
            uid="uid-1",
            uuid_value="local-user",
            md5_value="md5-1",
            full_file_path=str(file_path),
            current_time="2026-03-03 10:00:00",
        )
    finally:
        os.chdir(cwd)

    default_project_uid = ensure_default_project_for_user("local-user", db_name=str(db_path))
    second_project = create_project(
        uuid="local-user",
        project_name="p2",
        db_name=str(db_path),
    )

    add_file_to_project(
        project_uid=second_project["project_uid"],
        file_uid="uid-1",
        uuid="local-user",
        db_name=str(db_path),
    )

    files_default = list_project_files(default_project_uid, "local-user", db_name=str(db_path))
    files_second = list_project_files(second_project["project_uid"], "local-user", db_name=str(db_path))

    assert any(item["uid"] == "uid-1" for item in files_default)
    assert any(item["uid"] == "uid-1" for item in files_second)


def test_remove_binding_does_not_delete_other_project_membership(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))
    file_path = tmp_path / "doc2.txt"
    file_path.write_text("hello2", encoding="utf-8")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        save_file_to_database(
            original_file_name="doc2.txt",
            uid="uid-x",
            uuid_value="local-user",
            md5_value="md5-x",
            full_file_path=str(file_path),
            current_time="2026-03-03 10:01:00",
        )
    finally:
        os.chdir(cwd)

    first = create_project(uuid="local-user", project_name="p1", db_name=str(db_path))
    second = create_project(uuid="local-user", project_name="p2", db_name=str(db_path))

    add_file_to_project(first["project_uid"], "uid-x", "local-user", db_name=str(db_path))
    add_file_to_project(second["project_uid"], "uid-x", "local-user", db_name=str(db_path))

    removed = remove_file_from_project(first["project_uid"], "uid-x", "local-user", db_name=str(db_path))
    assert removed

    files_first = list_project_files(first["project_uid"], "local-user", db_name=str(db_path), active_only=False)
    files_second = list_project_files(second["project_uid"], "local-user", db_name=str(db_path), active_only=False)

    assert all(item["uid"] != "uid-x" for item in files_first)
    assert any(item["uid"] == "uid-x" for item in files_second)


def test_list_project_files_deduplicates_duplicate_file_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))

    file_path = tmp_path / "doc3.txt"
    file_path.write_text("hello3", encoding="utf-8")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        save_file_to_database(
            original_file_name="doc3.txt",
            uid="uid-z",
            uuid_value="local-user",
            md5_value="md5-z",
            full_file_path=str(file_path),
            current_time="2026-03-04 20:45:00",
        )
    finally:
        os.chdir(cwd)

    project = create_project(uuid="local-user", project_name="dedupe", db_name=str(db_path))
    add_file_to_project(project["project_uid"], "uid-z", "local-user", db_name=str(db_path))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO files (original_filename, uid, md5, file_path, uuid, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("doc3-copy.txt", "uid-z", "md5-z", str(file_path), "local-user", "2026-03-04 20:46:00"),
    )
    conn.commit()
    conn.close()

    listed = list_project_files(project["project_uid"], "local-user", db_name=str(db_path))

    assert len([item for item in listed if item["uid"] == "uid-z"]) == 1
