import os
import sqlite3
from pathlib import Path

from utils.utils import init_database, save_file_to_database


def test_save_file_to_database_on_fresh_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))

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
            current_time="2026-02-27 22:10:00",
        )
    finally:
        os.chdir(cwd)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT original_filename, uid, uuid FROM files WHERE uid = ?", ("uid-1",)
    )
    row = cursor.fetchone()
    conn.close()

    assert row == ("doc.txt", "uid-1", "local-user")


def test_save_file_to_database_reupload_same_uid_not_duplicated(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))

    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello", encoding="utf-8")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        save_file_to_database(
            original_file_name="doc.txt",
            uid="uid-dup",
            uuid_value="local-user",
            md5_value="md5-dup",
            full_file_path=str(file_path),
            current_time="2026-03-04 20:40:00",
        )
        save_file_to_database(
            original_file_name="doc-renamed.txt",
            uid="uid-dup",
            uuid_value="local-user",
            md5_value="md5-dup",
            full_file_path=str(file_path),
            current_time="2026-03-04 20:41:00",
        )
    finally:
        os.chdir(cwd)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM files WHERE uid = ? AND uuid = ?",
        ("uid-dup", "local-user"),
    )
    count_row = cursor.fetchone()
    cursor.execute(
        "SELECT original_filename FROM files WHERE uid = ? AND uuid = ?",
        ("uid-dup", "local-user"),
    )
    filename_row = cursor.fetchone()
    conn.close()

    assert count_row == (1,)
    assert filename_row == ("doc-renamed.txt",)
