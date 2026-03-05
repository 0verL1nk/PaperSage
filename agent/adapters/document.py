from typing import Any

from utils import extract_files
from utils.utils import get_content_by_uid, save_content_to_database


def extract_document_payload(file_path: str) -> dict[str, Any]:
    return extract_files(file_path)


def load_cached_extraction(uid: str) -> str | None:
    cached = get_content_by_uid(uid, "file_extraction")
    if isinstance(cached, str) and cached.strip():
        return cached
    return None


def save_cached_extraction(
    *,
    uid: str,
    file_path: str,
    content: str,
) -> None:
    save_content_to_database(
        uid=uid,
        file_path=file_path,
        content=content,
        content_type="file_extraction",
    )
