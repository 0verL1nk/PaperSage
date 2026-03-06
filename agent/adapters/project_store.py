from utils.utils import (
    count_project_session_messages,
    create_project_session,
    delete_project_session,
    ensure_default_project_session,
    list_project_files,
    list_project_session_messages,
    list_project_session_messages_page,
    list_project_sessions,
    list_projects,
    save_project_session_messages,
    update_project_session,
)


def list_user_projects(*, uuid: str, include_archived: bool = False):
    return list_projects(uuid, include_archived=include_archived)


def list_project_files_for_user(*, project_uid: str, uuid: str, active_only: bool = True):
    return list_project_files(project_uid=project_uid, uuid=uuid, active_only=active_only)


def list_sessions_for_project(*, project_uid: str, uuid: str):
    return list_project_sessions(project_uid=project_uid, uuid=uuid)


def ensure_default_session_for_project(*, project_uid: str, uuid: str) -> None:
    ensure_default_project_session(project_uid=project_uid, uuid=uuid)


def create_session_for_project(*, project_uid: str, uuid: str, session_name: str):
    return create_project_session(
        project_uid=project_uid,
        uuid=uuid,
        session_name=session_name,
    )


def delete_session_for_project(*, session_uid: str, project_uid: str, uuid: str) -> None:
    delete_project_session(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
    )


def update_session_for_project(
    *,
    session_uid: str,
    project_uid: str,
    uuid: str,
    session_name: str,
    is_pinned: int,
) -> None:
    update_project_session(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        session_name=session_name,
        is_pinned=is_pinned,
    )


def list_session_messages_for_project(*, session_uid: str, project_uid: str, uuid: str):
    return list_project_session_messages(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
    )


def count_session_messages_for_project(*, session_uid: str, project_uid: str, uuid: str) -> int:
    return count_project_session_messages(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
    )


def list_session_messages_page_for_project(
    *,
    session_uid: str,
    project_uid: str,
    uuid: str,
    offset: int,
    limit: int,
):
    return list_project_session_messages_page(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        offset=offset,
        limit=limit,
    )


def save_session_messages_for_project(
    *,
    session_uid: str,
    project_uid: str,
    uuid: str,
    messages: list[dict],
) -> None:
    save_project_session_messages(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        messages=messages,
    )
