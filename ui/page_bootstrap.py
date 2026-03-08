from copy import deepcopy
from typing import Any, Mapping, MutableMapping

from agent.adapters.sqlite.project_repository import ensure_default_project_for_user
from utils.utils import ensure_local_user, init_database


def bootstrap_page_context(
    *,
    session_state: MutableMapping[str, Any],
    db_name: str = "./database.sqlite",
    ensure_default_project: bool = False,
    sync_existing_files: bool = False,
    state_defaults: Mapping[str, Any] | None = None,
) -> str:
    init_database(db_name)

    raw_uuid = session_state.get("uuid")
    if isinstance(raw_uuid, str) and raw_uuid.strip():
        user_uuid = raw_uuid
    else:
        user_uuid = "local-user"
        session_state["uuid"] = user_uuid

    ensure_local_user(user_uuid, db_name=db_name)

    if state_defaults:
        for key, value in state_defaults.items():
            if key not in session_state:
                session_state[key] = deepcopy(value)

    if ensure_default_project:
        ensure_default_project_for_user(
            user_uuid,
            db_name=db_name,
            sync_existing_files=sync_existing_files,
        )

    return user_uuid
