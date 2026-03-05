def session_key(project_uid: str, session_uid: str, mode: str) -> str:
    return f"{mode}:{project_uid}:{session_uid}"


def conversation_key(project_uid: str, session_uid: str) -> str:
    return f"{project_uid}:{session_uid}"


def scope_signature(scope_docs: list[dict]) -> str:
    return ",".join(sorted(str(item.get("uid")) for item in scope_docs if item.get("uid")))
