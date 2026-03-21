"""共享工具函数"""
import os
import re
from typing import Any

# 常量定义
DEFAULT_MAX_QUERY_CHARS = 1200
DEFAULT_WEB_MAX_RESULTS = 5
DEFAULT_WEB_TIMEOUT_SECONDS = 8.0
DEFAULT_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_SEARXNG_INSTANCES = (
    "https://searx.be",
    "https://search.inetol.net",
    "https://opnxng.com",
)

_DANGEROUS_QUERY_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"system\s+prompt",
    r"reveal\s+.*(api\s*key|token|password)",
    r"(api\s*key|access\s*token|password)\s*[:=]",
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bssh\b",
]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def _read_from_dotenv(name: str, dotenv_path: str = ".env") -> str:
    try:
        with open(dotenv_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                if key.strip() != name:
                    continue
                value = raw_value.strip()
                if not value:
                    return ""
                if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
                    value = value[1:-1]
                return value.strip()
    except Exception:
        return ""
    return ""


def _load_secret(name: str) -> str:
    value = _env_value(name, default="")
    if value:
        return value
    return _read_from_dotenv(name)


def _sanitize_query(query: str) -> str:
    value = query.strip()
    if len(value) > DEFAULT_MAX_QUERY_CHARS:
        value = value[:DEFAULT_MAX_QUERY_CHARS]
    return value


def _normalize_query_cache_key(query: str) -> str:
    value = _sanitize_query(query).lower()
    if not value:
        return ""
    normalized = value.replace("-", " ").replace("_", " ").replace("/", " ")
    raw_tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        if "." in token:
            try:
                numeric = float(token)
            except ValueError:
                pass
            else:
                if numeric.is_integer():
                    token = str(int(numeric))
        tokens.append(token)
    if not tokens:
        return value
    return " ".join(sorted(tokens))


def _query_similarity_tokens(query: str) -> tuple[str, ...]:
    key = _normalize_query_cache_key(query)
    if not key:
        return ()
    return tuple(token for token in key.split() if token)


def _query_overlap_score(query_a: str, query_b: str) -> float:
    tokens_a = set(_query_similarity_tokens(query_a))
    tokens_b = set(_query_similarity_tokens(query_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection_size = len(tokens_a & tokens_b)
    if intersection_size == 0:
        return 0.0
    return intersection_size / min(len(tokens_a), len(tokens_b))


def _preview(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def _is_dangerous_query(query: str) -> bool:
    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in _DANGEROUS_QUERY_PATTERNS)


def _format_web_results(
    results: Any,
    *,
    title_key: str,
    url_key: str,
    snippet_key: str,
) -> str:
    if not isinstance(results, list) or not results:
        return "No web search results found."
    lines: list[str] = []
    for idx, item in enumerate(results[:DEFAULT_WEB_MAX_RESULTS], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get(title_key) or "").strip()
        href = str(item.get(url_key) or "").strip()
        body = str(item.get(snippet_key) or "").strip()
        snippet = _preview(body, limit=180) if body else ""
        lines.append(
            f"{idx}. {title or 'Untitled'}\nURL: {href or 'n/a'}\nSnippet: {snippet or '-'}"
        )
    return "\n\n".join(lines) if lines else "No web search results found."
