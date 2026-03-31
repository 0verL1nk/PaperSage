"""Web搜索提供者模块"""
import logging
import os
import re
from typing import Any
from urllib.parse import quote

import httpx
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .utils import (
    DEFAULT_BRAVE_SEARCH_URL,
    DEFAULT_SEARXNG_INSTANCES,
    DEFAULT_WEB_MAX_RESULTS,
    DEFAULT_WEB_TIMEOUT_SECONDS,
    _env_flag,
    _env_value,
    _format_web_results,
    _is_dangerous_query,
    _load_secret,
    _preview,
    _sanitize_query,
)

logger = logging.getLogger(__name__)


class SearchWebInput(BaseModel):
    query: str = Field(
        description="Supplemental public-web query used only when document evidence is insufficient."
    )


def _parse_searxng_instances() -> list[str]:
    configured = str(os.getenv("AGENT_SEARXNG_BASE_URLS", "") or "").strip()
    if configured:
        items = [item.strip().rstrip("/") for item in configured.split(",")]
        return [item for item in items if item]
    return [item.rstrip("/") for item in DEFAULT_SEARXNG_INSTANCES]


def _build_tavily_web_search_client():
    api_key = _load_secret("TAVILY_API_KEY")
    if not api_key:
        return None

    class _TavilyWebSearch:
        def __init__(self, key: str):
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=key)

        def run(self, query: str) -> str:
            response = self._client.search(
                query=query,
                max_results=DEFAULT_WEB_MAX_RESULTS,
                search_depth="basic",
            )
            results = response.get("results") if isinstance(response, dict) else None
            return _format_web_results(
                results,
                title_key="title",
                url_key="url",
                snippet_key="content",
            )

    try:
        return _TavilyWebSearch(api_key)
    except Exception as exc:
        logger.warning("tool.search_web tavily client init failed: %s", exc)
        return None


def _build_brave_web_search_client():
    api_key = _load_secret("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return None
    search_url = _env_value("BRAVE_SEARCH_URL", default=DEFAULT_BRAVE_SEARCH_URL)

    class _BraveWebSearch:
        def __init__(self, key: str, url: str):
            self.api_key = key
            self.search_url = url

        def run(self, query: str) -> str:
            response = httpx.get(
                self.search_url,
                params={
                    "q": query,
                    "count": DEFAULT_WEB_MAX_RESULTS,
                },
                headers={
                    "X-Subscription-Token": self.api_key,
                    "Accept": "application/json",
                    "User-Agent": "llm-app/1.0 (+search_web)",
                },
                timeout=DEFAULT_WEB_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"brave status={response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("brave invalid payload")
            web_block = payload.get("web")
            results = web_block.get("results") if isinstance(web_block, dict) else None
            return _format_web_results(
                results,
                title_key="title",
                url_key="url",
                snippet_key="description",
            )

    return _BraveWebSearch(api_key, search_url)


def _build_searxng_web_search_client():
    instances = _parse_searxng_instances()
    if not instances:
        return None

    class _SearxngWebSearch:
        def __init__(self, base_urls: list[str]):
            self.base_urls = base_urls

        def run(self, query: str) -> str:
            last_error = "unknown error"
            for base_url in self.base_urls:
                try:
                    response = httpx.get(
                        f"{base_url}/search",
                        params={
                            "q": query,
                            "format": "json",
                            "safesearch": "1",
                        },
                        headers={"User-Agent": "llm-app/1.0 (+search_web)"},
                        timeout=DEFAULT_WEB_TIMEOUT_SECONDS,
                    )
                    if response.status_code >= 400:
                        last_error = f"{base_url} status={response.status_code}"
                        continue
                    payload = response.json()
                    if not isinstance(payload, dict):
                        last_error = f"{base_url} invalid json payload"
                        continue
                    results = payload.get("results")
                    rendered = _format_web_results(
                        results,
                        title_key="title",
                        url_key="url",
                        snippet_key="content",
                    )
                    if rendered != "No web search results found.":
                        return rendered
                    last_error = f"{base_url} no results"
                except Exception as exc:
                    last_error = f"{base_url} {exc}"
                    continue
            raise RuntimeError(f"SearXNG unavailable: {last_error}")

    return _SearxngWebSearch(instances)


def _build_wikipedia_web_search_client():
    class _WikipediaWebSearch:
        def run(self, query: str) -> str:
            response = httpx.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": DEFAULT_WEB_MAX_RESULTS,
                },
                headers={"User-Agent": "llm-app/1.0 (+search_web)"},
                timeout=DEFAULT_WEB_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"wikipedia status={response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("wikipedia invalid payload")
            query_block = payload.get("query")
            search_items = query_block.get("search") if isinstance(query_block, dict) else None
            if not isinstance(search_items, list):
                return "No web search results found."
            normalized: list[dict[str, str]] = []
            for item in search_items[:DEFAULT_WEB_MAX_RESULTS]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                snippet = re.sub(r"<[^>]+>", "", snippet)
                pageid = item.get("pageid")
                if isinstance(pageid, int):
                    url = f"https://en.wikipedia.org/?curid={pageid}"
                else:
                    url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                normalized.append(
                    {
                        "title": title or "Untitled",
                        "url": url,
                        "content": snippet,
                    }
                )
            return _format_web_results(
                normalized,
                title_key="title",
                url_key="url",
                snippet_key="content",
            )

    return _WikipediaWebSearch()


def _build_native_web_search_client():
    try:
        from duckduckgo_search import DDGS
    except Exception:
        return None

    class _NativeDuckDuckGoSearch:
        def run(self, query: str) -> str:
            with DDGS(timeout=int(DEFAULT_WEB_TIMEOUT_SECONDS)) as client:
                results = client.text(query, max_results=DEFAULT_WEB_MAX_RESULTS)
            return _format_web_results(
                results, title_key="title", url_key="href", snippet_key="body"
            )

    return _NativeDuckDuckGoSearch()


# Module-level cache for web search clients
_web_search_clients: list[tuple[str, Any]] | None = None


def _ensure_web_search_clients() -> list[tuple[str, Any]]:
    global _web_search_clients
    if _web_search_clients is not None:
        return _web_search_clients

    clients: list[tuple[str, Any]] = []

    tavily_client = _build_tavily_web_search_client()
    if tavily_client is not None:
        clients.append(("tavily_search_api", tavily_client))
        logger.info("tool.search_web provider initialized: tavily_search_api")

    brave_client = _build_brave_web_search_client()
    if brave_client is not None:
        clients.append(("brave_search_api", brave_client))
        logger.info("tool.search_web provider initialized: brave_search_api")

    searxng_client = _build_searxng_web_search_client()
    if searxng_client is not None:
        clients.append(("searxng_public_pool", searxng_client))
        logger.info("tool.search_web provider initialized: searxng_public_pool")

    wikipedia_client = _build_wikipedia_web_search_client()
    if wikipedia_client is not None:
        clients.append(("wikipedia_api", wikipedia_client))
        logger.info("tool.search_web provider initialized: wikipedia_api")

    if not clients:
        logger.warning("tool.search_web no primary provider initialized")

    if not clients:
        allow_ddg_fallback = _env_flag("AGENT_WEB_ENABLE_DDG_FALLBACK", default=False)
        if allow_ddg_fallback:
            fallback_client = _build_native_web_search_client()
            if fallback_client is not None:
                clients.append(("native_duckduckgo_search", fallback_client))
                logger.info("tool.search_web provider fallback initialized: native_duckduckgo_search")
            else:
                try:
                    native_client = DuckDuckGoSearchRun()
                    clients.append(("langchain_duckduckgo_search", native_client))
                    logger.info("tool.search_web provider fallback initialized: langchain_duckduckgo_search")
                except Exception:
                    logger.warning("tool.search_web no fallback provider available")

    _web_search_clients = clients
    return _web_search_clients


def _run_web_search_internal(query: str) -> tuple[str | None, str | None, str | None]:
    clients = _ensure_web_search_clients()
    if not clients:
        return (
            None,
            None,
            (
                "Web search is unavailable in current environment. "
                "Set BRAVE_SEARCH_API_KEY in .env, or configure AGENT_SEARXNG_BASE_URLS, "
                "or enable AGENT_WEB_ENABLE_DDG_FALLBACK=1."
            ),
        )
    errors: list[str] = []
    for provider_name, provider in clients:
        try:
            response = provider.run(query)
            response_text = response if isinstance(response, str) else str(response)
            if not response_text or response_text.strip() == "No web search results found.":
                errors.append(f"{provider_name}: no results")
                continue
            return provider_name, response_text, None
        except Exception as exc:
            errors.append(f"{provider_name}: {exc}")
            logger.info("tool.search_web provider failed: %s (%s)", provider_name, exc)
    return None, None, f"Web search failed: {' | '.join(errors)}"


@tool(
    "search_web",
    description="Search public web content for supplemental context.",
    args_schema=SearchWebInput,
)
def search_web(query: str) -> str:
    safe_query = _sanitize_query(query)
    logger.info(
        "tool.search_web called: query_len=%s query_preview=%s",
        len(safe_query),
        _preview(safe_query),
    )
    if not safe_query:
        logger.warning("tool.search_web blocked: empty query after sanitization")
        return "Web search query is empty after sanitization."
    if _is_dangerous_query(safe_query):
        logger.warning("tool.search_web blocked by policy")
        return "Blocked by tool policy: query appears unsafe for web search."
    provider_name, response_text, error_text = _run_web_search_internal(safe_query)
    if response_text is None:
        logger.warning("tool.search_web unavailable: %s", error_text)
        return str(error_text or "Web search failed.")
    logger.info(
        "tool.search_web success: provider=%s response_len=%s",
        provider_name,
        len(response_text),
    )
    return response_text
