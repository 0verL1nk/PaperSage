"""Optional anonymous Parallel Search MCP client for the public-web tool."""

import asyncio
from datetime import timedelta

import httpx

from .utils import DEFAULT_WEB_TIMEOUT_SECONDS

PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
PARALLEL_USER_AGENT = "PaperSage (+https://github.com/0verL1nk/PaperSage)"
# Match the documented MCP excerpt budget to bound model context consumption.
MAX_PARALLEL_RESULT_CHARS = 25_000


async def search_parallel(query: str) -> str:
    """Search using the MCP SDK's Streamable HTTP transport, with one deadline."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise RuntimeError("Parallel search requires: uv sync --extra parallel") from exc

    # Bound initialization, discovery and execution together. Cancellation exits
    # both SDK contexts and closes the HTTP connection instead of leaving a worker.
    async with asyncio.timeout(DEFAULT_WEB_TIMEOUT_SECONDS):
        async with httpx.AsyncClient(
            headers={"User-Agent": PARALLEL_USER_AGENT},
            timeout=DEFAULT_WEB_TIMEOUT_SECONDS,
        ) as client:
            async with streamable_http_client(PARALLEL_MCP_URL, http_client=client) as streams:
                async with ClientSession(
                    streams[0], streams[1],
                    read_timeout_seconds=timedelta(seconds=DEFAULT_WEB_TIMEOUT_SECONDS),
                ) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    if not any(item.name == "web_search" for item in tools.tools):
                        raise RuntimeError("Parallel MCP does not expose web_search")
                    result = await session.call_tool(
                        "web_search", {"objective": query, "search_queries": [query]},
                    )
    if result.isError:
        raise RuntimeError("Parallel MCP returned a search error")
    text = "\n\n".join(
        item.text for item in result.content if item.type == "text" and item.text.strip()
    )
    return text[:MAX_PARALLEL_RESULT_CHARS] if text else "No web search results found."


def run_parallel(query: str) -> str:
    """Bridge the existing synchronous search tool to the asynchronous MCP SDK."""
    return asyncio.run(search_parallel(query))


async def search_parallel_result(query: str) -> str:
    """Expose bounded, credential-free diagnostics through the tool contract."""
    try:
        return await search_parallel(query)
    except TimeoutError:
        return "Web search failed: Parallel MCP timed out."
    except Exception:
        # SDK exception groups may include request headers or response bodies.
        # Never relay them into model-visible output or fall back to a provider
        # the user did not select. CancelledError remains uncaught.
        return "Web search failed: Parallel MCP unavailable. Check the parallel extra and retry later."
