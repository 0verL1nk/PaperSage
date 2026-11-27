# Parallel web search

PaperSage can use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for supplemental public-web evidence through the existing `search_web` tool.
Anonymous search uses Fast mode and needs no Parallel account or API key.

Install the optional MCP client alongside the application's dependencies:

```bash
uv sync --extra parallel
```

Set the environment variable before starting PaperSage, then restart the backend:

```bash
export AGENT_WEB_SEARCH_PROVIDER=parallel
make run
```

This is a process environment variable, not a setting read from `.env` by the
search tool. For a packaged desktop installation, set it in the environment that
launches the application and install an application build containing the
`parallel` extra.

Ask a question needing current public sources. The research agent calls
`search_web` and receives source URLs and excerpts from Parallel at
`https://search.parallel.ai/mcp`, using the MCP SDK's Streamable HTTP transport.
Queries are sent to that external service only when the tool is called.
This integration exposes search; it does not add Parallel's page-fetching tool.

`AGENT_WEB_SEARCH_PROVIDER` accepts:

| Value | Behavior |
| --- | --- |
| unset or `auto` | Existing chain: configured Brave, SearXNG, Firecrawl, Wikipedia, then optional DuckDuckGo. Existing credentials and provider disable flags continue to apply. |
| `parallel` | Only Parallel, with no implicit fallback to another provider after an error or empty result. No Parallel credentials are loaded or sent. |
| `off` | Disable public-web search. Academic `search_papers` remains available. |

Parallel's free service has usage limits. Timeouts and service failures return a
search failure through the existing tool contract, so the agent can report
insufficient evidence. A single eight-second deadline covers connection,
discovery and search. Async runtime cancellation closes the MCP connection.
Existing query policy checks and evaluation overrides apply before network
execution, and capability manifests still determine which agents receive the tool.

To restore the original provider chain, unset `AGENT_WEB_SEARCH_PROVIDER` or set
it to `auto`, then restart the backend. Remove the optional extra with `uv sync`
if it is no longer needed.
