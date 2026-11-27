# Optional Parallel Search MCP provider

## Why

`agent/tools/web_search.py` currently tries configured Brave, the public SearXNG
pool, keyless Firecrawl and Wikipedia, with optional DuckDuckGo. It has no
Parallel provider or explicit provider selector. Researchers who prefer
Parallel's free anonymous search cannot select it through the existing tool.

## What Changes

- Add an explicit `AGENT_WEB_SEARCH_PROVIDER=parallel` selection using the MCP
  SDK's Streamable HTTP client and the public anonymous endpoint.
- Preserve the original chain for unset/auto settings. Support `off` to disable
  public-web search, and report invalid selections without network requests.
- Keep policy checks, evaluation overrides and capability exposure controls.
- Add optional dependencies, usage documentation and deterministic tests through
  the normal tool dispatcher and scripted runtime model.

## Non-goals

No default-provider replacement, API-key management, page-fetching tool, UI
settings, document retrieval changes or production-model benchmark.

## Capabilities

- Added: `parallel-web-search` defines the explicit optional MCP provider.

## Impact

The optional `parallel` extra installs the MCP SDK. Only users selecting Parallel
send searches to its endpoint; service failures remain visible to the agent.
