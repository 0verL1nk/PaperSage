# Design

## Decisions

### Explicit provider selection

Use a process environment variable with `auto`, `parallel` and `off` values.
The existing free keyless chain remains the default. Parallel selection is
exclusive so an error never sends a query to an unexpected alternative service.
An enable flag appending Parallel to the fallback chain was rejected because it
would not provide predictable explicit selection.

### Standard MCP transport

Use the official Python MCP SDK's Streamable HTTP client. Initialize, discover
`web_search` and execute it with the original sanitized query as objective and
search query. Retain the server's text excerpts and URLs, capped at 25,000
characters. Do not load ambient Parallel credentials. Set a PaperSage User-Agent
on every owned HTTP request.

### Async cancellation and bounded errors

Attach a native coroutine to the existing LangChain tool. Parallel calls run
asynchronously with an eight-second deadline covering the entire MCP exchange.
The synchronous entrypoint bridges through `asyncio.run`; existing providers keep
their existing sync implementation. Cancellation unwinds the SDK and HTTP client
contexts. Service exceptions become bounded messages without provider response
details, which may contain sensitive information.

### Optional installation

Keep the MCP SDK in the `parallel` extra and import it only on use. Existing
installations and search providers do not require new runtime dependencies.

## Risks and rollback

Parallel can be unavailable or rate limited. The agent receives a failure instead
of silently switching services. Restore `auto` to use the original chain; revert
this change to remove the provider, selector and optional extra.

## Validation

Use mocked HTTP transport under the real MCP SDK and `search_web` dispatcher,
including a scripted model run through `create_runtime_agent`. Check default
configuration, saved credentials, override and policy behavior, disabled search,
timeouts, cancellation, errors, destination and User-Agent. Retain one live
anonymous caller observation separately from network-free integration tests.
