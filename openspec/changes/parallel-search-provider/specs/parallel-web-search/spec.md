## ADDED Requirements

### Requirement: Explicit optional Parallel MCP search

Researchers SHALL be able to select anonymous Parallel web search through the
existing `search_web` tool without changing unconfigured installations.

#### Purpose:

Provide an explicitly selected keyless search alternative for public evidence.

#### Scenario: Default provider chain

- **WHEN** `AGENT_WEB_SEARCH_PROVIDER` is unset or `auto`
- **THEN** the existing provider order, credentials and enable flags SHALL apply
- **AND** no Parallel request SHALL be made

#### Scenario: Anonymous Parallel selection

- **WHEN** the optional `parallel` extra is installed and the process selects `parallel`
- **THEN** `search_web` SHALL call `web_search` at `https://search.parallel.ai/mcp`
  using Streamable HTTP with a PaperSage User-Agent and no Authorization header
- **AND** returned text excerpts and source URLs SHALL be available to the agent

#### Scenario: Explicit selection fails

- **WHEN** Parallel returns an error, no results, or exceeds the eight-second deadline
- **THEN** the tool SHALL expose a failure or empty result without secret details
- **AND** it SHALL NOT invoke another search provider

#### Scenario: Runtime cancellation

- **WHEN** the async runtime cancels a Parallel tool call
- **THEN** cancellation SHALL propagate and close its MCP and HTTP contexts

#### Scenario: Disabled or disallowed search

- **WHEN** search is `off`, the provider setting is invalid, or query policy blocks a query
- **THEN** the tool SHALL NOT send a network request

#### Scenario: Evaluation overrides

- **WHEN** a valid query has a configured evaluation override
- **THEN** the existing override SHALL supply the result without a live provider call
