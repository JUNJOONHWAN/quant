# Open-World Market Research MCP

This quant-owned MCP is a card-scoped research coordinator for the Hermes
market Role Shell.

The Role Shell first verifies the user's question by separating claims,
initial hypotheses, falsification questions, and current source coverage. It
requests this MCP from the central MultiTool catalog only when the existing
market data is insufficient, contradictory, or likely to benefit from novel
hypothesis discovery.

Tools:

- `market_research_health`
- `market_research_start`
- `market_research_search`
- `market_research_add_evidence`
- `market_research_evaluate`
- `market_research_export`

Search results are always stored as `SEARCH_LEAD`, never as confirmed market
evidence. The market Role Shell must verify a lead through an official,
structured, browser-verified, or otherwise independent source and add that
result with its actual lifecycle status.

Default state is stored at:

`/home/zooh/Documents/GitHub/STOCKDATA/OPEN_WORLD_MARKET_RESEARCH/research.sqlite3`

Override it for tests with `OPEN_WORLD_MARKET_RESEARCH_DB_PATH`.

The server has no order, account, scheduler, or daemon mutation capability.
