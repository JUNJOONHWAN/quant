# AGENTS Guidelines

- This repository is the canonical home for quant research and market analysis.
  Barchart, thermometer, ETF Flow, ETF RADAR/GoStop analysis, regime reports,
  single-stock deep research, network/timeline research, backtests, and model
  evaluation belong here even when execution data or compatibility shims still
  live in `STOCK`.
- Treat `/home/zooh/Documents/GitHub/quant` as the analysis source of truth.
  Treat `/home/zooh/Documents/GitHub/STOCK` as the trading/runtime platform:
  account access, live orders, KIS execution, daemons, production schedulers,
  and legacy compatibility entrypoints.
- For any market, options, futures, Barchart, thermometer, ETF Flow, ETF RADAR,
  deep-research, or portfolio-timing request, first read
  `knowledge/market_research_knowhow_db.json` when present. If it is missing,
  read `/home/zooh/.codex/knowledge/stock_research_knowhow_db.json` as a
  compatibility fallback and state that the quant-local DB is missing.
- Keep data-source contracts under `data_sources/` and workflow contracts under
  `workflows/`. Do not add new market-analysis operating instructions to the
  STOCK repo unless they are specifically about trading/runtime execution.
- Report data sources as `confirmed`, `partial`, or `failed`. Do not silently
  omit failed Barchart/FMP/Massive/Topstep/KIS/KRX/DART/community checks.
- If source coverage is incomplete, label the result as preliminary and state
  the missing source or stale data explicitly.
- Hermes should enter this project for quant assistant work. Hermes may load
  STOCK-specific skills only when execution/runtime/account actions are needed.
