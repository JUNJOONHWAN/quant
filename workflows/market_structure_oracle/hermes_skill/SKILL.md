---
name: quant-market-analysis
description: Use when Hermes handles market analysis, quant research, Barchart, thermometer, ETF Flow, ETF RADAR/GoStop analysis, regime reports, deep research, network/timeline research, options/futures context, or portfolio-timing briefs. This skill routes analysis to the quant project and keeps STOCK limited to trading/runtime execution.
category: data-science
---

# Quant Market Analysis

## Canonical Routing

Use `/home/zooh/Documents/GitHub/quant` as the working project for market
analysis and quant research.

Use `/home/zooh/Documents/GitHub/STOCK` only for trading/runtime execution:
account access, order placement, KIS execution, daemon control, production
schedulers, and legacy compatibility scripts.

Do not treat STOCK as the canonical analysis source when equivalent analysis
logic exists in quant.

## Required Preflight

Before any market conclusion, read:

1. `/home/zooh/Documents/GitHub/quant/AGENTS.md`
2. `/home/zooh/Documents/GitHub/quant/knowledge/market_research_knowhow_db.json`
3. If the quant-local DB is missing, use `/home/zooh/.codex/knowledge/stock_research_knowhow_db.json` as a compatibility fallback and say so.

For live market data, also load `stock-research-skills` as the data-source
access reference. That skill is not the analysis owner; it documents access
patterns for Barchart, FMP, Massive, Topstep, KIS/KRX/DART, and community data.

## Domain Map

- `data_sources/barchart/`: Barchart Premier/CDP page contracts, option pages,
  IV, gamma, max pain, put/call, time and sales.
- `data_sources/fmp/`: FMP quote, profile, analyst, financials, news, calendar.
- `data_sources/massive/`: Massive market data, ETF fund flows, SEC/news where
  available through configured access.
- `data_sources/topstep/`: Futures and macro pulse context.
- `data_sources/kis_krx_dart/`: Korea market, KIS, KRX, DART/OpenDART context.
- `data_sources/community/`: Naver Cafe, Reddit, and mood scorecards.
- `workflows/thermometer/`: Korea/US/full market thermometer process.
- `workflows/etf_flow/`: ETF Flow report process.
- `workflows/etf_radar/`: ETF RADAR and GoStop analysis logic.
- `workflows/regime/`: market stability and probabilistic regime reports.
- `workflows/deep_research/`: single-stock deep research.
- `workflows/network_timeline/`: network dominance and timeline reports.

## Output Contract

Start with the conclusion. Then separate:

- source status: confirmed / partial / failed
- key numbers and levels
- interpretation
- limitations
- next action or verification needed

If required data is missing, stale, or partially blocked, label the answer
preliminary. Never hide source failures.

## Safety Boundary

Do not place trades, change live daemons, or touch account/runtime state from
this skill. Escalate to the STOCK runtime workflow only when the user explicitly
requests execution or account-affecting work.
