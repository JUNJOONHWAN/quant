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

## Market Structure Oracle

When the user says `오라클` or `Oracle`, use the managed
`market-structure-oracle` application. Do not run its Python entrypoint
directly.

- Whole market: `hermes apps run market-structure-oracle --json`
- Named sector/theme: resolve it to the canonical scope below and pass a JSON
  request with `query` and `scope`.
- Arbitrary ETF basket: pass `query`, a stable `scope`, and `etfs`.

Example request:

```bash
hermes apps run market-structure-oracle \
  --input-json '{"query":"반도체 섹터를 오라클로 분석해줘","scope":"semiconductors"}' \
  --json
```

Canonical scopes include `technology`, `semiconductors`,
`communication_services`, `consumer_discretionary`, `consumer_staples`,
`energy`, `financials`, `healthcare`, `industrials`, `materials`,
`real_estate`, `utilities`, `biotechnology`, `cybersecurity`,
`clean_energy`, `defense`, `regional_banks`, `homebuilders`,
`gold_miners`, and `uranium`.

The app computes or reuses one full-market state cube, then conditions the
scope view on the full market. Never describe a scope result as an isolated
sector-only run. The governing formula is:

`scope future | full market state + scope internal structure + relative position + ETF Flow D+2`

Read the successful App Manager receipt and use the report paths in its stdout.
If the receipt is not `PASS`, do not claim that the Oracle completed.

## Quant AI Radar

When the user says `AI Radar`, `Quant AI Radar`, or requests analysis from the
trained quant model, use the managed `quant-ai-radar` application. Do not run
its Python modules directly.

- Full daily reference publish:
  `hermes apps run quant-ai-radar --json`
- Full non-publishing shadow:
  `hermes apps run quant-ai-radar --input-json '{"action":"daily","shadow":true}' --json`
- Explicit symbols:
  `hermes apps run quant-ai-radar --input-json '{"action":"analyze","symbols":["AAPL","NVDA"]}' --json`
- Runtime status:
  `hermes apps run quant-ai-radar --input-json '{"action":"status"}' --json`

The daily 64 ETF / 192 stock values are maximum inference capacities, not fixed
universes. An explicitly requested symbol bypasses that daily selection but
must still pass the same point-in-time packet, Oracle binding, model release,
and response-contract gates. AI Radar is analysis-only and must never be
described as a trading or order-execution application.

Market Structure Oracle is the single writer for the shared current-market
database. Both apps may request preparation, but the interprocess lock permits
only one Massive/FMP incremental capture and the other caller reuses the sealed
snapshot. ETF RADAR remains a separate workflow and is not an Oracle/AI Radar
source, release-date gate, universe gate, or inference input. New securities
enter through Massive grouped daily; new ETFs enter through Massive flow/FMP
profile discovery and receive missing-constituent refresh priority.

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
