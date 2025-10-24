# Market Stability Dashboard Project Plan

## 1. Objectives and Constraints

- **Purpose**: Provide a public, web-based dashboard that tracks correlations between major assets and translates them into a "Market Stability Index" between 0 and 1.
- **Assets**: QQQ, SPY, TLT, GLD, BTC-USD.
- **Data Freshness**: Daily close data refreshed once per day (free data tier).
- **Audience**: Open access; raw market data is not redistributed to comply with licensing.
- **Language/Locale**: Primary audience in Korea; timestamps displayed in KST.

## 2. Data Pipeline

1. **Source**: Yahoo Finance (unofficial) via `yfinance`. Supports historical and daily close data for all assets.
2. **Scheduler**: GitHub Actions cron at `30 21 * * *` (UTC) ≈ 06:30 KST. Triggers the collector once per day.
3. **Collector (`collector/run.py`)**:
   - Download two years of daily OHLCV data for the tracked symbols.
   - Align trading days, remove dates with insufficient overlap, prevent forward filling across holidays.
   - Compute log returns \(r_{a,t} = \ln(P_{a,t} / P_{a,t-1})\).
   - Calculate rolling correlations for 20, 30, and 60 session windows.
   - Derive the Market Stability Index and sub-indexes (see §4).
   - Export JSON artifacts into `public/data/`.
4. **Artifacts**:
   - `current.json`: Latest index snapshot, trend delta, sub-index values.
   - `history_<window>.json`: Time series for selected window (30, 60, 90, 180 day lookbacks).
   - `corr_matrix_<date>_w<window>.json`: Correlation matrix with sample coverage for each lookback window.
   - `meta.json`: Asset metadata, update timestamp, disclaimer text.
5. **Hosting**: GitHub Pages (or Cloudflare Pages) serves `/public`. JSON files cached for 24 hours; bust cache by appending query string `?v=<iso-date>` when fetching.
6. **Monitoring**: GitHub Actions email/Slack notification on workflow failure. "Stale" badge rendered in UI when `asof` is older than 36 hours.

## 3. Repository Structure

```
quant/
├── collector/
│   ├── run.py            # Entry point for the ETL job
│   ├── config.json       # Correlation windows, weights, threshold values
│   ├── utils.py          # Shared helpers (fetch, normalize, compute)
│   └── tests/            # Regression tests using fixture CSV snapshots
├── public/
│   ├── data/             # Generated JSON artifacts served to the frontend
│   └── index.html        # Bundled static site (built by web/)
├── web/
│   ├── package.json
│   ├── src/
│   │   ├── components/   # Gauges, heatmap, correlation charts
│   │   ├── hooks/        # Data fetching, stale detection
│   │   └── pages/        # Next.js pages (or static SPA entrypoint)
│   └── tsconfig.json
├── .github/workflows/
│   └── daily-collector.yml
└── README.md             # High-level documentation
```

## 4. Computation Specifications

### 4.1 Market Stability Index

- Pair weights \(w_{ij}\):
  - Stock–Bond: 2.0 (QQQ/SPY with TLT)
  - Stock–Gold: 1.5 (QQQ/SPY with GLD)
  - Stock–Crypto: 1.0 (QQQ/SPY with BTC-USD)
  - Stock–Stock: 1.0 (QQQ with SPY)
  - Bond–Gold: 1.0 (TLT with GLD)
- Raw index:
  \[
  S(t) = \frac{\sum_{i<j} w_{ij} |\rho_{ij}(t)|}{\sum_{i<j} w_{ij}}
  \]
- Smoothing: apply EMA with span 10 to obtain \(S^*(t)\).
- Trend arrow: difference between EMA spans 3 and 10 on \(S^*(t)\).

### 4.2 Sub-Indexes

- **Stock–Crypto**: Mean absolute correlation for \{QQQ, SPY\} × \{BTC-USD\}.
- **Traditional Assets**: Mean absolute correlation across \{QQQ, SPY, TLT, GLD\}.
- **Safe-Haven Coupling**: Mean of `max(0, -ρ)` for stock pairs vs. TLT/GLD (captures inverse co-movement strength).

### 4.3 Quality Guards

- Minimum valid observations per window: 70%; exclude pairs below threshold.
- Missing data: Drop the date rather than forward-fill across non-trading days.
- Consistency check: Recompute correlations using cached CSV fixtures during tests and ensure absolute error < 1e-6.

## 5. API & Frontend Contract

- The frontend consumes JSON via `fetch('/data/<file>.json?v=<date>')`.
- Gauge components expect the following structure:

```json
{
  "S": 0.447,
  "dS": 0.012,
  "window": 30,
  "sub": {
    "stock_crypto": 0.667,
    "traditional": 0.447,
    "safe_neg": 0.310
  },
  "bands": {
    "red": [0.0, 0.3],
    "yellow": [0.3, 0.4],
    "green": [0.4, 1.0]
  },
  "updated_at_kst": "2025-10-24T06:30:00+09:00"
}
```

- Correlation matrix file contains:
  - `symbols`: ordered list of asset tickers.
  - `matrix`: 2D array of correlations for the specified window.
  - `coverage`: 2D array of valid observation ratios (0–1).

## 6. Frontend Experience

- **Main Gauge**: Semi-circular speedometer from 0 to 1 with red/yellow/green bands. Current value needle plus secondary marker for long-term average.
- **Sub Gauges**: Three compact gauges for the sub-indexes, sharing the same color coding.
- **Time-Series Panel**: Area chart of stability index over the past 6 months with 30-day moving average overlay and shaded stability bands.
- **Correlation Heatmap**: 5×5 matrix (upper triangle) with diverging color palette. Clicking a cell opens the pair detail panel.
- **Pair Detail View**: Dual-axis chart plotting normalized prices and rolling correlation for the selected window.
- **Controls**: Dropdowns for date range (30/60/90/180 days), correlation window (20/30/60), asset toggles. Changes switch the JSON endpoint (no server computation required).
- **Tooltips**: `ⓘ` icons deliver glossary entries (correlation definition, stability theory, data limitations) sourced from `meta.json`.
- **Staleness Warning**: Banner if data older than 36 hours.

## 7. Deployment & Operations

- **CI/CD**: Pull requests run unit and regression tests. Merges to `main` trigger the collector workflow and frontend build.
- **Secrets**: No credentials required for Yahoo Finance. If migrated to paid feeds later, use GitHub Encrypted Secrets.
- **Error Handling**: If daily job fails, retain previous JSON and mark `status` as `stale`.
- **Documentation**: Update `README.md` with setup instructions, data source disclaimer, and frontend preview link.

## 8. Future Enhancements

- Swap daily data with intraday feed (Polygon, IEX, Finnhub) when licensing allows.
- Persist historical data in a lightweight database (DuckDB or SQLite) for faster backfills.
- Add alternative assets (e.g., EEM, DXY) with configurable weights.
- Incorporate alerting (email/webhook) when stability index exits predefined thresholds.

