# Market Stability Dashboard Project Plan

## 1. Objectives and Constraints

- **Purpose**: Provide a public, web-based dashboard that tracks correlations between major assets and translates them into a "Market Stability Index" between 0 and 1.
- **Assets**: QQQ (backtest/benchmark), IWM, SPY, TLT, GLD, BTC-USD (signal universe).
- **Data Freshness**: Daily open/close data refreshed once per day (free data tier).
- **Audience**: Open access; raw market data is not redistributed to comply with licensing.
- **Language/Locale**: Primary audience in Korea; timestamps displayed in KST.

## 2. Data Pipeline

1. **Source**: Financial Modeling Prep daily endpoints. ETFs/ETNs and BTCUSD use the `historical-price-full` API. Access requires the `FMP_API_KEY` secret.
2. **Generator (`scripts/generate-data.js`)**: Node.js (18+) script that pulls five years of adjusted opens and closes, aligns trading days, computes log returns, and derives rolling correlations for 20/30/60 session windows before emitting a consolidated JSON payload.
3. **Trigger**: `.github/workflows/deploy.yml` executes the generator on every push to `main` (and on manual dispatch). Add a cron trigger if an automatic daily refresh is desired.
4. **Artifact**: `static_site/data/precomputed.json` containing windowed records (stability, smoothed stability, deltas, correlation matrices, sub-indexes) plus pair-level price/correlation series, **both close and open price series** (for T+1 execution modelling), and metadata (generation timestamp, asset catalog).
5. **Hosting**: The same workflow uploads `static_site/`—including the freshly generated JSON—to GitHub Pages via the official Actions integration.
6. **Monitoring**: Workflow failures surface in the GitHub Actions UI; the frontend shows a descriptive error if the JSON is missing or stale.

## 3. Repository Structure

```
quant/
├── .github/workflows/
│   └── deploy.yml              # Builds data + deploys static_site/ to GitHub Pages
├── docs/
│   └── market_stability_app_plan.md
├── scripts/
│   └── generate-data.js        # FMP fetcher + precomputation logic
├── static_site/
│   ├── assets/                 # Browser JavaScript, CSS, shared metric helpers
│   ├── data/                   # Holds precomputed.json generated during deployment
│   └── index.html              # Dashboard entry point served by Pages
├── tests/
│   └── metrics.test.js         # Node-based regression tests for metric helpers
└── README.md                   # Setup, deployment, and data source docs
```

## 4. Computation Specifications

### 4.1 Market Stability Index

- Pair weights \(w_{ij}\):
  - Stock–Bond: 2.0 (IWM/SPY with TLT)
  - Stock–Gold: 1.5 (IWM/SPY with GLD)
  - Stock–Crypto: 1.0 (IWM/SPY with BTC-USD)
  - Stock–Stock: 1.0 (IWM with SPY)
  - Bond–Gold: 1.0 (TLT with GLD)
- Raw index:
  \[
  S(t) = \frac{\sum_{i<j} w_{ij} |\rho_{ij}(t)|}{\sum_{i<j} w_{ij}}
  \]
- Smoothing: apply EMA with span 10 to obtain \(S^*(t)\).
- Trend arrow: difference between EMA spans 3 and 10 on \(S^*(t)\).

### 4.2 Sub-Indexes

- **Stock–Crypto**: Mean absolute correlation for \{IWM, SPY\} × \{BTC-USD\}.
- **Traditional Assets**: Mean absolute correlation across \{IWM, SPY, TLT, GLD\}.
- **Safe-Haven Coupling**: Mean of `max(0, -ρ)` for stock pairs vs. TLT/GLD (captures inverse co-movement strength).

### 4.3 Quality Guards

- Minimum valid observations per window: 70%; exclude pairs below threshold.
- Missing data: Drop the date rather than forward-fill across non-trading days.
- Consistency check: Recompute correlations using cached CSV fixtures during tests and ensure absolute error < 1e-6.

## 5. API & Frontend Contract

- The frontend downloads a single payload: `static_site/data/precomputed.json`.
- Top-level fields:
  - `generatedAt`: ISO timestamp (UTC) when the dataset was produced.
  - `analysisDates`: Array of ISO dates aligned with the rolling return series.
  - `normalizedPrices`: Map from symbol to normalized price index (aligned with `analysisDates`).
  - `assets`: Symbol metadata consumed by tooltips and selectors.
  - `priceSeries`: Map of adjusted closes by symbol.
  - `priceSeriesOpen`: Map of adjusted opens by symbol (used for T+1 open execution in backtests).
  - `windows`: Object keyed by window length (`"20"`, `"30"`, `"60"`). Each entry contains:
    - `records`: Array ordered by date with `stability`, `smoothed`, `delta`, `sub` (sub-index breakdown), and `matrix` (5×5 correlation matrix).
    - `average180`: Mean stability over the trailing 180 trading days.
    - `latest`: Convenience copy of the last record.
    - `pairs`: Map of `<symbolA>|<symbolB>` to aligned `dates`, `correlation`, `priceA`, `priceB` arrays for the pair detail chart.
- Downstream backtests use `priceSeriesOpen` to apply regimes with a one-day execution lag (signals based on close, trades at next session's open) while neutrals track close-to-close benchmark returns.

```json
{
  "generatedAt": "2025-10-24T08:30:17.122Z",
  "analysisDates": ["2024-01-02", "2024-01-03", "…"],
  "normalizedPrices": {
    "QQQ": [1.0, 1.003, "…"],
    "IWM": [0.99, 1.01, "…"],
    "SPY": [1.0, 0.998, "…"]
  },
  "assets": [
    { "symbol": "QQQ", "label": "QQQ (NASDAQ 100 ETF)", "category": "stock" },
    { "symbol": "IWM", "label": "IWM (Russell 2000 ETF)", "category": "stock" }
  ],
  "windows": {
    "30": {
      "average180": 0.412,
      "latest": { "date": "2025-10-23", "stability": 0.447, "delta": 0.012, "sub": { "stockCrypto": 0.667, "traditional": 0.447, "safeNegative": 0.310 } },
      "records": [
        { "date": "2024-02-15", "stability": 0.352, "smoothed": 0.344, "delta": -0.008, "matrix": [[1, 0.89, "…"]], "sub": { "stockCrypto": 0.51, "traditional": 0.44, "safeNegative": 0.28 } }
      ],
      "pairs": {
        "IWM|BTC-USD": { "dates": ["2024-02-15", "…"], "correlation": [0.42, "…"], "priceA": [1.0, "…"], "priceB": [1.0, "…"] },
        "QQQ|SPY": { "dates": ["2024-02-15", "…"], "correlation": [0.94, "…"], "priceA": [1.0, "…"], "priceB": [1.0, "…"] }
      }
    }
  }
}
```

## 6. Frontend Experience

- **Main Gauge**: Semi-circular speedometer from 0 to 1 with red/yellow/green bands. Current value needle plus secondary marker for long-term average.
- **Sub Gauges**: Three compact gauges for the sub-indexes, sharing the same color coding.
- **Time-Series Panel**: Area chart of stability index over the past 6 months with 30-day moving average overlay and shaded stability bands.
- **Correlation Heatmap**: 5×5 matrix (upper triangle) with diverging color palette. Clicking a cell opens the pair detail panel.
- **Pair Detail View**: Dual-axis chart plotting normalized prices and rolling correlation for the selected window.
- **Controls**: Dropdowns for date range (30/60/90/180 days), correlation window (20/30/60), asset toggles. Changes switch the JSON endpoint (no server computation required).
- **Tooltips**: `ⓘ` icons deliver glossary entries (correlation definition, stability theory, data limitations) embedded directly in the frontend copy.
- **Staleness Warning**: Banner if data older than 36 hours.

## 7. Deployment & Operations

- **CI/CD**: Pull requests run unit tests (`npm test`). Pushes to `main` execute `deploy.yml`, which generates `precomputed.json` and deploys Pages.
- **Secrets**: `FMP_API_KEY` is mandatory. Store it as a repository secret so the workflow can read it. Future paid feeds can reuse the same mechanism.
- **Error Handling**: Workflow failure leaves the previous deployment intact; the frontend displays a "데이터를 생성하세요" message if the JSON is missing.
- **Documentation**: README now covers key provisioning, local previews, and the FMP data disclaimer.

## 8. Future Enhancements

- Swap daily data with intraday feed (Polygon, IEX, Finnhub) when licensing allows.
- Persist historical data in a lightweight database (DuckDB or SQLite) for faster backfills.
- Add alternative assets (e.g., EEM, DXY) with configurable weights.
- Incorporate alerting (email/webhook) when stability index exits predefined thresholds.
