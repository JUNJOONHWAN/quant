# STOCK analysis code

Canonical analysis/research code moved out of `STOCK` for the DGX split.

- `market_analysis/`: ETF RADAR, GoStop, deep-research, report generation.
- `ml_ai_system/`: ML/AI analysis and optimizer modules.

`STOCK` keeps compatibility shim packages so existing execution scripts can
continue importing `market_analysis.*` and `ml_ai_system.*` while the canonical
source now lives in `quant/stock_analysis/`.
