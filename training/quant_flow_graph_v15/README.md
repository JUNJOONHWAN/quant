# ETF Flow v15 constituent refresh

This package audits the exact ETF-to-stock relationships used by the sealed v14
forward lockbox and creates a separate FMP Ultimate overlay for missing
historical constituent snapshots.

The workflow is intentionally split:

1. `audit` extracts every ETF connected to a stock on the eleven v14 test dates
   and retains only ETFs passing the ETF RADAR strict point-in-time eligibility
   mask on at least one date.
2. `refresh --phase discover` downloads FMP disclosure-date lists only and
   computes the exact missing `(ETF, effective_date)` keys against the sealed
   base database.
3. `refresh --phase download` downloads only those missing snapshot payloads.
4. `bulk` paginates the FMP Ultimate `stable/etf-holder-bulk` endpoint, stores
   immutable raw parts, and normalizes the current holdings capture.  This is a
   current/future topology source and is never used to impute historical folds.
5. `combine` makes a SQLite-consistent copy of the Oracle incremental database
   and overlays complete historical constituent snapshots without modifying the
   source database.
6. `compare-graphs` compares every test snapshot by stable stock/ETF pair and
   verifies prices, targets, masks, ETF Flow histories, and non-topology arrays.
7. `audit-current-bulk` measures current bulk coverage and topology turnover for
   the exact connected ETF and stock universe.

`topology_sensitivity.py build-graph` creates a full-history control graph whose
training snapshots and ETF Flow cube are byte-identical to v14; only repaired
test-date constituent edges are substituted.  The corresponding `run` command
reuses the v14 split, purge, CatBoost parameters, seed, and controls without
retuning.  Use `run_topology_sensitivity_in_container.sh` so the isolated v12
CatBoost dependency directory and resource limits match the v14 runtime.

The overlay is post-hoc research data. It must never overwrite the authority
database or change the sealed v14 FAIL verdict. Historical captures made today
are not true as-observed data and can only support data-quality diagnosis,
repaired-topology sensitivity, and future pipeline repair.
