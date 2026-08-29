# ETF Flow v11-R2 Phase A

This module leaves v6 and every live service untouched. It builds new,
read-only-derived research artifacts under `flow_graph_v11/r2_phase_a`.

The fixed timing contract is signal `T`, price/liquidity `T-1`, and exact
Massive ETF Flow `T-2` visible by `T`. A missing exact row is never replaced by
`T-3`. Exact zero, missing, stale, and lifecycle states remain distinct.

The ETF RADAR floors are hard exclusions from the modeling denominator:

- assets at least USD 50 million
- T-1 price at least USD 3
- T-1 dollar volume at least USD 1 million

Absolute common Flow is preserved. It is never date-centered. The
selection-biased `48_MASSIVE_ACCUM_CLUSTER.flow_breadth` field is forbidden.
Independent breadth is computed from benchmark/lineage and PIT-holdings
overlap families, while typed special products remain a separate observation
channel.

Run in the already-installed NVIDIA image:

```bash
bash training/quant_flow_graph_v11_r2/run_in_container.sh --replace
```

Run the fixed interpretable market tournament only after Phase A passes:

```bash
bash training/quant_flow_graph_v11_r2/run_in_container.sh phase-b-market
```

Run the full mapped cluster rotation tournament regardless of whether the broad
market path survives:

```bash
bash training/quant_flow_graph_v11_r2/run_in_container.sh phase-b-cluster
```

Phase A can activate only the interpretable Phase B tournament. It cannot
activate the graph model or NVFP4 deployment.
### Phase B stock-level all-ETF propagation

```bash
bash training/quant_flow_graph_v11_r2/run_in_container.sh phase-b-stock
```

This fixed tournament uses every eligible ETF in the global Drift and in each
stock's cluster-mediated Diffusion state, while keeping direct PIT constituent
pressure separate.  It evaluates all 12 five- and twenty-session stock targets
with expanding 2021-2026 OOS folds, a 20-session purge, an equal-width nonlinear
price comparator, 5/20-session Flow lags, and a within-date topology shuffle.
Absolute common Flow is never date-centred and Table 48 breadth is never used.
