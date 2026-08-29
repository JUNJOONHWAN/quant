# ETF Flow v12 residual canary

This is a separate successor experiment. It does not reinterpret or overwrite
the v11-R2 Phase B FAIL receipts.

The canary asks one narrow but full-scope question: after balancing every
market date equally, can a fixed-capacity nonlinear model use the existing
all-ETF Drift, direct PIT holdings, and indirect Diffusion features to improve
all twelve stock targets without damaging the price-only baseline?

The fixed timing remains signal `T`, price `T-1`, and exact ETF Flow `T-2`.
Every 2021-2026 outer fold uses a 20-session purge. The same CatBoost capacity
is used for price-only, current Flow, 5/20-session lagged Flow, and a within-date
topology shuffle. The primary prediction is a pre-fixed 25% capped residual
adapter from the price-only prediction toward the full model prediction.

Freeze the preregistration before looking at results:

```bash
bash training/quant_flow_graph_v12/run_in_container.sh --preregister-only
```

Then run or resume the full canary:

```bash
bash training/quant_flow_graph_v12/run_in_container.sh
```

Exit code `3` is a valid predictive FAIL verdict. The run is CPU-only and does
not stop or reconfigure any live model or service. Historical OOS years already
influenced the architecture decision, so even a PASS is exploratory and still
requires a new future lockbox before deployment or trading use.
