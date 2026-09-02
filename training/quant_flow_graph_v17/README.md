# ETF Flow v17: prequential confidence and avoidance gate

v17 R2 does not retrain or retune v16. It consumes only sealed v16 outer-year
OOS predictions and asks whether the slow full-ETF state can decide when the
v16 query should replace the v12 forecast, or when a date/target should be
avoided.

For every evaluation year 2022-2026, the meta learner uses earlier OOS years
only. The final 20 sessions of the latest calibration year are purged. The
full-ETF query, price-only, global-only, lag-5, ETF-axis-shuffle, and
date-shuffle candidates all receive identical features and model capacity.

Freeze the preregistration first:

```bash
training/quant_flow_graph_v17/run_in_container.sh --preregister-only
```

Then pass the printed SHA explicitly:

```bash
training/quant_flow_graph_v17/run_in_container.sh \
  --expected-prereg-sha SHA256_FROM_PREVIOUS_COMMAND
```

Exit code 3 is a valid fixed-gate FAIL, not an execution error. A historical
PASS authorizes only a future prospective shadow lockbox, never deployment,
orders, BF16 Set Transformer training, or NVFP4 conversion.

R2 explicitly stores and maps the cumulative v16 `date_codes`; the preserved
R1 preregistration/partial checkpoint stopped before performance evaluation
because it incorrectly treated those codes as zero-based within each year.
