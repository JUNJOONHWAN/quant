# ETF Flow v16: full-ETF identity latent gate

This is a new, isolated research path. It does not modify v11-v15 outputs.

The v11 stock model reduced indirect ETF information to 44 cluster-family
states. v16 retains every strictly eligible ETF identity until a target-free,
outer-train-only 32-component TruncatedSVD. Each stock's point-in-time ETF
holding exposure is combined with the full-universe factor states to measure
propagation, alignment, convergence, and divergence.

The estimator and residual cap are identical to v12. Sealed v12 price-only and
current aggregate-Flow predictions are exact-identity controls. Global-only,
5-session lag, ETF-axis shuffle, and date-block shuffle receive equal capacity.

Run preregistration first:

```bash
training/quant_flow_graph_v16/run_in_container.sh --preregister-only
```

Then run without `--replace`. Exit code 3 is a valid FAIL verdict.
