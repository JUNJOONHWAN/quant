# ETF Flow v13 adaptive graph-state canary

This experiment keeps the proven v12 price-preserving CatBoost residual and
tests a causal state-space memory layer over the audited v11 PIT graph signals.
It is intentionally smaller than a Transformer or a learned GNN.

The primary model receives current Flow plus structure and a forward-only state
for market Drift, direct ETF-to-stock pressure, and indirect all-ETF cluster
Diffusion. Market Flow shocks increase the observation gain so stale history is
forgotten faster; stale source coverage reduces the gain. Fixed-memory,
five-session-lag, and topology-shuffle controls use the same model capacity.

A historical PASS is not deployment proof. It only permits the next learned
BF16 sparse graph-SSM canary and a genuinely new forward lockbox. NVFP4 remains
a deployment conversion target, not a training shortcut.
