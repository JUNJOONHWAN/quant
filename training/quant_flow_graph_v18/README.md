# ETF Flow v18: Orthogonal Diffusion / Divergence

v18 does not refit the v16 ETF graph or add new market data. It consumes only
sealed, hash-verified v16 outer-year OOS predictions.

The absolute global ETF Flow state remains the Drift baseline. The stock-level
graph contribution is candidate minus global-only. Only that Diffusion branch
is residualized within each signal date against price and aggregate/global Flow
prediction structure. A fixed CatBoost residual adapter is then trained on
earlier completed OOS years and tested on the next year after a 20-session
purge.

Primary and common-only, lag-5, ETF-axis-shuffled, and date-shuffled controls
use identical features and estimator capacity. A historical pass is eligible
only for a future prospective shadow lockbox; deployment, trading, BF16 model
training, and NVFP4 conversion are forbidden by the preregistration.

Run preregistration first, record its SHA-256, then provide that exact digest to
the full run with `--expected-prereg-sha`.
