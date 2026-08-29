# ETF Flow v14 forward avoidance lockbox

This experiment evaluates the frozen v13 adaptive graph state on exactly eleven
signal dates after the v11-v13 event-cube end. It rebuilds the v11 event cube
through 2026-07-29 from a read-only base/incremental/repair union and refuses to
fit any model unless every historical event and daily-state value through
2026-07-14 is identical to the frozen v11 cube.

The primary targets and all pass checks are fixed in
`v14_forward_avoidance_preregistration.json`. All twelve stock targets are
reported. CatBoost capacity, date balancing, the 20-session purge, the capped
residual adapter, and the v13 causal state equations are reused without tuning.

The Massive repair is PIT-reconstructed historical data rather than an
as-observed archive. Eleven dates are also far below the frozen 60-date minimum,
so a preliminary PASS cannot activate deployment, trading, BF16 training, or
NVFP4 conversion.
