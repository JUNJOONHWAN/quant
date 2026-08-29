# ETF Flow Convergence/Divergence Forecast v6

This is a new numeric forecast engine. It does not fine-tune Qwen and does not
modify the existing AI RADAR, GoStop, mail, Sheets, scheduler, or order paths.
The existing Forecast v2 HGB models remain frozen comparison baselines.

## Point-in-time contract

- decision/trade session: `T`
- stock price and non-flow inputs: through `T-1`
- Massive ETF Flow: effective session exactly `T-2`, visible by `T`
- ETF holdings: only snapshots available by `T-1`
- missing flow is represented by an explicit mask, never silently filled from
  `T-3`

## Architecture

1. Every ETF with at least one PIT-visible Flow report in the prior 60 sessions
   remains an active node, even when its exact T-2 row is missing. Missing T-2
   stays masked and reporting age is retained; it is never filled with T-3.
2. A bias-free temporal Transformer encodes values, feature-specific masks,
   persistence, and reporting age. Masks and age can alter keys and weighting,
   but exact zero Flow still produces an exact zero dynamic value.
3. Learned inducing queries transform the complete ETF set into latent Flow
   factors. A second learned query estimates consensus; factor-to-consensus
   agreement and deviation explicitly form convergence/divergence factors.
   This is not a hand-written ETF mean, top-K, or cross-sectional neutralization.
4. The model keeps two all-ETF paths. The common path learns broad market and
   benchmark Flow pressure. The rotation path lets each stock's identity and
   T-1 price state query convergence/divergence factors. Learned latent
   Flow-price products capture confirmation, lag, opposition, and possible
   5/20-session resolution.
5. A separate linked branch combines sparse graph attention with an explicit
   PIT holding-weighted pool for ETFs that actually hold the stock. A third
   relation branch measures the static holding structure without Flow.
6. A frozen price-only model supplies an out-of-fold baseline. Learned bounded
   gates add relation and Flow residuals to that baseline instead of allowing
   an unconstrained graph residual to dominate it.
7. Auxiliary losses teach the common branch on absolute return/upside/loss and
   the rotation branch on benchmark-relative excess/upside capture/downside
   defense. No target or Flow representation is cross-sectionally centered.
8. Evaluation includes price-only, branch ablations, zero Flow, ETF-axis
   shuffle, five-session Flow lag, and stock-query shuffle controls. It also
   reports learned factor convergence/dispersion and Flow-price convergence vs
   divergence resolution strata. Acceptance requires both common and rotation
   OOS value; a large residual alone is not an edge.

Targets are 5/20-session return, MFE/upside, MAE/loss, benchmark excess return,
benchmark upside capture, and benchmark downside defense. Point, ordered
q10/q50/q90, direction, and within-date ranking losses are trained jointly.

## Existing-container runtime

No new Python package installation is required on DGX. The existing
`nvcr.io/nvidia/nemo-automodel:26.06.00` image provides PyTorch/CUDA and reports
BF16 support on NVIDIA GB10. `torch-geometric` is intentionally not required.
The container runner defaults to a 28 GiB host limit, CUDA allocator fraction
0.15, lazy module loading, and disabled CUDA caching so the protected 27B lane
is not displaced by transient graph-shape allocations.

The 2026-07-15/16 Flow dates were absent from the prior derived cache and the
2026-07-17 date contained only 26 rows. Rebuild the isolated overlay from the
verified 30,000-row Massive historical capture before dataset construction:

```bash
python3 -m training.quant_flow_graph overlay-flow
```

The overlay copies the canonical cache, inserts only missing ticker/effective
keys, and never mutates the canonical base or Oracle increment. Its receipt
labels the source `historical_window_captured`: provider effective/processed
dates reconstruct eligibility, but it is not misrepresented as an as-observed
archive.

Build the sealed one-month/10-stock smoke dataset with the host Python:

```bash
python3 -m training.quant_flow_graph build-smoke
```

Every signal date is compared with the lower decile of the preceding 20
sessions' ETF counts, where each historical count is itself measured only with
rows visible at that date's own `T`. A date below 50% of that robust lower
envelope is listed and excluded. This catches catastrophic source gaps without
mistaking the provider's recurring reporting cadence for missing data. Exact
`T-2` gaps are never filled with `T-3` Flow, and accepted dates retain every
visible ETF—there is no top-K shortcut.

Run a real BF16 gradient smoke inside the existing NVIDIA image:

```bash
bash training/quant_flow_graph/run_in_container.sh train-smoke \
  --dataset-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/smoke_20260629_20260729_10stocks \
  --output-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/training/smoke
```

The smoke receipt is contract/gradient evidence only and is explicitly marked
`SMOKE_ONLY_NOT_PERFORMANCE_EVIDENCE`.

Build the full SPY + Nasdaq-100 point-in-time panel graph after the smoke gate:

```bash
python3 -m training.quant_flow_graph build-dataset \
  --output-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/full_20180102_20260729_allpanel \
  --start-date 2018-01-02 --end-date 2026-07-29
```

Then run expanding walk-forward evaluation. Each outer year has a 20-session
purge; price residual labels are generated by inner expanding OOF folds.

```bash
bash training/quant_flow_graph/run_in_container.sh walk-forward \
  --dataset-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/full_20180102_20260729_allpanel \
  --output-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/training/walk_forward \
  --test-year-start 2021 --test-year-end 2026
```

Each completed outer fold is written with a checkpoint SHA-256. The default
`--resume` mode reuses only a fold whose dataset manifest, training
configuration, model schema, and checkpoint hash all match. `run_state.json`
records the current year and each pretrain/graph epoch so an interrupted
multi-hour run is observable and restartable without trusting a partial
checkpoint.

After a sealed candidate fold completes, run the fixed AI-specificity gate:

```bash
bash training/quant_flow_graph/run_in_container.sh ai-specific-gate \
  --dataset-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/full_20180102_20260729_allpanel \
  --fold-receipt /home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/training/walk_forward_candidate/fold_2026.json \
  --output /home/zooh/Documents/GitHub/STOCKDATA/QUANT_FORECAST/flow_graph_v6/training/walk_forward_candidate/ai_specific_gate.json
```

The thresholds are fixed before reading the candidate: exact zero-Flow
neutrality on 12/12 targets; material common and rotation Flow on 5/6 relevant
targets each; nonnegative mean price-only improvement; actual Flow better than
ETF-shuffled and five-session-lagged Flow on at least 7/12; common Flow helping
at least 4/6 absolute targets; rotation Flow and the correct stock query helping
at least 4/6 benchmark-relative targets. The linked-holdings branch is reported
separately and cannot substitute for either all-ETF branch.

NVFP4 is not a training mode on this GB10 lane. Quantization is a separate,
post-freeze deployment gate and must preserve BF16 walk-forward predictions,
rankings, intervals, and calibration before release.
