# Quant Qwen3-8B training plan and final preflight contract

## Objective

Build an evidence-bounded quant **analysis** model that explains what was known
at an as-of U.S. trading session, how ETF capital flow propagates into member
stocks, where price and flow confirm or diverge, and what remains unknown. It
does not execute trades, predict a guaranteed return, or receive future outcome
data in SFT prompts or answers.

The numerical engine remains deterministic Python. The LLM learns structured
interpretation and source-aware explanation. The required response keys are
`facts`, `interpretation`, `counter_evidence`, `unknowns`, `regime`,
`confidence`, and `conclusion`.

## Candidate universe

The candidate corpus is not restricted to ETF constituents.

1. `all_stock_control_analysis`: every eligible FMP symbol-session pair in the
   collected daily universe, including historically traded securities before
   delisting. This is the no-ETF-signal control group.
2. `etf_own_flow_analysis`: an ETF's own price, liquidity, Massive fund flow,
   NAV, shares outstanding, and PIT FMP constituent snapshot.
3. `stock_constituent_flow_analysis`: a stock plus every PIT-eligible ETF that
   held it, with each ETF's visible flow allocated by the then-visible holding
   weight.

Present-day active/ETF lists are forbidden as historical filters. A delisted
security remains in its genuine pre-delisting history and naturally stops when
same-session price/volume disappears.

## Source contract

| Source | v1 role | Availability gate | Status |
|---|---|---|---|
| FMP daily OHLCV | full collected price backbone | trade date; historical backfill flagged non-as-observed | included |
| Massive ETF Global fund flow | fund flow, NAV, shares | later of effective+2 U.S. sessions and processed+1 session | included |
| FMP historical ETF holdings | ETF weights and stock memberships | first U.S. session strictly after acceptance date | included |
| FMP extended training features | fundamentals/news/ownership/etc. | endpoint-specific | excluded until paused backfill is complete |
| Massive ETF profiles | leverage/type/AUM metadata | effective/processed dates | endpoint returned HTTP 403; not silently used |

The U.S. session calendar is positive-volume SPY/QQQ daily observations. A few
weekend or corrupt instrument rows cannot advance any delay clock.

## Point-in-time flow-to-stock attribution

For stock `s`, ETF `e`, and packet date `t`:

```text
eligible_membership(e,s,t) =
  FMP acceptance date's next U.S. session <= t

eligible_flow(e,t) =
  max(second session after effective date,
      first session after processed date) <= t

allocated_flow(e,s,t) =
  fund_flow(e,t) * holding_weight(e,s,t) / 100
```

Multiple positions for the same stock inside one ETF snapshot have their
weights summed before the ETF flow is applied once. ETF snapshots qualify as a
direct-equity proxy only when they contain at least five positions, at least 80%
ticker resolution, 70-130% positive ticker weight, and no negative weights.
This conservatively blocks many inverse, derivative, bond, malformed, and
incomplete funds while Massive profile metadata is unavailable.

The fund-flow endpoint does not reliably declare currency. Raw allocated values
are therefore labeled provider-reported units, not USD. Raw amounts are summed
only if every contributing ETF has the same explicit non-null currency.
Currency-free ratios use `fund_flow / (NAV * shares_outstanding)` when both NAV
and shares exist.

## Preprocessing gates

All gates use only data visible at the packet's `as_of_date`.

- Packet schema must be `quant.analysis_packet.v3`; v1/v2 are rejected.
- Same-session price and positive volume are mandatory.
- At least five trailing sessions are mandatory for every security.
- Packets retain 21 price sessions so a genuine 20-session return can be
  computed; liquidity and ETF Flow normalizers use their trailing 20 rows.
- ETF-specific gate: at least 10 of 20 sessions, at least 75% nonzero-volume
  sessions, and at least USD 1 million median dollar volume.
- Price features use raw close. Retrospectively adjusted absolute prices are
  not exposed because later corporate actions can rewrite history.
- A raw one-session price discontinuity of 45% or more is excluded until a PIT
  split/corporate-action feed explains it.
- ETF flow versions dedupe by `(ticker, effective_date)` to the latest revision
  visible as-of.
- SFT packet/example duplicates are removed with an on-disk unique index.
- Flow normalization uses only visible trailing median/MAD and flow divided by
  estimated net assets; no full-period scaler or global future statistic.
- Extreme robust z-score magnitude above 8 is flagged, not silently clipped.
- Mixed/unknown currencies are not aggregated as dollars.
- Every exclusion reason and count is persisted in the manifest.
- Full raw packets retain every provenance pointer, while the SFT prompt uses a
  bounded primary-task evidence view plus a digest of the complete provenance
  set. This contract is versioned as `quant.qwen3_8b_sft_contract.v2`.
- Training-time truncation is disabled. Every selected train/validation row is
  rendered with the pinned Qwen tokenizer and must fit 4096 tokens with a
  non-empty assistant target; otherwise the run fails before GPU allocation.

## Corpus, training view, and time split

The complete source corpus remains in the immutable normalized/raw database.
Rich per-stock packets are not duplicated for all roughly 24 million
symbol-sessions because large-cap membership/provenance packets can exceed 2 MB
per date. Instead, a first pass scans every FMP symbol-session and keeps the
lowest salted pair hashes within each historical task proxy and time split.
Selected pairs are then rebuilt as complete v3 packets in memory, passed through
exact PIT/quality/task gates, reduced to compact SFT rows, and recorded in a
resumable SQLite materialization ledger. Full scan counts, proxy-to-actual task
transitions, exclusions, selected counts, and hashes remain in manifests. No
selected reservoir is claimed to represent the full database.

The pair reservoir is deliberately larger than the training view. The first
40,000 ETF-own proxy pass produced 16,621 exact ETF-own train rows, below the
fixed 20,000 target, so the reservoir was expanded without shrinking the target:
60,000 ETF-own, 50,000 stock constituent-flow, and 30,000 controls for train;
5,000 per task for validation; and 10,000 per task for the sealed test. Exact
reclassification can reduce these counts; the balanced training view is built
only after the materialization manifest proves sufficient actual-task coverage.

Candidate time split:

- train: 2017 through 2023, excluding the final 20 U.S. sessions;
- validation: 2024, excluding the final 20 U.S. sessions;
- sealed test: 2025 through the latest complete packet;
- no random cross-time mixing.

The selected candidate reservoir and the actual training view are separate
artifacts. The initial training view is selected deterministically within each
time split and actual task type:

- 20,000 ETF-own-flow examples;
- 25,000 stock constituent-flow examples;
- 15,000 all-stock controls;
- 512 validation examples per task.

These are explicit starting quotas, not a claim that a subset represents the
full database. Candidate counts, selected counts, dropped counts, hashes, and
the untouched sealed-test reference are in the balanced manifest. Quotas may be
changed only through a new versioned manifest.

Production candidate materialization is restartable from its SQLite ledger and
runs independently of GPU training.  Hourly progress intent is KST minute 05,
stored in an explicitly `Asia/Seoul` systemd calendar.  Each snapshot records
both UTC and KST check/ETA values. Completion requires the final candidate
manifest and split hashes, not only the service exit code or ledger percentage.

## Model and optimization

- Base checkpoint: `Qwen/Qwen3-8B` BF16.
- Framework: NVIDIA NeMo AutoModel image
  `nvcr.io/nvidia/nemo-automodel:26.06.00`.
- Method: answer-only BF16 LoRA SFT; `*_proj`, rank 8, alpha 32.
- Spark topology: one process, FSDP2, TP=1, CP=1, no pipeline parallelism.
- Full context: 4096; local batch 1; global batch 8; two epochs.
- Optimizer: Adam, LR `1e-5`, cosine decay to `1e-6`.
- Checkpoint every 500 steps; validation every 100; consolidated safetensors
  final adapter/checkpoint.
- Prompt ends in `/no_think`; chain-of-thought is not a target. Deployment must
  use `chat_template_kwargs={"enable_thinking": false}`.

Rank/context/learning-rate sweeps happen only after the same candidate version,
balanced manifest, and sealed test are frozen. Nemotron 30B is a later
comparison, not the first Spark training run.

## Evaluation contract

Training cannot start unless all data gates below are zero-failure:

- flow or constituent availability later than packet date;
- weekend junk date advancing a session gate;
- present-day active status used to remove historical rows;
- future/forward/target fields in SFT input or answer;
- duplicate example IDs across splits;
- split overlap or 20-session embargo violation;
- manifest row count or SHA mismatch;
- unknown task type or missing source policy;
- test file referenced by a training dataloader.

Model acceptance compares base Qwen and LoRA on the frozen validation/test
artifacts:

- JSON schema validity and required-key completeness;
- exact extraction of dates, signs, weights, and source status;
- ETF-to-stock allocation arithmetic error;
- regime and confirmation/divergence macro-F1 against deterministic targets;
- counter-evidence and unknown-field recall;
- hallucinated future facts, currency, or trade directives (target: zero);
- human review of a separately versioned expert-label set before calling the
  model an expert quant model.

Forward returns may exist only in a separately sealed offline evaluation table
after the packet is frozen. They never enter SFT prompts, answers, scalers, task
selection, or model checkpoints.

## Runtime and service safety

The preparation step may pull the image and base checkpoint while Qwen/Gemma
services remain running. Training runners never stop services. The read-only
gate requires image and model completeness plus at least 36,000 MiB free unified
memory for smoke or 48,000 MiB for full training. Any service stop is a separate
explicit operator decision.

Training order is fixed:

1. preflight and unit tests;
2. v3 packet shards and packet manifest;
3. candidate corpus build and validation;
4. balanced view build and structural/hash validation;
5. full-row tokenizer/context-length audit inside the pinned image, with no
   truncation allowed;
6. two-step smoke run and checkpoint reload;
7. held-out smoke evaluation;
8. full run only after the smoke report is green.
