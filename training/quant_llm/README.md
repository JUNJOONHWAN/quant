# Qwen3-8B quant-analysis training lane

This lane fine-tunes `Qwen/Qwen3-8B` with BF16 LoRA in NVIDIA NeMo
AutoModel. It is an analysis model: deterministic code computes numerical
features, and the model learns to organize as-of evidence into facts,
interpretation, counter-evidence, unknowns, regime, confidence, and conclusion.
It does not learn order execution or receive future returns in its prompt or SFT
answer.

## Leakage contract

Only `quant.analysis_packet.v3` is accepted. Massive ETF Flow must carry
`massive_etf_flow_us_sessions_v1`: visibility begins at the later of the second
U.S. trading session after `effective_date` and the first session after
`processed_date`. Missing future calendar rows fail closed. Old v1 packets are
rejected.

The lag calendar uses positive-volume SPY/QQQ sessions, not every distinct date
in the raw daily table. This prevents isolated weekend rows from thin ETFs,
warrants, or source defects from advancing D+1/D+2.

ETF rows also pass `asof_etf_liquidity_20s_v1`: the packet must have a same-day
positive-volume price, at least 10 observed sessions in the trailing 20,
nonzero volume on at least 75% of them, and median trailing dollar volume of at
least USD 1 million. These thresholds are command-line parameters and the exact
values plus every exclusion count are written to the dataset manifest.

Flow revisions are reduced by `(ticker, effective_date)` to the latest revision
that was visible as of the packet. Scale features use flow/AUM and trailing
median/MAD computed only from already-visible observations; raw values stay in
evidence and extreme robust z-scores are flagged instead of silently clipped.
Exact duplicate packets/examples are removed with an on-disk uniqueness index.

Delisted ETFs are not erased from their historical trading period. A current
active-ETF list is forbidden as a historical filter because that creates
survivorship bias. The same-session price/volume gate naturally stops samples
after trading ends; any future delisting-event feed must itself carry an
`available_date <= as_of_date` contract.

Price features use raw close. Retrospectively adjusted absolute prices are not
shown to the model because a later split can rewrite earlier levels. Every
security requires a same-session positive volume and at least five trailing
sessions; a raw one-session discontinuity of 45% or more is excluded until a
separate point-in-time corporate-action event proves how it should be adjusted.
Each packet retains 21 price sessions so the 20-session return has 21 closes;
liquidity and ETF Flow normalization remain trailing-20 calculations.

The default split is train through 2023, validation in 2024, and test from 2025
through the latest packet. The final 20 observed trading sessions before each
later split are purged from the earlier split. No random time mixing occurs.

## Model and framework

- base: `Qwen/Qwen3-8B`, downloaded separately to
  `/home/zooh/models/Qwen3-8B-bf16`;
- image: `nvcr.io/nvidia/nemo-automodel:26.06.00`;
- method: LoRA on `*_proj`, rank 8, alpha 32, BF16, answer-only masked CE;
- Spark process count: one; FSDP2, TP=1, CP=1, no pipeline parallelism;
- full context: 4096 tokens; local batch 1, global batch 8, two epochs;
- smoke: the same 4096-token contract, two optimizer steps.

## Preparation

```bash
cd /home/zooh/Documents/GitHub/quant
bash training/quant_llm/prepare_spark.sh
```

The preparation script pulls the pinned container, downloads the BF16 base
checkpoint, and verifies CUDA imports. It does not stop Qwen/Gemma services and
does not begin training.

## Build the full-universe-scanned time-split corpus

Do not duplicate rich packets for every one of the roughly 24 million collected
symbol-sessions: large-cap membership/provenance packets can exceed 2 MB each.
The production path scans the complete FMP daily table, selects the lowest
salted hashes per historical task proxy, then constructs and validates each
selected v3 packet in memory. The immutable raw database is not reduced or
deleted. Selection counts, proxy-to-actual task changes, exact exclusions, and
hashes are persisted.

```bash
python3 -m training.quant_llm.select_training_pairs \
  --database /home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET/normalized/daily_observations.sqlite3 \
  --output-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/pair_selections/qwen3_8b_pairs_v3 \
  --train-quotas 'etf_own_flow_analysis=60000,stock_constituent_flow_analysis=50000,all_stock_control_analysis=30000' \
  --from 2017-01-01 --to 2026-07-14

python3 -m training.quant_llm.build_selected_sft_from_db \
  --pair-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/pair_selections/qwen3_8b_pairs_v3 \
  --data-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_DATASET \
  --output-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_candidate_v3 \
  --seed-output-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_candidate_v2

python3 -m training.quant_llm.build_balanced_training_set \
  /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_candidate_v3 \
  /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_sft_v2 \
  --replace

python3 -m training.quant_llm.validate_sft_dataset \
  /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_sft_v2
```

The production materializer is installed as a resumable user service. Candidate
v3 uses `quant-qwen3-dataset-materialize-v3.service`: it proves that all 165,000
sealed v2 pairs are unchanged before seeding the completed v2 ledger, so only
the 20,000 additional ETF-own proxy pairs are newly materialized. An explicitly
KST-zoned timer writes an hourly status snapshot at minute 05, including exact
processed/remaining counts, task counts, throughput, and ETA. The status paths
are:

```text
/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/status/qwen3_8b_materialization_v3_latest.json
/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/status/qwen3_8b_materialization_v3_history.jsonl
```

The timer uses `OnCalendar=*-*-* *:05:00 Asia/Seoul`; UTC log timestamps remain
UTC and are not presented as the intended user schedule.  A successful manifest
is the completion gate.  A merely running process or a 100% ledger without the
manifest is not reported as complete.

`export_packet_shards` remains available for bounded readiness/audit subsets.
It writes verified monthly v3 packet shards and proves resume/hash behavior; it
is not the production full-universe storage format.

In-memory v3 packets resolve every raw provenance pointer. The compact SFT row
carries a bounded evidence view plus a digest of the complete provenance set;
the immutable source database remains the regeneration source. This avoids
silently cutting the assistant target. Truncation is disabled. Both runners
tokenize every selected row with the pinned Qwen tokenizer and fail if any
rendered conversation exceeds 4096 tokens.

Create the smoke view from the same frozen candidate set with explicit per-task
quotas. Its manifest records that it is not a full or representative corpus:

```bash
python3 -m training.quant_llm.build_balanced_training_set \
  /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_candidate_v3 \
  /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_sft_smoke_v2 \
  --train-quotas 'etf_own_flow_analysis=2,stock_constituent_flow_analysis=2,all_stock_control_analysis=2' \
  --validation-per-task 1 \
  --replace
```

## Run gates and training

```bash
# Read-only: reports the exact image/model/memory/service state.
python3 -m training.quant_llm.service_gate --mode smoke

# These commands abort before Docker if dataset or memory gates fail.
bash training/quant_llm/run_smoke.sh
bash training/quant_llm/run_train.sh
```

Neither runner stops a live service. A human must deliberately free enough
unified memory after checking which Qwen/Gemma lane may be interrupted. Smoke
requires 36,000 MiB free; full training requires 48,000 MiB free.

The durable completion lane waits for the active full run, resumes from the
latest checkpoint only when the zero-based final step-14027 adapter is absent,
serves the exact adapter on localhost port 8018, evaluates all 21,106 sealed
test rows, and creates the release manifest only for a green report:

```bash
systemctl --user enable --now quant-qwen3-lora-complete.service
```

Its machine-readable state is written to
`/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/status/qwen3_8b_training_completion.json`.
The four-worker local collector binds every resumable prediction row to the
exact endpoint model, adapter artifact-set SHA256, and sealed-test SHA256. A
stale prediction file with different bindings is quarantined instead of being
reused, and the evaluator independently rejects any binding mismatch. The vLLM
container mount is also checked against the final adapter directory before an
already-running endpoint is accepted.
An evaluation-red result exits with status 2 and is not automatically retried;
transient training, endpoint, or collector failures remain restartable. The
lane does not resume FMP backfill and does not enable Quant AI Radar timers.

## Training direction after baseline

The baseline target is deterministic and auditable, not an expert human label.
The prompt-compacted, no-truncation dataset contract is v2. After the baseline
passes held-out 2025+ tests, create a separate expert-label corpus with
provenance, reviewer agreement, and counterfactual/leakage QA. Rank/alpha or
context sweeps happen only after the same frozen test set and structured schema
are used; the 30B model remains a later comparison, not the first Spark run.
