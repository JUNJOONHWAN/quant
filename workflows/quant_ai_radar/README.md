# Quant AI Radar

This is the post-training market and security analysis workflow. It does not
use the legacy 31-ticker ETF Flow dashboard, Hermes writer responses, or a
hard-coded ticker list.

## Standalone application and CLI

Quant AI Radar is registered as the independent Hermes application
`quant-ai-radar`. The `ai-radar` command automatically loads
`/home/zooh/.config/quant/quant-ai-radar.env`; callers do not need to export the
model endpoint or release manifest manually.

```bash
# Read-only readiness and current runtime state
ai-radar preflight
ai-radar status

# Full daily run. This publishes status/latest.json only after the complete
# selected queue and market synthesis finish.
ai-radar daily

# Complete non-publishing reference run
ai-radar daily --shadow

# Explicit symbols bypass the daily capacity selection while retaining every
# packet, point-in-time, release, and response-contract gate.
ai-radar analyze AAPL NVDA
```

Hermes uses the same entrypoint through Operations App Manager:

```bash
hermes apps run quant-ai-radar --json

hermes apps run quant-ai-radar \
  --input-json '{"action":"daily","shadow":true}' \
  --json

hermes apps run quant-ai-radar \
  --input-json '{"action":"analyze","symbols":["AAPL","NVDA"]}' \
  --json
```

App Manager seals request JSON and passes only its file path and canonical
SHA-256 to the app. The app schedule is registered disabled. Registering or
running this application does not enable the existing systemd timer, resume the
paused FMP history backfill, modify ETF Flow, or place trades.

Market Structure Oracle and Quant AI Radar remain separate applications.
Oracle owns the current-market incremental database; AI Radar reads that
sealed database as an input and creates its own trained-model reports. It does
not run or embed an Oracle report.

The complete model card, training recipe, daily operating contract, and
input/output examples are in
[`OPERATIONS_AND_MODEL_CARD.md`](OPERATIONS_AND_MODEL_CARD.md).
The authoritative current-data ownership, new-listing, lookahead, locking, and
recovery contract is
[`ORACLE_SHARED_DATA_CONTRACT.md`](ORACLE_SHARED_DATA_CONTRACT.md).

## End-to-end contract

```text
historical FMP/Massive + PIT ETF relations
  -> Qwen3-8B BF16 LoRA training
  -> frozen test evaluation and accepted release manifest
  -> Oracle single-writer current-market store
  -> immutable FMP history + Massive grouped EOD/ETF Global
  -> bounded FMP ETF-constituent refresh (new ETFs first)
  -> Oracle snapshot seal and SHA-256 receipt
  -> sealed read-only shared source binding
  -> persistent ETF/security relation index plus complete observed-universe scan
  -> quant.analysis_packet.v3 PIT and liquidity gates
  -> dynamic material-evidence selection (default max 64 ETFs + 192 stocks)
  -> trained LoRA judgement for the selected daily evidence
  -> complete coverage ledger for selected and nonselected securities
  -> deterministic selected-scope aggregate
  -> same trained LoRA market synthesis
  -> market_report.json and security_judgements.jsonl
```

The deterministic layer computes all numbers. The model may interpret facts,
counter-evidence, unknowns, regime, confidence, and conclusion. It may not
change facts, use a post-as-of date, or issue a trade instruction.

## Source roles

- `QUANT_DATASET`: source-preserving FMP/Massive daily observations, Massive
  ETF Global flow revisions, and PIT FMP ETF constituents/memberships. This is
  the training-aligned packet source.
- `Market Structure Oracle incremental store`: the only current-session writer.
  It repairs every session after the immutable FMP cutoff with Massive grouped
  daily data, captures Massive ETF Global revisions, refreshes FMP ETF
  constituents, and writes the snapshot seal. Quant AI Radar attaches it
  read-only to the immutable long-history database.
- `ETF RADAR`: a separate application. It is not a source, release gate,
  universe gate, selection input, or runtime dependency of Oracle/AI Radar.
- learned Qwen3-8B LoRA: the only judgement backend. There is no other model,
  cached prose, or hardcoded judgement fallback.

## Full-scan and selective-inference policy

The universe scan starts from every positive-volume security observed on the
as-of date. Every security with a visible ETF flow, ETF constituent snapshot,
or ETF membership receives a full v3 packet review. Securities with no ETF
relationship are counted as `all_stock_control_symbols` but are outside this
ETF-grounded product. There is no fixed ticker list.

The expensive model lane is intentionally not run over every related security.
Oracle-derived evidence dynamically ranks material daily flow, rotation,
price/flow disagreement, and constituent-transmission cases. The default
capacity ceiling is 64 ETFs and 192 stocks. These are ceilings, not fixed daily
counts or fixed symbol lists. Every nonselected candidate remains in
`coverage_ledger.jsonl` with its quantitative priority and explicit reason.

The SFT liquidity and leakage gates are reused unchanged:

- same-session positive-volume price and at least five price sessions;
- ETF trailing-20 liquidity gate: at least 10 sessions, 75% positive volume,
  and USD 1 million median dollar volume;
- raw one-session discontinuity below 45% unless a PIT corporate action exists;
- Massive ETF flow visible only under `massive_etf_flow_us_sessions_v1`;
- FMP ETF membership visible only after its derived next-session gate;
- no present-day active list removes a historically valid observation.

Selected securities are written to the restartable inference queue. The
25-item leader arrays in the final market report are presentation-only; every
selected result remains in `security_judgements.jsonl`.

## Prepare now without training or inference

Run with the quant host Python runtime:

```bash
cd /home/zooh/Documents/GitHub/quant
PYTHONPATH=$PWD /usr/bin/python3 \
  -m workflows.quant_ai_radar.run_quant_ai_radar --prepare-only
```

This writes the universe/release manifests and a resumable queue. It does not
start training, stop Qwen/Gemma, or call a model endpoint.

## After training

Training output is not deployable by itself. Serve the candidate adapter under
an isolated endpoint, collect every frozen-test response, and evaluate it. The
collector is resumable and rejects a server that returns a different model id:

```bash
python3 -m training.quant_llm.collect_frozen_predictions \
  --test-file /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_candidate_v3/test.jsonl \
  --endpoint http://127.0.0.1:8018/v1/chat/completions \
  --endpoint-model qwen3-8b-quant-lora-v1 \
  --output /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/evaluations/qwen3_8b_quant_lora_v1/predictions.jsonl

python3 -m training.quant_llm.evaluate_frozen_test \
  --dataset-manifest /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_sft_v2/manifest.json \
  --test-file /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_candidate_v3/test.jsonl \
  --predictions /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/evaluations/qwen3_8b_quant_lora_v1/predictions.jsonl \
  --endpoint-model qwen3-8b-quant-lora-v1 \
  --adapter-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/checkpoints/qwen3_8b_quant_lora_v1 \
  --artifact /absolute/path/to/adapter_model.safetensors \
  --artifact /absolute/path/to/adapter_config.json \
  --output /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/evaluations/qwen3_8b_quant_lora_v1/frozen_test_evaluation.json
```

The evaluator requires complete frozen-test coverage and gates schema validity,
exact deterministic facts, regime/signal accuracy, counter-evidence/unknown
recall, and zero post-as-of dates or trade directives. It hashes the test file,
predictions, dataset manifest, and exact adapter artifact set. Only then create
the release manifest:

```bash
python3 -m training.quant_llm.create_model_release \
  --model-id qwen3-8b-quant-lora-v1 \
  --endpoint-model qwen3-8b-quant-lora-v1 \
  --base-model /home/zooh/models/Qwen3-8B-bf16 \
  --adapter-root /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/checkpoints/qwen3_8b_quant_lora_v1 \
  --artifact /absolute/path/to/adapter_model.safetensors \
  --artifact /absolute/path/to/adapter_config.json \
  --dataset-manifest /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/datasets/qwen3_8b_sft_v2/manifest.json \
  --evaluation-report /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/evaluations/qwen3_8b_quant_lora_v1/frozen_test_evaluation.json \
  --output /home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/qwen3_8b_quant_lora_v1/release_manifest.json
```

The evaluation report must be `green`, report zero prohibited violations, and
have every required gate set to `true`. Release creation also proves that the
evaluated endpoint model, dataset manifest, and adapter artifact set are exactly
the ones being released. The runtime independently rechecks all bound hashes.

Create `/home/zooh/.config/quant/quant-ai-radar.env` only after the accepted
LoRA is served by an OpenAI-compatible endpoint:

```text
QUANT_AI_MODEL_ENDPOINT=http://127.0.0.1:8018/v1/chat/completions
QUANT_AI_RELEASE_MANIFEST=/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM/releases/qwen3_8b_quant_lora_v1/release_manifest.json
QUANT_AI_WORKERS=4
QUANT_AI_MAX_ETFS=64
QUANT_AI_MAX_STOCKS=192
QUANT_AI_MAX_CONSTITUENT_AVAILABLE_LAG_DAYS=45
# QUANT_AI_MODEL_TOKEN_FILE=/home/zooh/.config/quant/model-token
```

Use `--smoke-max-items N` for a non-publishable endpoint smoke. A production
market judgement is written only when the entire queue has no pending, running,
or error rows.

Use `--shadow` for a complete selected-scope run that renders every report but
does not update `status/latest.json`. Shadow is the required mode before timer
approval.

## On-demand symbol analysis

The daily 64 ETF / 192 stock ceilings do not restrict an explicit user
request. Any symbol with a valid current packet can be analyzed with the same
accepted model, including a no-ETF-relation `all_stock_control_analysis`:

```bash
PYTHONPATH=$PWD /usr/bin/python3 \
  -m workflows.quant_ai_radar.analyze_on_demand AAPL NVDA
```

Results are written under
`QUANT_AI_RADAR/on_demand/YYYY-MM-DD/<SYMBOL>.json|html`; the request receipt
contains both paths. Missing current price, quality, or PIT evidence fails
closed with a recorded reason.

## Shared Oracle data and scheduling

`run_daily_cycle.py` performs the following in order:

1. Run `prepare_shared_data.py`.
2. Reuse the existing COMPLETE Oracle snapshot when it already matches the
   latest fully closed NYSE session; otherwise let the Oracle single writer
   repair the missing sessions.
3. Verify the target-session row gate, D+1/D+2 flow freshness, Oracle snapshot
   seal, PIT constituent visibility, and source fingerprint.
4. Attach the immutable FMP history and Oracle incremental database read-only.
5. Refresh the persistent relation index from only new source row markers.
6. Scan the complete current universe, rank material evidence, and hand only
   the selected daily scope to the restartable inference queue.

This service does not run the old independent `refresh_daily_data.py` path and
does not recollect the same FMP/Massive data into a second database. The
historical FMP backfill remains an independent, paused/resumable job and is
never resumed by this workflow. PIT FMP constituents come from the latest
visible stored snapshot and fail closed when their provider-available date is
older than the configured maximum lag.

The supplied timer expresses a KST Tue-Sat 09:15 intent as the explicit UTC
stored schedule `Tue..Sat 00:15 UTC`. It is installed disabled. Enable it only
after training, evaluation, release creation, model serving, and a successful
non-publishable smoke. Do not enable it until an observed capacity benchmark
proves that the complete selected queue finishes inside the daily operating
window and the shadow run has no pending, running, or error rows.

Validate a completed manual shadow with the observed wall-clock duration:

```bash
PYTHONPATH=$PWD /usr/bin/python3 \
  -m workflows.quant_ai_radar.validate_shadow_run \
  --run-dir /home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/runs/YYYY-MM-DD \
  --elapsed-seconds <OBSERVED_SECONDS>
```

The validator fails closed on queue residue, prompt/response hash gaps,
post-as-of dates, trade directives, source fingerprint gaps, rendered artifact
hash drift, accidental `status/latest.json` publication, or an over-window
runtime. It writes `shadow_gate_audit.json`. One passing frozen-date shadow is
only the first activation prerequisite; the timer still requires five
consecutive daily passing shadows and explicit user approval.

Capture the matching DGX runtime window after each shadow:

```bash
PYTHONPATH=$PWD /usr/bin/python3 \
  -m workflows.quant_ai_radar.validate_runtime_readiness \
  --run-dir /home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/runs/YYYY-MM-DD \
  --since "<shadow start KST>"
```

`runtime_readiness_audit.json` keeps Linux OOM-kill, NVIDIA Xid, NVIDIA
`NV_ERR_NO_MEMORY`, vLLM OOM/restart, Docker boot recovery, user linger, disabled
Radar units, and paused FMP backfill as separate evidence. A graphics-context
`NV_ERR_NO_MEMORY` warning does not invalidate hash-verified reference output
when vLLM stayed healthy, but that shadow does not count toward timer activation.

## Decision-grade report contract

The released LoRA is used for structured judgement, not arithmetic. Python owns
all point-in-time facts, ETF-to-constituent weighted flow calculations, breadth
counts, and displayed values. The model selects the market state, relevant
confirmation/contradiction evidence IDs, leaders, and qualitative summary.

Market synthesis uses vLLM guided JSON Schema. Evidence IDs are enums built from
the current immutable evidence catalog, so an unknown ID cannot be generated.
Model-written evidence prose is discarded; confirmation and contradiction cards
render the exact catalog object selected by the model. Digits in free-form
summary/unknown text are removed because exact values are renderer-owned. A
separate semantic gate still rejects reversed directional comparisons.

`quality_audit.json` scores these six publication gates from zero to ten:

- data integrity
- numeric faithfulness
- security analysis
- market structure
- model judgement integration
- report usability

Every category must be at least `8.0`. A lower score produces
`shadow_quality_failed_not_published` or blocks reference publication. This is
a program/runtime contract and does not require retraining the released adapter.

## Outputs

```text
/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/
  runs/YYYY-MM-DD/
    universe_manifest.json
    candidates.jsonl
    selection_manifest.json
    selected_candidates.jsonl
    coverage_ledger.jsonl
    oracle_market_features.json
    selected_run_queue.sqlite3
    security_judgements.jsonl
    market_report.json
    market_report.html
    quality_audit.json
    security_index.html
    security_reports/<SYMBOL>.json
    security_reports/<SYMBOL>.html
    rendered_reports_manifest.json
    shadow_gate_audit.json
    runtime_readiness_audit.json
    run_state.json
  on_demand/YYYY-MM-DD/
    <SYMBOL>.json
    <SYMBOL>.html
    latest_request.json
  status/latest.json
  status/daily_data_refresh.json
  status/daily_cycle.json
```

`market_report.json` is analysis only. Order generation and trade execution are
not part of this repository or service.
