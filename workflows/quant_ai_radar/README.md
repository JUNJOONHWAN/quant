# Quant AI Radar

This is the post-training market and security analysis workflow. It does not
use the legacy 31-ticker ETF Flow dashboard, Hermes writer responses, or a
hard-coded ticker list.

## End-to-end contract

```text
historical FMP/Massive + PIT ETF relations
  -> Qwen3-8B BF16 LoRA training
  -> frozen test evaluation and accepted release manifest
  -> daily FMP all-symbol EOD + Massive grouped EOD/ETF Global
  -> hash-verified immutable ETF RADAR release reuse
  -> complete observed-universe scan
  -> quant.analysis_packet.v3 PIT and liquidity gates
  -> trained LoRA judgement for every eligible ETF-related security
  -> deterministic full-run aggregate
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
- `ETFRADAR`: existing full ETF universe and daily ETF/cluster/stock evidence.
  A COMPLETE immutable release is reused, and every required release file is
  SHA256-verified. It is not downloaded again by this workflow.
- learned Qwen3-8B LoRA: the only judgement backend. There is no other model,
  cached prose, or hardcoded judgement fallback.

## Full-universe policy

The universe scan starts from every positive-volume security observed on the
as-of date. Every security with a visible ETF flow, ETF constituent snapshot,
or ETF membership receives a full v3 packet review. Securities with no ETF
relationship are counted as `all_stock_control_symbols` but are outside this
ETF-grounded product. There is no fixed ticker list or top-N inference gate.

The SFT liquidity and leakage gates are reused unchanged:

- same-session positive-volume price and at least five price sessions;
- ETF trailing-20 liquidity gate: at least 10 sessions, 75% positive volume,
  and USD 1 million median dollar volume;
- raw one-session discontinuity below 45% unless a PIT corporate action exists;
- Massive ETF flow visible only under `massive_etf_flow_us_sessions_v1`;
- FMP ETF membership visible only after its derived next-session gate;
- no present-day active list removes a historically valid observation.

All eligible securities are written to the inference queue. The 25-item leader
arrays in the final report are presentation-only; the complete results remain
in `security_judgements.jsonl`.

## Prepare now without training or inference

Run with the ETFRADAR virtual environment because it owns the pinned Parquet
runtime:

```bash
cd /home/zooh/Documents/GitHub/quant
PYTHONPATH=$PWD /home/zooh/Documents/GitHub/ETFRADAR/.venv/bin/python \
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
# QUANT_AI_MODEL_TOKEN_FILE=/home/zooh/.config/quant/model-token
```

Use `--smoke-max-items N` for a non-publishable endpoint smoke. A production
market judgement is written only when the entire queue has no pending, running,
or error rows.

## Daily collection and scheduling

`refresh_daily_data.py` performs the following in order:

1. Verify and reuse the same-date COMPLETE ETF RADAR release.
2. Capture FMP daily data for the complete filtered US stock/ETF symbol file.
3. Capture Massive grouped US-stock daily data.
4. Capture the recent Massive ETF Global processed-date window with strict
   freshness.
5. Hand off to the full-universe inference queue.

FMP uses the shared 240 requests/min limiter, leaving headroom under the
300/min account ceiling. The historical FMP backfill remains an independent,
paused/resumable job and is never resumed by this workflow.

The supplied timer expresses a KST Tue-Sat 09:15 intent as the explicit UTC
stored schedule `Tue..Sat 00:15 UTC`. It is installed disabled. Enable it only
after training, evaluation, release creation, model serving, and a successful
non-publishable smoke.

## Outputs

```text
/home/zooh/Documents/GitHub/STOCKDATA/QUANT_AI_RADAR/
  runs/YYYY-MM-DD/
    universe_manifest.json
    candidates.jsonl
    etfradar_release_binding.json
    run_queue.sqlite3
    security_judgements.jsonl
    market_report.json
    run_state.json
  status/latest.json
  status/daily_data_refresh.json
  status/daily_cycle.json
```

`market_report.json` is analysis only. Order generation and trade execution are
not part of this repository or service.
