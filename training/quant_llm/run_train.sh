#!/usr/bin/env bash
set -euo pipefail

REPO="/home/zooh/Documents/GitHub/quant"
DATA_ROOT="/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM"
MODEL_DIR="/home/zooh/models/Qwen3-8B-bf16"
IMAGE="${NEMO_AUTOMODEL_IMAGE:-nvcr.io/nvidia/nemo-automodel:26.06.00}"

cd "${REPO}"
python3 -m training.quant_llm.validate_sft_dataset "${DATA_ROOT}/datasets/qwen3_8b_sft_v2"
docker run --rm \
  -v "${REPO}:/workspace/quant:ro" \
  -v "${DATA_ROOT}:/data" \
  -v "${MODEL_DIR}:/models/Qwen3-8B:ro" \
  -w /workspace/quant \
  "${IMAGE}" \
  python -m training.quant_llm.audit_token_lengths \
    /data/datasets/qwen3_8b_sft_v2 \
    --model-path /models/Qwen3-8B \
    --max-length 4096 \
    --output /data/audits/qwen3_8b_sft_v2.token_audit.json
python3 -m training.quant_llm.service_gate --mode full --image "${IMAGE}" --model-path "${MODEL_DIR}"

docker run --rm \
  --gpus all \
  --ipc=host \
  -v "${REPO}:/workspace/quant:ro" \
  -v "${DATA_ROOT}:/data" \
  -v "${MODEL_DIR}:/models/Qwen3-8B:ro" \
  -w /workspace/quant \
  "${IMAGE}" \
  automodel training/quant_llm/configs/qwen3_8b_lora_spark.yaml --nproc-per-node 1
