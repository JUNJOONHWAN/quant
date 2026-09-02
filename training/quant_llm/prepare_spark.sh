#!/usr/bin/env bash
set -euo pipefail

IMAGE="${NEMO_AUTOMODEL_IMAGE:-nvcr.io/nvidia/nemo-automodel:26.06.00}"
MODEL_ID="${QUANT_BASE_MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_DIR="${QUANT_BASE_MODEL_DIR:-/home/zooh/models/Qwen3-8B-bf16}"
DATA_ROOT="${QUANT_LLM_DATA_ROOT:-/home/zooh/Documents/GitHub/STOCKDATA/QUANT_LLM}"

install -d "${MODEL_DIR}" "${DATA_ROOT}/datasets" "${DATA_ROOT}/checkpoints" "${DATA_ROOT}/logs" "${DATA_ROOT}/audits"
docker pull "${IMAGE}"
hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"

docker run --rm \
  --gpus all \
  --ipc=host \
  -v "${MODEL_DIR}:/models/Qwen3-8B:ro" \
  "${IMAGE}" \
  python -c 'import nemo_automodel, torch, transformers; print({"nemo_automodel": nemo_automodel.__file__, "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.cuda.is_available()})'
