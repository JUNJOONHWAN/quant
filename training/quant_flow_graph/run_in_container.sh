#!/usr/bin/env bash
set -euo pipefail

REPO="/home/zooh/Documents/GitHub/quant"
STOCKDATA="/home/zooh/Documents/GitHub/STOCKDATA"
IMAGE="${NEMO_AUTOMODEL_IMAGE:-nvcr.io/nvidia/nemo-automodel:26.06.00}"
MEMORY_LIMIT="${QUANT_FLOW_GRAPH_MEMORY_LIMIT:-28g}"
CUDA_MEMORY_FRACTION="${QUANT_FLOW_GRAPH_CUDA_MEMORY_FRACTION:-0.15}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <train-smoke|walk-forward|ai-specific-gate> [arguments...]" >&2
  exit 2
fi

case "$1" in
  train-smoke|walk-forward)
    MODULE="training.quant_flow_graph"
    ;;
  ai-specific-gate)
    MODULE="training.quant_flow_graph.ai_specific_gate"
    shift
    ;;
  *)
    echo "container runner only accepts train-smoke, walk-forward, or ai-specific-gate" >&2
    exit 2
    ;;
esac

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  --gpus all \
  --ipc=host \
  --memory "${MEMORY_LIMIT}" \
  --memory-swap "${MEMORY_LIMIT}" \
  --oom-score-adj 700 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e QUANT_FLOW_GRAPH_CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION}" \
  -v "${REPO}:/workspace/quant:ro" \
  -v "${STOCKDATA}:${STOCKDATA}" \
  -w /workspace/quant \
  "${IMAGE}" \
  python3 -m "${MODULE}" "$@"
