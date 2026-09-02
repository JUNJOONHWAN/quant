#!/usr/bin/env bash
set -euo pipefail

IMAGE="${QUANT_V11_R2_IMAGE:-nvcr.io/nvidia/nemo-automodel:26.06.00}"
REPO="${QUANT_REPO_ROOT:-/home/zooh/Documents/GitHub/quant}"
DATA="${QUANT_DATA_ROOT:-/home/zooh/Documents/GitHub/STOCKDATA}"

exec docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace \
  -v "${REPO}:/workspace:ro" \
  -v "${DATA}:${DATA}" \
  -w /workspace \
  "${IMAGE}" \
  python -m training.quant_flow_graph_v11_r2 "$@"
