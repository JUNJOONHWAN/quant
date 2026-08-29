#!/usr/bin/env bash
set -euo pipefail

IMAGE="${QUANT_V14_IMAGE:-nvcr.io/nvidia/nemo-automodel:26.06.00}"
REPO="${QUANT_REPO_ROOT:-/home/zooh/Documents/GitHub/quant}"
DATA="${QUANT_DATA_ROOT:-/home/zooh/Documents/GitHub/STOCKDATA}"
DEPS="${QUANT_V14_DEPS:-${DATA}/QUANT_FORECAST/flow_graph_v12/python_deps}"
CPU_LIMIT="${QUANT_V14_CPU_LIMIT:-10}"
MEMORY_LIMIT="${QUANT_V14_MEMORY_LIMIT:-40g}"

exec docker run --rm --network none \
  --cpus "${CPU_LIMIT}" \
  --memory "${MEMORY_LIMIT}" \
  --user "$(id -u):$(id -g)" \
  -e PYTHONPATH="/workspace:${DEPS}" \
  -e OMP_NUM_THREADS="${CPU_LIMIT}" \
  -e OPENBLAS_NUM_THREADS="${CPU_LIMIT}" \
  -e MKL_NUM_THREADS="${CPU_LIMIT}" \
  -v "${REPO}:/workspace:ro" \
  -v "${DATA}:${DATA}" \
  -w /workspace \
  "${IMAGE}" \
  python -m training.quant_flow_graph_v14 "$@"
