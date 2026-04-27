#!/usr/bin/env bash
# Train AIDE_3BRANCH with ProGAN + AIGCDetectionBenchmark/test/ADM as the
# diffusion source.
#
# This is a shortcut wrapper around train_3branch_mixed.sh for the current
# "keep it simple" experiment path:
#   1. use benchmark/test/ADM directly as DIFFUSION_DATA
#   2. train with the usual mixed-training pipeline
#   3. evaluate on AIGCDetectionBenchmark/test with ADM excluded
#
# Typical usage:
#   bash scripts/train_3branch_admtest.sh
#   NUM_GPUS=2 bash scripts/train_3branch_admtest.sh
#   MIX_RATIO=0.2 bash scripts/train_3branch_admtest.sh
#   OUTPUT_DIR=/path/to/results bash scripts/train_3branch_admtest.sh
#
# Matching evaluation example:
#   EXCLUDE_SUBSETS=ADM bash scripts/eval_aigcbenchmark.sh \
#       3branch /path/to/checkpoint-best.pth /path/to/progan_train /path/to/AIGCDetectionBenchmark/test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

DIFFUSION_DATA_DEFAULT="${DIFFUSION_DATA:-/data/default/AIGCDetectionBenchmark/test/ADM}"
EVAL_EXCLUDE_DEFAULT="${EVAL_EXCLUDE_DEFAULT:-ADM}"
NUM_GPUS_DEFAULT="${NUM_GPUS:-2}"
MIX_RATIO_DEFAULT="${MIX_RATIO:-0.1}"
MIX_TAG_DEFAULT="${MIX_TAG:-${MIX_RATIO_DEFAULT/./p}}"
OUTPUT_DIR_DEFAULT="${OUTPUT_DIR:-/AIGCDetect/models/123456/results/3branch_admtest_${MIX_TAG_DEFAULT}}"

echo "=========================================="
echo "ADM-test mixed training shortcut"
echo "  GPUs: ${NUM_GPUS_DEFAULT}"
echo "  Diffusion source: ${DIFFUSION_DATA_DEFAULT}"
echo "  Mix ratio: ${MIX_RATIO_DEFAULT}"
echo "  Output dir: ${OUTPUT_DIR_DEFAULT}"
echo "  Suggested eval exclude: ${EVAL_EXCLUDE_DEFAULT}"
echo "=========================================="

NUM_GPUS="${NUM_GPUS_DEFAULT}" \
DIFFUSION_DATA="${DIFFUSION_DATA_DEFAULT}" \
OUTPUT_DIR="${OUTPUT_DIR_DEFAULT}" \
bash "${SCRIPT_DIR}/train_3branch_mixed.sh" "$@"
