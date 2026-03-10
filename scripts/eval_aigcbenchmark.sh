#!/usr/bin/env bash
# Evaluate AIDE on AIGCBenchmark with the same protocol for 3-branch or 2-branch.
# Usage:
#   bash scripts/eval_aigcbenchmark.sh <3branch|2branch> <checkpoint_path> <train_data_path> <aigcbenchmark_path> [extra args...]
# Example:
#   bash scripts/eval_aigcbenchmark.sh 3branch results/3branch_train/checkpoint-best.pth /data/progan/train /data/AIGCBenchmark
#   bash scripts/eval_aigcbenchmark.sh 2branch /path/to/official_2branch.pth /data/progan/train /data/AIGCBenchmark

set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <3branch|2branch> <checkpoint_path> <train_data_path> <aigcbenchmark_path> [extra args...]"
  exit 1
fi

MODE="$1"
CKPT_PATH="$2"
TRAIN_DATA="$3"
AIGC_BENCHMARK="$4"
shift 4
PY_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

CONVNEXT_PATH="pretrained_ckpts/open_clip_pytorch_model.bin"
NPR_PATH="pretrained_ckpts/NPR.pth"
BATCH_SIZE=8
NUM_WORKERS=4

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERROR] checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

if [ ! -d "${TRAIN_DATA}" ]; then
  echo "[ERROR] train_data_path not found: ${TRAIN_DATA}"
  echo "main_finetune.py still instantiates TrainDataset in eval mode, so this path is required."
  exit 1
fi

if [ ! -d "${AIGC_BENCHMARK}" ]; then
  echo "[ERROR] AIGCBenchmark path not found: ${AIGC_BENCHMARK}"
  exit 1
fi

if [ ! -f "${CONVNEXT_PATH}" ]; then
  echo "[ERROR] CONVNEXT_PATH not found: ${CONVNEXT_PATH}"
  exit 1
fi

if [ "${MODE}" = "3branch" ]; then
  MODEL="AIDE_3BRANCH"
  OUTPUT_DIR="results/eval_3branch"
  if [ ! -f "${NPR_PATH}" ]; then
    echo "[ERROR] NPR_PATH not found: ${NPR_PATH}"
    exit 1
  fi
  EXTRA_MODEL_ARGS=(--npr_path "${NPR_PATH}")
elif [ "${MODE}" = "2branch" ]; then
  MODEL="AIDE_2BRANCH"
  OUTPUT_DIR="results/eval_2branch"
  EXTRA_MODEL_ARGS=()
else
  echo "[ERROR] MODE must be '3branch' or '2branch', got: ${MODE}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================="
echo "Evaluation config"
echo "  Model:       ${MODEL}"
echo "  Checkpoint:  ${CKPT_PATH}"
echo "  Test set:    ${AIGC_BENCHMARK}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "=============================="

python main_finetune.py \
  --model "${MODEL}" \
  --eval True \
  --resume "${CKPT_PATH}" \
  --convnext_path "${CONVNEXT_PATH}" \
  "${EXTRA_MODEL_ARGS[@]}" \
  --data_path "${TRAIN_DATA}" \
  --eval_data_path "${AIGC_BENCHMARK}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  "${PY_ARGS[@]}"

echo "Evaluation finished. CSV saved in: ${OUTPUT_DIR}/"
