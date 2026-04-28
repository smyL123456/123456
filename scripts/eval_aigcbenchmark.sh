#!/usr/bin/env bash
# Evaluate AIDE on AIGCBenchmark with the same protocol for 3-branch or 2-branch.
# Usage:
#   With arguments:
#     bash scripts/eval_aigcbenchmark.sh <3branch|2branch> <checkpoint_path> <train_data_path> <aigcbenchmark_path> [extra args...]
#   With defaults (3branch):
#     bash scripts/eval_aigcbenchmark.sh
# Example:
#   bash scripts/eval_aigcbenchmark.sh 3branch /AIGCDetect/models/123456/results/checkpoint-best.pth /data/CNNSpot/progan_train /data/default/AIGCDetectionBenchmark/test
#   bash scripts/eval_aigcbenchmark.sh 2branch /path/to/official_2branch.pth /data/CNNSpot/progan_train /data/default/AIGCDetectionBenchmark/test
#   EXCLUDE_SUBSETS=ADM bash scripts/eval_aigcbenchmark.sh 3branch /path/to/checkpoint.pth /path/to/progan_train /path/to/AIGCDetectionBenchmark/test

set -euo pipefail

# Default paths (from train_3branch.sh)
DEFAULT_MODE="3branch"
DEFAULT_CKPT="/AIGCDetect/models/123456/results/3branch_train/checkpoint-best.pth"
DEFAULT_TRAIN="/data/CNNSpot/progan_train"
DEFAULT_TEST="/data/default/AIGCDetectionBenchmark/test"

if [ "$#" -eq 0 ]; then
  MODE="${DEFAULT_MODE}"
  CKPT_PATH="${DEFAULT_CKPT}"
  TRAIN_DATA="${DEFAULT_TRAIN}"
  AIGC_BENCHMARK="${DEFAULT_TEST}"
  PY_ARGS=()
elif [ "$#" -lt 4 ]; then
  echo "Usage: $0 [<3branch|2branch> <checkpoint_path> <train_data_path> <aigcbenchmark_path> [extra args...]]"
  echo "Or run without arguments to use default paths."
  exit 1
else
  MODE="$1"
  CKPT_PATH="$2"
  TRAIN_DATA="$3"
  AIGC_BENCHMARK="$4"
  shift 4
  PY_ARGS=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

CONVNEXT_PATH="/AIGCDetect/models/123456/pretrained_ckpts/open_clip_pytorch_model.bin"
NPR_PATH="/AIGCDetect/models/123456/pretrained_ckpts/NPR.pth"
# Changed default from concat to residual to match E3 checkpoint (residual fusion mode).
# For older E2 concat checkpoints, run with: FUSION_TYPE=concat bash scripts/eval_aigcbenchmark.sh ...
# Default to residual (E3 mode); set FUSION_TYPE=concat for E2 checkpoints
FUSION_TYPE="${FUSION_TYPE:-residual}"
NPR_RESIDUAL_ALPHA_INIT="${NPR_RESIDUAL_ALPHA_INIT:-0.1}"

# GPU configuration
NUM_GPUS=${NUM_GPUS:-2}
EXCLUDE_SUBSETS="${EXCLUDE_SUBSETS:-}"

if [ "${NUM_GPUS}" -eq 1 ]; then
  BATCH_SIZE=64
  NUM_WORKERS=16
else
  BATCH_SIZE=64
  NUM_WORKERS=16
fi

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
  EXTRA_MODEL_ARGS=(
    --npr_path "${NPR_PATH}"
    --fusion_type "${FUSION_TYPE}"
    --npr_residual_alpha_init "${NPR_RESIDUAL_ALPHA_INIT}"
  )
elif [ "${MODE}" = "2branch" ]; then
  MODEL="AIDE_2BRANCH"
  OUTPUT_DIR="results/eval_2branch"
  EXTRA_MODEL_ARGS=()
else
  echo "[ERROR] MODE must be '3branch' or '2branch', got: ${MODE}"
  exit 1
fi

EXTRA_EVAL_ARGS=()
if [ -n "${EXCLUDE_SUBSETS}" ]; then
  EXTRA_EVAL_ARGS+=(--exclude_eval_subsets "${EXCLUDE_SUBSETS}")
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================="
echo "Evaluation config"
echo "  Model:       ${MODEL}"
echo "  Checkpoint:  ${CKPT_PATH}"
echo "  Test set:    ${AIGC_BENCHMARK}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Batch size:  ${BATCH_SIZE}"
if [ "${MODE}" = "3branch" ]; then
  echo "  Fusion:      ${FUSION_TYPE}"
fi
if [ -n "${EXCLUDE_SUBSETS}" ]; then
  echo "  Exclude:     ${EXCLUDE_SUBSETS}"
fi
echo "=============================="

if [ "${NUM_GPUS}" -eq 1 ]; then
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
    --use_amp True \
    "${EXTRA_EVAL_ARGS[@]}" \
    "${PY_ARGS[@]}"
else
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 main_finetune.py \
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
    --use_amp True \
    "${EXTRA_EVAL_ARGS[@]}" \
    "${PY_ARGS[@]}"
fi

echo "Evaluation finished. CSV saved in: ${OUTPUT_DIR}/"
