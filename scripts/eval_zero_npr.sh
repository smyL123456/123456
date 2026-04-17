#!/usr/bin/env bash
# E2a diagnostic: evaluate 3-branch model with NPR features zeroed out at eval time.
# This proves whether NPR causes negative transfer on OOD generators.
# Usage:
#   bash scripts/eval_zero_npr.sh <checkpoint_path> <train_data_path> <aigcbenchmark_path> [extra args...]
#   bash scripts/eval_zero_npr.sh   # use defaults

set -euo pipefail

DEFAULT_CKPT="/AIGCDetect/models/123456/results/3branch_train/checkpoint-best.pth"
DEFAULT_TRAIN="/data/CNNSpot/progan_train"
DEFAULT_TEST="/data/default/AIGCDetectionBenchmark/test"

if [ "$#" -eq 0 ]; then
  CKPT_PATH="${DEFAULT_CKPT}"
  TRAIN_DATA="${DEFAULT_TRAIN}"
  AIGC_BENCHMARK="${DEFAULT_TEST}"
  PY_ARGS=()
elif [ "$#" -lt 3 ]; then
  echo "Usage: $0 [<checkpoint_path> <train_data_path> <aigcbenchmark_path> [extra args...]]"
  exit 1
else
  CKPT_PATH="$1"
  TRAIN_DATA="$2"
  AIGC_BENCHMARK="$3"
  shift 3
  PY_ARGS=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

CONVNEXT_PATH="/AIGCDetect/models/123456/pretrained_ckpts/open_clip_pytorch_model.bin"
NPR_PATH="/AIGCDetect/models/123456/pretrained_ckpts/NPR.pth"
OUTPUT_DIR="results/eval_zero_npr"

NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-24}

mkdir -p "${OUTPUT_DIR}"

echo "=============================="
echo "E2a: Zero-NPR diagnostic eval"
echo "  Checkpoint:  ${CKPT_PATH}"
echo "  Test set:    ${AIGC_BENCHMARK}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "=============================="

if [ "${NUM_GPUS}" -eq 1 ]; then
  python main_finetune.py \
    --model AIDE_3BRANCH \
    --eval True \
    --resume "${CKPT_PATH}" \
    --convnext_path "${CONVNEXT_PATH}" \
    --npr_path "${NPR_PATH}" \
    --zero_npr_at_eval True \
    --data_path "${TRAIN_DATA}" \
    --eval_data_path "${AIGC_BENCHMARK}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 16 \
    --use_amp True \
    "${PY_ARGS[@]}"
else
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 main_finetune.py \
    --model AIDE_3BRANCH \
    --eval True \
    --resume "${CKPT_PATH}" \
    --convnext_path "${CONVNEXT_PATH}" \
    --npr_path "${NPR_PATH}" \
    --zero_npr_at_eval True \
    --data_path "${TRAIN_DATA}" \
    --eval_data_path "${AIGC_BENCHMARK}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 16 \
    --use_amp True \
    "${PY_ARGS[@]}"
fi

echo "E2a eval finished. CSV saved in: ${OUTPUT_DIR}/"
