#!/usr/bin/env bash
# Evaluate 3-branch model with frozen NPR on specific Diffusion datasets.
# Usage:
#   bash scripts/eval_freeze_npr.sh <checkpoint_path> <train_data_path> <aigcbenchmark_path> [extra args...]
#   bash scripts/eval_freeze_npr.sh   # use defaults

set -euo pipefail

DEFAULT_CKPT="/AIGCDetect/models/123456/results/before_checkpoint-best.pth"
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
OUTPUT_DIR="results/eval_freeze_npr"

NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-24}

# Specify which datasets to evaluate (SD1.4, Midjourney, ADM)
EVAL_SUBSETS="stable_diffusion_v_1_4,Midjourney,wukong"

mkdir -p "${OUTPUT_DIR}"

echo "=============================="
echo "Freeze-NPR eval (SD1.4, MJ, ADM only)"
echo "  Checkpoint:  ${CKPT_PATH}"
echo "  Test set:    ${AIGC_BENCHMARK}"
echo "  Datasets:    ${EVAL_SUBSETS}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "=============================="

if [ "${NUM_GPUS}" -eq 1 ]; then
  python main_finetune.py \
    --model AIDE_3BRANCH \
    --eval True \
    --resume "${CKPT_PATH}" \
    --convnext_path "${CONVNEXT_PATH}" \
    --npr_path "${NPR_PATH}" \
    --freeze_npr True \
    --data_path "${TRAIN_DATA}" \
    --eval_data_path "${AIGC_BENCHMARK}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 16 \
    --use_amp True \
    --eval_subsets "${EVAL_SUBSETS}" \
    "${PY_ARGS[@]}"
else
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 main_finetune.py \
    --model AIDE_3BRANCH \
    --eval True \
    --resume "${CKPT_PATH}" \
    --convnext_path "${CONVNEXT_PATH}" \
    --npr_path "${NPR_PATH}" \
    --freeze_npr True \
    --data_path "${TRAIN_DATA}" \
    --eval_data_path "${AIGC_BENCHMARK}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 16 \
    --use_amp True \
    --eval_subsets "${EVAL_SUBSETS}" \
    "${PY_ARGS[@]}"
fi

echo "Freeze-NPR eval finished. CSV saved in: ${OUTPUT_DIR}/"
