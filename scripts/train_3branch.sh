#!/usr/bin/env bash
# Train AIDE_3BRANCH on a single GPU.
# Usage:
#   bash scripts/train_3branch.sh
#   or override variables below and run.

set -euo pipefail

# Resolve repo root so script can be launched from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Paths (edit these).
TRAIN_DATA="/path/to/progan/train"
EVAL_DATA="/path/to/progan/eval"
RESNET_PATH="pretrained_ckpts/resnet50.pth"
CONVNEXT_PATH="pretrained_ckpts/open_clip_pytorch_model.bin"
NPR_PATH="pretrained_ckpts/NPR.pth"
OUTPUT_DIR="results/3branch_train"

# Single-GPU defaults.
BATCH_SIZE=8
UPDATE_FREQ=2
BLR=1e-4
EPOCHS=20
SAVE_FREQ=5
NUM_WORKERS=4

PY_ARGS=("$@")

if [ ! -d "${TRAIN_DATA}" ]; then
  echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"
  exit 1
fi

if [ ! -d "${EVAL_DATA}" ]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi

if [ ! -f "${RESNET_PATH}" ]; then
  echo "[ERROR] RESNET_PATH not found: ${RESNET_PATH}"
  exit 1
fi

if [ ! -f "${CONVNEXT_PATH}" ]; then
  echo "[ERROR] CONVNEXT_PATH not found: ${CONVNEXT_PATH}"
  exit 1
fi

if [ ! -f "${NPR_PATH}" ]; then
  echo "[ERROR] NPR_PATH not found: ${NPR_PATH}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python main_finetune.py \
  --model AIDE_3BRANCH \
  --data_path "${TRAIN_DATA}" \
  --eval_data_path "${EVAL_DATA}" \
  --resnet_path "${RESNET_PATH}" \
  --convnext_path "${CONVNEXT_PATH}" \
  --npr_path "${NPR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --update_freq "${UPDATE_FREQ}" \
  --blr "${BLR}" \
  --epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --use_amp True \
  --freeze_npr True \
  --npr_proj_dim 128 \
  --npr_branch_dropout 0.3 \
  --save_ckpt True \
  --save_ckpt_freq "${SAVE_FREQ}" \
  "${PY_ARGS[@]}"

echo "Training finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/checkpoint-best.pth"
