#!/usr/bin/env bash
# Train AIDE_3BRANCH on single or multi-GPU.
# Usage:
#   Single GPU:  bash scripts/train_3branch.sh
#   Dual GPU:    NUM_GPUS=2 bash scripts/train_3branch.sh
#   or override variables below and run.

set -euo pipefail

# Resolve repo root so script can be launched from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Paths (edit these).
TRAIN_DATA="/data/CNNSpot/progan_train"
EVAL_DATA="/data/CNNSpot/progan_val"
RESNET_PATH="/AIGCDetect/models/123456/pretrained_ckpts/resnet50.pth"
CONVNEXT_PATH="/AIGCDetect/models/123456/pretrained_ckpts/open_clip_pytorch_model.bin"
NPR_PATH="/AIGCDetect/models/123456/pretrained_ckpts/NPR.pth"
OUTPUT_DIR="/AIGCDetect/models/123456/results"

# GPU configuration (set NUM_GPUS=2 for dual-GPU training).
NUM_GPUS=${NUM_GPUS:-2}

# Training hyperparameters (dual RTX 4090, effective batch=256).
if [ "${NUM_GPUS}" -eq 1 ]; then
  BATCH_SIZE=32
  UPDATE_FREQ=8
  NUM_WORKERS=8
else
  BATCH_SIZE=32
  UPDATE_FREQ=4
  NUM_WORKERS=8
fi

BLR=1e-4
WARMUP_EPOCHS=2
EPOCHS=10
SAVE_FREQ=1
WEIGHT_DECAY=0.05
NPR_BRANCH_DROPOUT=0.3 # reduced from 0.5 for more stable convergence

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

echo "=========================================="
echo "Training Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${UPDATE_FREQ}"
echo "  Effective batch size: $((BATCH_SIZE * UPDATE_FREQ * NUM_GPUS))"
echo "  Workers: ${NUM_WORKERS}"
echo "=========================================="

if [ "${NUM_GPUS}" -eq 1 ]; then
  # Single GPU training
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
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --use_amp True \
    --freeze_npr False \
    --npr_proj_dim 128 \
    --npr_branch_dropout "${NPR_BRANCH_DROPOUT}" \
    --save_ckpt True \
    --save_ckpt_freq "${SAVE_FREQ}" \
    "${PY_ARGS[@]}"
else
  # Multi-GPU training with torchrun
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 main_finetune.py \
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
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --use_amp True \
    --freeze_npr False \
    --npr_proj_dim 128 \
    --npr_branch_dropout "${NPR_BRANCH_DROPOUT}" \
    --save_ckpt True \
    --save_ckpt_freq "${SAVE_FREQ}" \
    "${PY_ARGS[@]}"
fi

echo "Training finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/checkpoint-best.pth"
