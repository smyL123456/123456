#!/usr/bin/env bash
# Train AIDE_3BRANCH with ProGAN + X% Diffusion mixed training (E3/E4).
# Usage:
#   Single GPU:  bash scripts/train_3branch_mixed.sh
#   Dual GPU:    NUM_GPUS=2 bash scripts/train_3branch_mixed.sh
#   Override MIX_RATIO=0.2 to run E4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Paths (override via env when running remotely).
TRAIN_DATA="${TRAIN_DATA:-/data/CNNSpot/progan_train}"
DIFFUSION_DATA="${DIFFUSION_DATA:-/data/CNNSpot/diffusion_train}"   # must be deduplicated vs test set
MIX_RATIO="${MIX_RATIO:-0.1}"
EVAL_DATA="${EVAL_DATA:-/data/CNNSpot/progan_val}"
DEDUP_REFERENCE_PATH="${DEDUP_REFERENCE_PATH:-}"
DEDUP_MODE="${DEDUP_MODE:-name}"
RESNET_PATH="${RESNET_PATH:-/AIGCDetect/models/123456/pretrained_ckpts/resnet50.pth}"
CONVNEXT_PATH="${CONVNEXT_PATH:-/AIGCDetect/models/123456/pretrained_ckpts/open_clip_pytorch_model.bin}"
NPR_PATH="${NPR_PATH:-/AIGCDetect/models/123456/pretrained_ckpts/NPR.pth}"
MIX_TAG="${MIX_TAG:-${MIX_RATIO/./p}}"
OUTPUT_DIR="${OUTPUT_DIR:-/AIGCDetect/models/123456/results/3branch_mixed_${MIX_TAG}}"

NUM_GPUS=${NUM_GPUS:-2}

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
SAVE_FREQ=3
WEIGHT_DECAY=0.05
NPR_BRANCH_DROPOUT=0.3

PY_ARGS=("$@")

if [ ! -d "${TRAIN_DATA}" ]; then
  echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"
  exit 1
fi

if [ ! -d "${DIFFUSION_DATA}" ]; then
  echo "[ERROR] DIFFUSION_DATA not found: ${DIFFUSION_DATA}"
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
echo "Training Configuration (E3/E4 mixed):"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${UPDATE_FREQ}"
echo "  Effective batch size: $((BATCH_SIZE * UPDATE_FREQ * NUM_GPUS))"
echo "  Workers: ${NUM_WORKERS}"
echo "  ProGAN data: ${TRAIN_DATA}"
echo "  Diffusion data: ${DIFFUSION_DATA}"
echo "  Mix ratio: ${MIX_RATIO}"
if [ -n "${DEDUP_REFERENCE_PATH}" ]; then
  echo "  Dedup refs: ${DEDUP_REFERENCE_PATH}"
  echo "  Dedup mode: ${DEDUP_MODE}"
fi
echo "=========================================="

EXTRA_ARGS=()
if [ -n "${DEDUP_REFERENCE_PATH}" ]; then
  EXTRA_ARGS+=(--dedup_reference_path "${DEDUP_REFERENCE_PATH}" --dedup_mode "${DEDUP_MODE}")
fi

if [ "${NUM_GPUS}" -eq 1 ]; then
  python main_finetune.py \
    --model AIDE_3BRANCH \
    --data_path "${TRAIN_DATA}" \
    --diffusion_path "${DIFFUSION_DATA}" \
    --mix_ratio "${MIX_RATIO}" \
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
    --skip_pretrained True \
    --npr_proj_dim 128 \
    --npr_branch_dropout "${NPR_BRANCH_DROPOUT}" \
    --save_ckpt True \
    --save_ckpt_freq "${SAVE_FREQ}" \
    "${EXTRA_ARGS[@]}" \
    "${PY_ARGS[@]}"
else
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 main_finetune.py \
    --model AIDE_3BRANCH \
    --data_path "${TRAIN_DATA}" \
    --diffusion_path "${DIFFUSION_DATA}" \
    --mix_ratio "${MIX_RATIO}" \
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
    --skip_pretrained True \
    --npr_proj_dim 128 \
    --npr_branch_dropout "${NPR_BRANCH_DROPOUT}" \
    --save_ckpt True \
    --save_ckpt_freq "${SAVE_FREQ}" \
    "${EXTRA_ARGS[@]}" \
    "${PY_ARGS[@]}"
fi

echo "Training finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/checkpoint-best.pth"
