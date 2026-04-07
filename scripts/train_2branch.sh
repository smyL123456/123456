#!/usr/bin/env bash
# Train AIDE_2BRANCH baseline on single or multi-GPU.
# Usage:
#   Single GPU:  bash scripts/train_2branch.sh
#   Dual GPU:    NUM_GPUS=2 bash scripts/train_2branch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Paths (same as train_3branch.sh)
TRAIN_DATA="/data/CNNSpot/progan_train"
EVAL_DATA="/data/CNNSpot/progan_val"
RESNET_PATH="/AIGCDetect/models/123456/pretrained_ckpts/resnet50.pth"
CONVNEXT_PATH="/AIGCDetect/models/123456/pretrained_ckpts/open_clip_pytorch_model.bin"
OUTPUT_DIR="/AIGCDetect/models/123456/results/2branch_train"

NUM_GPUS=${NUM_GPUS:-1}

if [ "${NUM_GPUS}" -eq 1 ]; then
  BATCH_SIZE=16
  UPDATE_FREQ=6
  NUM_WORKERS=8
else
  BATCH_SIZE=48
  UPDATE_FREQ=2
  NUM_WORKERS=16
fi

BLR=7.5e-5
EPOCHS=6
WARMUP_EPOCHS=1
WEIGHT_DECAY=0.05
SAVE_FREQ=1

PY_ARGS=("$@")

for p in "${TRAIN_DATA}" "${EVAL_DATA}" "${RESNET_PATH}" "${CONVNEXT_PATH}"; do
  if [ ! -e "${p}" ]; then
    echo "[ERROR] Not found: ${p}"; exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Training AIDE_2BRANCH"
echo "  GPUs: ${NUM_GPUS}  Batch: ${BATCH_SIZE}  Effective: $((BATCH_SIZE * UPDATE_FREQ * NUM_GPUS))"
echo "=========================================="

COMMON_ARGS=(
  --model AIDE_2BRANCH
  --data_path "${TRAIN_DATA}"
  --eval_data_path "${EVAL_DATA}"
  --resnet_path "${RESNET_PATH}"
  --convnext_path "${CONVNEXT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --update_freq "${UPDATE_FREQ}"
  --blr "${BLR}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --weight_decay "${WEIGHT_DECAY}"
  --epochs "${EPOCHS}"
  --num_workers "${NUM_WORKERS}"
  --use_amp True
  --save_ckpt True
  --save_ckpt_freq "${SAVE_FREQ}"
  "${PY_ARGS[@]}"
)

if [ "${NUM_GPUS}" -eq 1 ]; then
  python main_finetune.py "${COMMON_ARGS[@]}"
else
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29501 main_finetune.py "${COMMON_ARGS[@]}"
fi
