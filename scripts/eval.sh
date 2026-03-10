PY_ARGS=${@:1}  # Any other arguments

# Single-GPU default. For baseline evaluation, override with: --model AIDE_2BRANCH
python main_finetune.py \
    --model AIDE_3BRANCH \
    --batch_size 8 \
    --blr 1e-4 \
    --epochs 5 \
    ${PY_ARGS}
