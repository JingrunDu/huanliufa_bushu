#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/training_config.sh"
PROJECT=${FASTGS_PROJECT:-/data/oss_bucket_0/beifen/home/dujingrun.djr/code/3dgs/huanliufa_bushu}
: "${DATA_PATH:?缺少 DATA_PATH}"
: "${OUTPUT_PATH:?缺少 OUTPUT_PATH}"
PYTHON=${TRAIN_PYTHON:-/tmp/fmc/fastgs/bin/python}
export CAMERA_LOAD_WORKERS CAMERA_TORCH_THREADS MASK_SIGMA
export PYTHONUNBUFFERED=1
export WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_BASE_URL
if [[ "$WANDB_MODE" != disabled ]]; then
  if ! "$PYTHON" -c 'import wandb'; then
    echo '[W&B] 安装 SDK'
    if ! "$PYTHON" -m pip install --timeout 20 --retries 1 wandb==0.25.0; then
      echo '[W&B] SDK 安装失败，保留原有训练日志并继续'
      export WANDB_MODE=disabled
    fi
  fi
fi
mkdir -p "$OUTPUT_PATH"
"$PYTHON" - "$DATA_PATH" "$NUM_IMAGES" <<'PY'
import struct,sys
from pathlib import Path
with (Path(sys.argv[1])/'sparse/0/images.bin').open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
assert n==int(sys.argv[2]), f'COLMAP 注册图像数 {n} 与 NUM_IMAGES={sys.argv[2]} 不一致'
print(f'[训练配置] 全部 {n} 张图像参与训练，无验证集划分',flush=True)
PY
ARGS=(
  -s "$DATA_PATH" -i images -m "$OUTPUT_PATH" -r 1
  --iterations "$ITERATIONS" --position_lr_max_steps "$POSITION_LR_STEPS"
  --densification_interval "$DENSIFY_INTERVAL"
  --densify_from_iter "$DENSIFY_FROM" --densify_until_iter "$DENSIFY_UNTIL"
  --opacity_reset_interval "$OPACITY_RESET_INTERVAL"
  --densify_grad_threshold 0.0002 --grad_thresh 0.0002
  --grad_abs_thresh 0.0003 --percent_dense 0.003 --dense 0.003
  --optimizer_type default --opacity_lr 0.025 --highfeature_lr 0.025
  --lambda_dssim 0.4 --mult 0.6 --loss_thresh 0.05
  --freq_weight_alpha "$FREQ_WEIGHT_ALPHA" --use_ms_ssim
  --use_semantic_weight
  --semantic_equipment_weight "$SEMANTIC_EQUIPMENT_WEIGHT"
  --semantic_background_weight "$SEMANTIC_BACKGROUND_WEIGHT"
  --densify_equipment_factor 1.0
  --progressive_prune_interval "$PROGRESSIVE_PRUNE_INTERVAL"
  --test_iterations -1
  --save_iterations "$ITERATIONS"
  --data_device cpu
)
cp "$SCRIPT_DIR/training_config.sh" "$OUTPUT_PATH/training_config.sh"
printf '%q ' "$PYTHON" "$SCRIPT_DIR/run_training.py" "$PROJECT/train.py" "${ARGS[@]}" > "$OUTPUT_PATH/training_command.txt"
printf '\n' >> "$OUTPUT_PATH/training_command.txt"
echo "[训练配置] iterations=$ITERATIONS ($NUM_IMAGES × $EPOCHS), densify=$DENSIFY_FROM..$DENSIFY_UNTIL / $DENSIFY_INTERVAL, opacity_reset=$OPACITY_RESET_INTERVAL"
cd "$PROJECT"
"$PYTHON" -u "$SCRIPT_DIR/run_training.py" "$PROJECT/train.py" "${ARGS[@]}" 2>&1 | tee "$OUTPUT_PATH/train.log"
