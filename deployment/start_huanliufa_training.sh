#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/data/oss_bucket_0/beifen/home/dujingrun.djr/code/3dgs/huanliufa_bushu
ENV_ARCHIVE=${ENV_ARCHIVE:-/data/oss_bucket_0/beifen/data/3dgs_bushu/env/fastgs_py310_v1.tar}
ARCHIVE_HELPER=/home/dujingrun.djr/huanliufa_nebula/environment_archive.py
TRAIN_ENV=${TRAIN_ENV:-/tmp/fmc/fastgs}
export DATA_PATH=${DATA_PATH:-/data/oss_bucket_0/beifen/home/dujingrun.djr/code/3dgs/FastGS/data/perspective}
# OSS 挂载目录：oss://fmc/beifen/data/3dgs_bushu/
export OUTPUT_PATH=${OUTPUT_PATH:-/data/oss_bucket_0/beifen/data/3dgs_bushu/huanliufa_$(date +%Y%m%d_%H%M%S)_$$}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

if [[ ${1:-} != --worker ]]; then
    mkdir -p "$OUTPUT_PATH"
    nohup bash "$(readlink -f "$0")" --worker > "$OUTPUT_PATH/launcher.log" 2>&1 < /dev/null &
    job_pid=$!
    echo "$job_pid" > "$OUTPUT_PATH/launcher.pid"
    echo "后台任务 PID: $job_pid（先准备环境，再训练）"
    echo "输出目录: $OUTPUT_PATH"
    echo "查看进度: tail -f '$OUTPUT_PATH/launcher.log'"
    exit 0
fi

trap 'echo "启动或训练失败，行号 $LINENO，请查看上方错误。" >&2' ERR

[[ -f "$PROJECT/train.py" ]]
[[ -d "$DATA_PATH/images" && -d "$DATA_PATH/sparse/0" ]]

# 单文件传输、SHA-256 校验、本地解压；完成后才启用环境。
python3 "$ARCHIVE_HELPER" --archive "$ENV_ARCHIVE" --target "$TRAIN_ENV"

export PATH="$TRAIN_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
PYTHON="$TRAIN_ENV/bin/python"

"$PYTHON" -m pip --version
if ! "$PYTHON" -c 'import plyfile'; then
    "$PYTHON" -m pip install plyfile
fi

echo "检查 GPU 和 FastGS 扩展……"
"$PYTHON" -u - <<'PY'
import sys
import torch
import torchvision
import numpy
import plyfile
import tqdm
import PIL
import websockets
import fused_ssim
import diff_gaussian_rasterization_fastgs
import simple_knn._C
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA 不可用，请检查 GPU 和驱动'
print('GPU:', torch.cuda.get_device_name(0))
print('依赖检查通过')
PY

cd "$PROJECT"
echo "开始训练，数据: $DATA_PATH"
echo "模型输出: $OUTPUT_PATH"
# pipefail 使 Python 训练失败时返回失败状态，而非 tee 的成功状态。
export TRAIN_PYTHON="$PYTHON" FASTGS_PROJECT="$PROJECT"
exec bash -o pipefail /home/dujingrun.djr/huanliufa_nebula/train_6650.sh
