"""恢复本地 FastGS 环境，直接读写 OSS 挂载，单卡训练。"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from environment_archive import DEFAULT_ARCHIVE, restore_environment


def run_stage(command, label, **kwargs):
    started = time.monotonic()
    print(f'[{label}] 开始', flush=True)
    with subprocess.Popen(command, **kwargs) as proc:
        while True:
            try:
                code = proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                print(f'[{label}] 进行中，已用 {now - started:.0f} 秒', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, command)
    print(f'[{label}] 完成，用时 {time.monotonic() - started:.0f} 秒', flush=True)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        raise ValueError('run-name 只能包含字母、数字、下划线或连字符')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('请使用单 worker、单 GPU')
    mount = Path(os.environ.get('OSS_MOUNT', '/data/oss_bucket_0'))
    project = mount / 'beifen/home/dujingrun.djr/code/3dgs/huanliufa_bushu'
    data = mount / 'beifen/home/dujingrun.djr/code/3dgs/FastGS/data/perspective'
    archive = mount / os.environ.get('ENV_ARCHIVE_REL', DEFAULT_ARCHIVE)
    output = mount / 'beifen/data/3dgs_bushu' / args.run_name
    for path in [project / 'train.py', data / 'images', data / 'sparse/0',
                 data / 'mask_sam3', data / 'semantic_weights', archive, Path(str(archive) + '.json')]:
        if not path.exists():
            raise FileNotFoundError(f'缺少 {path}，请检查 OSS 挂载和数据路径')
    target = Path('/tmp/fmc/fastgs')
    python = restore_environment(archive, target)
    env = os.environ.copy()
    for key in ['PYTHONHOME', 'PYTHONPATH']:
        env.pop(key, None)
    env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               DATA_PATH=str(data), OUTPUT_PATH=str(output))
    env['PATH'] = str(target / 'bin') + os.pathsep + env.get('PATH', '')
    # 只补缺失的 Python 辅助包；torch 和 CUDA 扩展沿用备份版本。
    dependencies = {'plyfile': 'plyfile', 'tqdm': 'tqdm', 'PIL': 'Pillow',
                    'websockets': 'websockets', 'tensorboard': 'tensorboard'}
    probe = 'import importlib.util,json; print(json.dumps([p for m,p in ' + repr(dependencies) + '.items() if importlib.util.find_spec(m) is None]))'
    import json
    missing = json.loads(subprocess.check_output([python, '-c', probe], env=env, text=True))
    if missing:
        run_stage([python, '-m', 'pip', 'install', *missing], '补充缺失依赖', env=env)
    check = (
        'import torch, torchvision, plyfile, tqdm, PIL, websockets, tensorboard; '
        'import fused_ssim, diff_gaussian_rasterization_fastgs, simple_knn._C; '
        'assert torch.cuda.is_available(), "CUDA unavailable"; '
        'print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'
    )
    run_stage([python, '-u', '-c', check], 'GPU 与扩展检查', env=env)
    env['TRAIN_PYTHON'] = python
    auth_file = Path(__file__).resolve().with_name('.wandb_auth.json')
    if auth_file.exists():
        env['WANDB_AUTH_FILE'] = str(auth_file)
    env['FASTGS_PROJECT'] = str(project)
    output.mkdir(parents=True, exist_ok=True)
    print(f'数据: {data}\n输出: {output}', flush=True)
    launcher = Path(__file__).resolve().with_name('train_6650.sh')
    result = subprocess.run(['bash', '-o', 'pipefail', str(launcher)], cwd=project, env=env)
    print(f'[任务] 退出码 {result.returncode}，总用时（含环境准备）{time.monotonic() - started:.0f} 秒', flush=True)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
