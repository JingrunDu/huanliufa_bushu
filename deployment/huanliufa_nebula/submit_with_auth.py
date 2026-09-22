"""在临时提交目录中携带 W&B 凭据，避免在平台环境变量日志中暴露。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from wandb_auth import api_key


def submit(command):
    root = Path(__file__).resolve().parent
    mode = os.environ.get('WANDB_MODE', 'online')
    key = api_key() if mode == 'online' else None
    if mode == 'online' and not key:
        raise RuntimeError('未找到 W&B 登录凭据，请先 wandb login 或设置 WANDB_API_KEY；也可使用 WANDB_MODE=offline')
    with tempfile.TemporaryDirectory(prefix='huanliufa_submit_') as temporary:
        stage = Path(temporary)
        # 仅打包运行文件，不包含本地日志、历史提交包或其他认证文件。
        for source in root.iterdir():
            if source.is_file() and source.suffix in {'.py', '.sh', '.json', '.md'} and not source.name.startswith('.'):
                shutil.copy2(source, stage / source.name)
        if key:
            path = stage / '.wandb_auth.json'
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as handle:
                json.dump({'api_key': key}, handle)
        args = list(command)
        for index, arg in enumerate(args):
            if arg.startswith('--file.cluster_file='):
                source = Path(arg.split('=', 1)[1])
                if not source.is_absolute():
                    source = root / source
                shutil.copy2(source, stage / 'cluster.json')
                args[index] = '--file.cluster_file=./cluster.json'
        # 不把用户终端中的 API key 继承给 CLI；worker 在运行时读取认证文件。
        env = dict(os.environ)
        env.pop('WANDB_API_KEY', None)
        return subprocess.run(args, cwd=stage, env=env).returncode


if __name__ == '__main__':
    sys.exit(submit(sys.argv[1:]))
