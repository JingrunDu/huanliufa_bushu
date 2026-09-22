"""W&B loss 曲线及运行生命周期。"""
import json
import os
from pathlib import Path
import time
from wandb_auth import api_key

SDK_REQUIREMENT = 'wandb==0.25.0'


class Tracker:
    def __init__(self, config=None, name=None, job_type='train', strict=False):
        self.run = None
        self.started = time.monotonic()
        self.strict = strict
        self.output = Path(os.environ.get('OUTPUT_PATH', '/tmp/huanliufa_wandb'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.mode = os.environ.get('WANDB_MODE', 'online')
        if self.mode == 'disabled':
            print('[W&B] 已关闭', flush=True)
            return
        try:
            import wandb
        except ImportError:
            if strict:
                raise
            print('[W&B] SDK 不可用，继续训练并保留 CSV', flush=True)
            return
        local = Path(os.environ.get('WANDB_DIR', '/tmp/huanliufa_wandb'))
        local.mkdir(parents=True, exist_ok=True)
        try:
            key = api_key()
            if self.mode == 'online' and not key:
                raise RuntimeError('W&B credentials missing')
            self.run = wandb.init(
                entity=os.environ.get('WANDB_ENTITY', 'jingrundu-tsinghua-university'),
                project=os.environ.get('WANDB_PROJECT', 'huanliufa_bushu'),
                name=name or self.output.name,
                job_type=job_type,
                mode=self.mode,
                dir=str(local),
                config=config or {},
                settings=wandb.Settings(api_key=key, init_timeout=20, save_code=False, disable_git=True, x_disable_stats=True),
            )
        except Exception as exc:
            if strict:
                raise
            print(f'[W&B] 在线初始化失败 ({type(exc).__name__})，尝试离线记录；训练继续', flush=True)
            self.mode = 'offline'
            try:
                self.run = wandb.init(
                    entity=os.environ.get('WANDB_ENTITY', 'jingrundu-tsinghua-university'),
                    project=os.environ.get('WANDB_PROJECT', 'huanliufa_bushu'),
                    name=name or self.output.name, job_type=job_type, mode='offline',
                    dir=str(local), config=config or {},
                    settings=wandb.Settings(init_timeout=20, save_code=False, disable_git=True, x_disable_stats=True),
                )
            except Exception as offline_exc:
                print(f'[W&B] 离线初始化失败 ({type(offline_exc).__name__})，仅保留原有日志', flush=True)
                return
        self.url = self.run.url if self.mode == 'online' else None
        print(f'[W&B] mode={self.mode}, run_id={self.run.id}, URL={self.url or "离线"}', flush=True)
        self.write_status('running')
        self.run.define_metric('train/iteration')
        self.run.define_metric('train/loss', step_metric='train/iteration')
        self.log({'pipeline/started': 1})

    def write_status(self, state):
        try:
            payload = {'mode': self.mode, 'id': self.run.id, 'url': self.url,
                       'state': state, 'local_dir': str(self.run.dir)}
            (self.output / 'wandb_run.json').write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            if self.strict:
                raise
            print(f'[W&B] 无法保存运行信息 ({type(exc).__name__})', flush=True)

    def log(self, metrics):
        if self.run:
            try:
                self.run.log(metrics)
            except Exception as exc:
                if self.strict:
                    raise
                print(f'[W&B] 指标发送失败 ({type(exc).__name__})，训练继续', flush=True)

    def finish(self, exit_code=0):
        if not self.run:
            return
        try:
            self.run.summary['pipeline/elapsed_seconds'] = time.monotonic() - self.started
            self.run.summary['pipeline/exit_code'] = exit_code
            self.log({'pipeline/finished': int(exit_code == 0)})
            self.run.finish(exit_code=exit_code)
            self.write_status('finished' if exit_code == 0 else 'failed')
            # 结束后备份 SDK 文件，便于离线同步及检查；不上传模型 PLY。
            import shutil
            source = Path(self.run.dir).parent
            shutil.copytree(source, self.output / 'wandb' / source.name, dirs_exist_ok=True)
        except Exception as exc:
            if self.strict:
                raise
            print(f'[W&B] 结束或备份失败 ({type(exc).__name__})，原训练退出码保持不变', flush=True)
