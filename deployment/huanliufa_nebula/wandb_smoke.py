"""创建短暂测试 Run，验证初始化、日志上传和正常结束；不训练模型。"""
import json
import os
from pathlib import Path
import time
from wandb_tracking import Tracker


def main():
    name = 'connectivity_' + time.strftime('%Y%m%d_%H%M%S')
    os.environ.setdefault('OUTPUT_PATH', '/home/dujingrun.djr/wandb_checks/' + name)
    tracker = Tracker(config={'purpose': 'connectivity_check', 'training_started': False},
                      name=name, job_type='connectivity', strict=True)
    if tracker.run is None:
        raise RuntimeError('W&B run was not created')
    try:
        for step in range(3):
            tracker.log({'connectivity/step': step, 'connectivity/ok': 1})
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish()
    if tracker.mode == 'online':
        import wandb
        api = wandb.Api(timeout=20)
        remote = api.run('/'.join([tracker.run.entity, tracker.run.project, tracker.run.id]))
        assert remote.summary.get('connectivity/ok') == 1, 'Remote metric verification failed'
        print('[W&B smoke] ONLINE VERIFIED:', tracker.url, flush=True)
    else:
        print('[W&B smoke] OFFLINE VERIFIED', flush=True)


if __name__ == '__main__':
    main()
