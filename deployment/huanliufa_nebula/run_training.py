"""为原训练脚本提供 Logview 进度条，不修改 OSS 中的训练代码。"""
import os
from pathlib import Path
import runpy
import sys
import time


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def install_progress():
    import tqdm
    original = tqdm.tqdm
    interval = max(1, float(os.environ.get('PROGRESS_SECONDS', '30')))

    class TrainingProgress(original):
        def __init__(self, *args, **kwargs):
            self._last_report = 0.0
            self._reported_n = None
            self._wall_start = time.monotonic()
            super().__init__(*args, **kwargs)

        def display(self, msg=None, pos=None):
            now = time.monotonic()
            finished = self.total is not None and self.n >= self.total
            if self._reported_n == self.n and finished:
                return
            if now - self._last_report < interval and not finished:
                return
            self._last_report = now
            self._reported_n = self.n
            elapsed = now - self._wall_start
            total = self.total or 0
            fraction = min(1.0, self.n / total) if total else 0.0
            filled = int(30 * fraction)
            bar = '#' * filled + '-' * (30 - filled)
            speed = self.n / elapsed if elapsed else 0
            eta = duration((total - self.n) / speed) if speed else '--:--:--'
            self.fp.write(
                f'{self.desc} [{bar}] {fraction:.1%} {self.n}/{total} '
                f'已用 {duration(elapsed)} 剩余约 {eta} '
                f'{speed:.2f} it/s {self.postfix or ""}\n')
            self.fp.flush()

    TrainingProgress.monitor_interval = 0
    tqdm.tqdm = TrainingProgress


def main():
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    install_progress()
    from wandb_tracking import Tracker
    tracker = Tracker(config={'architecture': 'FastGS', 'dataset': 'perspective',
                              'tracking_stage': 'loss', 'loss_log_interval': 100,
                              'nebula_job_name': os.environ.get('NEBULA_JOB_NAME', ''),
                              'output_path': os.environ.get('OUTPUT_PATH', '')})
    started = time.monotonic()
    exit_code = 0
    print('[训练] 正在加载代码、相机、图像和点云；初始化结束后显示迭代进度。', flush=True)
    try:
        from camera_loading import install_camera_loading
        install_camera_loading()
        from loss_tracking import track_loss_csv
        with track_loss_csv(tracker):
            runpy.run_path(str(script), run_name='__main__')
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        raise
    except BaseException:
        exit_code = 1
        raise
    finally:
        tracker.finish(exit_code=exit_code)
        print(f'[训练进程] 已用时间（含数据初始化）{duration(time.monotonic() - started)}', flush=True)


if __name__ == '__main__':
    main()
