"""并行准备图像/遮罩，主线程构造 CUDA 相机，输出加载进度。"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
from pathlib import Path
import threading
import time
import types
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def separable_blur(tensor, kernel_size, sigma):
    # 与原二维高斯核和零填充数学等价，减少 43×43 卷积的 CPU 开销。
    coords = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device) - kernel_size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()
    channels = tensor.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    tensor = F.conv2d(tensor, horizontal, padding=(0, kernel_size // 2), groups=channels)
    return F.conv2d(tensor, vertical, padding=(kernel_size // 2, 0), groups=channels)


def load_mask(image_path, resolution, source_path, soft_mask_sigma=None):
    sigma = float(os.environ.get('MASK_SIGMA', '0')) if soft_mask_sigma is None else soft_mask_sigma
    rel = Path(os.path.relpath(image_path, Path(source_path) / 'images')).with_suffix('.png')
    path = Path(source_path) / 'mask_sam3' / rel
    with Image.open(path) as image:
        resized = image.convert('L').resize(resolution, Image.NEAREST if sigma <= 0 else Image.BILINEAR)
        alpha = 1 - np.asarray(resized, dtype=np.float32) / 255.0
    if sigma <= 0:
        alpha = (alpha >= 0.5).astype(np.float32)
    tensor = torch.from_numpy(alpha).unsqueeze(0)
    if sigma > 0:
        kernel_size = int(6 * sigma + 1)
        kernel_size += (kernel_size % 2 == 0)
        tensor = separable_blur(tensor.unsqueeze(0), kernel_size, sigma).squeeze(0).clamp(0, 1)
    return tensor


def resident_gib():
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return 0.0


def load_parallel(cam_infos, resolution_scale, args, prepare, construct):
    total = len(cam_infos)
    if not total:
        return []
    workers = max(1, int(os.environ.get('CAMERA_LOAD_WORKERS', '8')))
    interval = max(1, float(os.environ.get('PROGRESS_SECONDS', '30')))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(os.environ.get('CAMERA_TORCH_THREADS', '1'))))
    started = time.monotonic()
    state = {'done': 0, 'waiting': ''}
    stopped = threading.Event()
    result = [None] * total

    def report():
        elapsed = max(time.monotonic() - started, 0.001)
        done = state['done']; rate = done / elapsed
        eta = f'{(total-done)/rate:.0f}s' if rate else '未知'
        print(f'[相机加载] {done}/{total} ({done/total:.1%})，已用 {elapsed:.0f}s，'
              f'剩余约 {eta}，{rate:.2f} 张/s，进程内存 {resident_gib():.2f} GiB'
              f'，等待: {state["waiting"]}', flush=True)

    def heartbeat():
        while not stopped.wait(interval):
            report()

    monitor = threading.Thread(target=heartbeat, daemon=True)
    print(f'[相机加载] 开始，{workers} 路并行，mask_sigma={os.environ.get("MASK_SIGMA", "0")}', flush=True)
    monitor.start()
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    next_index = 0
    succeeded = False
    try:
        while next_index < total or pending:
            while next_index < total and len(pending) < workers * 2:
                info = cam_infos[next_index]
                future = pool.submit(prepare, args, next_index, info, resolution_scale)
                pending[future] = next_index
                next_index += 1
            state['waiting'] = cam_infos[min(pending.values())].image_path
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    kwargs = future.result()
                    # 仅主线程操作 CUDA；工作线程只做 OSS 读取和 CPU 图像转换。
                    result[index] = construct(**kwargs)
                except Exception as exc:
                    raise RuntimeError(f'加载相机失败: {cam_infos[index].image_path}') from exc
                state['done'] += 1
        state['waiting'] = '无'
        report()
        succeeded = True
        return result
    finally:
        stopped.set(); monitor.join()
        pool.shutdown(wait=True, cancel_futures=True)
        torch.set_num_threads(previous_threads)
        if not succeeded:
            print('[相机加载] 失败，见异常中的具体图像路径', flush=True)


def install_camera_loading():
    import scene
    import utils.camera_utils as camera_utils
    # 保留原 loadCam 的尺寸、语义权重和 alpha 逻辑，仅将最后的 Camera
    # 构造替换为参数收集，使 CPU 工作可安全并行，CUDA 构造移到主线程。
    prepare_globals = dict(camera_utils.loadCam.__globals__)
    prepare_globals.update(Camera=lambda **kwargs: kwargs, load_external_mask=load_mask)
    prepare = types.FunctionType(camera_utils.loadCam.__code__, prepare_globals,
                                 name='prepare_camera', argdefs=camera_utils.loadCam.__defaults__)
    construct = camera_utils.Camera
    def camera_list(infos, scale, args):
        return load_parallel(infos, scale, args, prepare, construct)
    camera_utils.cameraList_from_camInfos = camera_list
    scene.cameraList_from_camInfos = camera_list
