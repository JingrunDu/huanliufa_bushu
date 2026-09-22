"""一次性并行读取 OSS 环境，制作可校验的 tar 包并上传。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tarfile
import threading
import time
import oss2

SOURCE = 'beifen/envs/tmp/fmc/fastgs/'
DEST = 'beifen/data/3dgs_bushu/env/fastgs_py310_v1.tar'
WORK = Path('/tmp/fastgs_env_package')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    endpoint = os.environ['FMC_ENDPOINT']
    if '://' not in endpoint:
        endpoint = 'https://' + endpoint
    bucket = oss2.Bucket(oss2.Auth(os.environ['FMC_AK'], os.environ['FMC_SK']), endpoint, 'fmc', connect_timeout=60)
    print('[索引] 枚举环境文件……', flush=True)
    objects = []
    for item in oss2.ObjectIterator(bucket, prefix=SOURCE, max_keys=1000):
        if not item.key.endswith('/'):
            rel = item.key[len(SOURCE):]
            if rel.startswith('/') or '..' in Path(rel).parts:
                raise ValueError('Unexpected object path')
            objects.append((item.key, rel, item.size))
        if len(objects) and len(objects) % 10000 == 0:
            print(f'[索引] 已发现 {len(objects):,} 文件', flush=True)
    total = sum(x[2] for x in objects)
    print(f'[索引] {len(objects):,} 文件，共 {total/1024**3:.2f} GiB', flush=True)
    if args.inspect:
        return
    if not objects:
        raise RuntimeError('Source is empty')
    if bucket.object_exists(DEST + '.json'):
        raise RuntimeError('Archive manifest already exists; refusing to overwrite')
    root = WORK / 'environment'
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    done = byte_count = 0
    started = last_report = time.monotonic()

    def fetch(entry):
        nonlocal done, byte_count, last_report
        key, rel, size = entry
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != size:
            temporary = target.with_name(target.name + '.download-part')
            for attempt in range(3):
                try:
                    bucket.get_object_to_file(key, str(temporary))
                    if temporary.stat().st_size != size:
                        raise IOError('size mismatch')
                    temporary.replace(target)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        with lock:
            done += 1
            byte_count += size
            now = time.monotonic()
            if now - last_report >= 15 or done == len(objects):
                print(f'[读取] {done:,}/{len(objects):,} 文件，{byte_count/1024**3:.2f}/{total/1024**3:.2f} GiB，已用 {now-started:.0f}s', flush=True)
                last_report = now

    with ThreadPoolExecutor(max_workers=32) as pool:
        pending = [pool.submit(fetch, obj) for obj in objects]
        for future in as_completed(pending):
            future.result()
    archive = WORK / 'fastgs_py310_v1.tar'
    print('[打包] 生成 tar，恢复 bin 下可执行文件权限……', flush=True)
    last_report = time.monotonic()
    with tarfile.open(archive, 'w') as tar:
        for index, (_, rel, _) in enumerate(objects, 1):
            target = root / rel
            info = tar.gettarinfo(str(target), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mode = 0o755 if rel.startswith('bin/') else 0o644
            with target.open('rb') as src:
                tar.addfile(info, src)
            now = time.monotonic()
            if now - last_report >= 15 or index == len(objects):
                print(f'[打包] {index:,}/{len(objects):,} 文件', flush=True)
                last_report = now
    print('[校验] 计算 SHA-256', flush=True)
    digest = hashlib.sha256()
    with archive.open('rb') as src:
        for chunk in iter(lambda: src.read(16 * 1024**2), b''):
            digest.update(chunk)
    # 完整遍历验证 tar 结构及条目数量。
    with tarfile.open(archive, 'r:') as tar:
        count = sum(1 for member in tar if member.isfile())
    if count != len(objects):
        raise RuntimeError('Archive member count mismatch')
    last_report = 0
    def progress(consumed, size):
        nonlocal last_report
        now = time.monotonic()
        with lock:
            if now-last_report >= 15 or consumed == size:
                print(f'[上传] {consumed/size:.1%} {consumed/1024**3:.2f}/{size/1024**3:.2f} GiB', flush=True)
                last_report = now
    oss2.resumable_upload(bucket, DEST, str(archive), multipart_threshold=16*1024**2,
                         part_size=32*1024**2, num_threads=8, progress_callback=progress)
    remote_size = bucket.head_object(DEST).content_length
    if remote_size != archive.stat().st_size:
        raise RuntimeError('Uploaded archive size mismatch')
    manifest = {'archive': DEST, 'size': remote_size, 'sha256': digest.hexdigest(),
                'file_count': len(objects), 'unpacked_bytes': total, 'source': SOURCE}
    bucket.put_object(DEST + '.json', json.dumps(manifest, indent=2).encode())
    print('[完成] oss://fmc/' + DEST, flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
