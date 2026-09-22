"""固定采样两路相机，使用原训练渲染器输出 GT 对比；不加载全部图像。"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--per-camera', type=int, default=6)
a = p.parse_args()
project = Path('/data/oss_bucket_0/beifen/home/dujingrun.djr/code/3dgs/huanliufa_bushu')
data = Path('/data/oss_bucket_0/beifen/home/dujingrun.djr/code/3dgs/FastGS/data/perspective')
sys.path.insert(0, str(project))
import numpy as np
import torch
from PIL import Image, ImageDraw
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from scene.dataset_readers import readColmapCameras
from scene.gaussian_model import GaussianModel
from utils.camera_utils import loadCam
from gaussian_renderer import render_fastgs

torch.set_num_threads(4)
assert torch.cuda.is_available()
model = Path(a.model)
cloud = max(model.glob('point_cloud/iteration_*/point_cloud.ply'), key=lambda x:int(x.parent.name.split('_')[-1]))
out = Path(a.output)
for sub in ['gt', 'render', 'comparison', 'error']:
    (out/sub).mkdir(parents=True, exist_ok=True)
print('GPU:', torch.cuda.get_device_name(0), 'model:', cloud, flush=True)
g = GaussianModel(3)
g.load_ply(str(cloud))
print('Gaussians:',len(g.get_xyz),flush=True)
ext = read_extrinsics_binary(str(data/'sparse/0/images.bin'))
intr = read_intrinsics_binary(str(data/'sparse/0/cameras.bin'))
groups = defaultdict(list)
for key, value in ext.items(): groups[str(Path(value.name).parent)].append(key)
selected = []
for group, keys in sorted(groups.items()):
    keys.sort(key=lambda k:ext[k].name)
    for i in np.linspace(0,len(keys)-1,min(a.per_camera,len(keys)),dtype=int): selected.append(keys[i])
infos = readColmapCameras({k:ext[k] for k in selected}, intr, str(data/'images'))
args=SimpleNamespace(resolution=1,source_path=None,use_semantic_weight=False,data_device='cpu')
pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
rows=[]; tiles=[]
with torch.no_grad():
 for i,info in enumerate(infos):
    cam=loadCam(args,i,info,1.0)
    pred=render_fastgs(cam,g,pipe,torch.zeros(3,device='cuda'),0.6)['render'].clamp(0,1).cpu()
    gt=cam.original_image
    rel=Path(info.image_path).relative_to(data/'images')
    maskpath=data/'mask_sam3'/rel.with_suffix('.png')
    with Image.open(maskpath) as im:
        valid=np.asarray(im.convert('L').resize((cam.image_width,cam.image_height),Image.Resampling.NEAREST))<128
    diff=(pred-gt).numpy().transpose(1,2,0)
    mse=float(np.mean(diff[valid]**2)) if valid.any() else float('nan')
    psnr=-10*math.log10(max(mse,1e-12)) if valid.any() else None
    key=f'{i:02d}_{rel.parent.name}_{rel.stem}'
    def image(t): return Image.fromarray((t.numpy().transpose(1,2,0)*255).round().astype('uint8'))
    gi,ri=image(gt),image(pred)
    gi.save(out/'gt'/f'{key}.png');ri.save(out/'render'/f'{key}.png')
    err=np.clip(np.abs(diff).mean(2)*4,0,1);err[~valid]=0
    Image.fromarray((err*255).astype('uint8')).save(out/'error'/f'{key}.png')
    w,h=gi.size
    panel=Image.new('RGB',(w*2,h+48),(25,25,25)); panel.paste(gi,(0,48));panel.paste(ri,(w,48))
    draw=ImageDraw.Draw(panel);draw.text((8,5),f'GT | {rel}',fill='white');draw.text((w+8,5),f'Render | valid-region PSNR {psnr:.2f} dB',fill='white')
    panel.save(out/'comparison'/f'{key}.jpg',quality=95)
    thumb=panel.copy();thumb.thumbnail((900,420));tiles.append(thumb)
    rows.append(dict(image=str(rel),colmap_image_id=selected[i],width=w,height=h,valid_fraction=float(valid.mean()),valid_psnr=psnr,valid_l1=float(np.abs(diff[valid]).mean()),comparison=f'comparison/{key}.jpg'))
    print(f'[{i+1}/{len(infos)}] {rel} PSNR={psnr:.2f}',flush=True)
canvas=Image.new('RGB',(1800,420*math.ceil(len(tiles)/2)),(45,45,45))
for i,tile in enumerate(tiles):canvas.paste(tile,((i%2)*900,(i//2)*420))
canvas.save(out/'overview.jpg',quality=92)
with (out/'metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(out/'manifest.json').write_text(json.dumps(dict(model=str(cloud),gaussians=len(g.get_xyz),selection='Uniform indices in filename order, per image parent directory; training views, not held-out evaluation',mult=.6,background='black',metrics='Unweighted RGB PSNR/L1 over non-person hard-mask pixels; render clamped [0,1]',mean_psnr=float(np.mean([r['valid_psnr'] for r in rows])),views=rows),indent=2))
(out/'index.html').write_text('<meta charset="utf-8"><title>GT / Render</title><h1>Training-view comparison: GT left, render right</h1><p>Metrics exclude person masks. Not held-out evaluation.</p>'+''.join(f'<p>{r["image"]}: PSNR {r["valid_psnr"]:.2f} dB</p><img style="max-width:100%" src="{r["comparison"]}">' for r in rows))
print('Saved:',out,flush=True)
