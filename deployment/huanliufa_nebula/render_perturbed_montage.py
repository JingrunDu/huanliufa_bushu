"""Render reproducible perturbations of the previously selected 100 cameras."""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args()
sys.path.insert(0,'/data/oss_bucket_0/beifen/home/dujingrun.djr/code/3dgs/huanliufa_bushu')
import numpy as np
import torch
from PIL import Image, ImageDraw
from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary,qvec2rotmat
from scene.dataset_readers import readColmapCameras
from scene.gaussian_model import GaussianModel
from utils.camera_utils import loadCam
from gaussian_renderer import render_fastgs

base=Path('/home/dujingrun.djr/render_results/huanliufa_20260922_051004_100')
manifest=json.loads((base/'manifest.json').read_text())
data=Path('/data/oss_bucket_0/beifen/home/dujingrun.djr/code/3dgs/FastGS/data/perspective')
out=Path(a.output)
for name in ['render']: (out/name).mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
assert torch.cuda.is_available()
extr=read_extrinsics_binary(str(data/'sparse/0/images.bin'))
intr=read_intrinsics_binary(str(data/'sparse/0/cameras.bin'))
# Trajectory scale matches the normalization radius used by the training loader.
centers=np.array([-qvec2rotmat(e.qvec).T @ e.tvec for e in extr.values()])
radius=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
translation=radius*.01
rng=np.random.default_rng(20260922)
model=GaussianModel(3);model.load_ply(manifest['model'])
print('GPU:',torch.cuda.get_device_name(0),'radius:',radius,'translation:',translation,flush=True)
ids=[v['colmap_image_id'] for v in manifest['views']]
assert len(ids)==len(set(ids))==100
infos=readColmapCameras({k:extr[k] for k in ids},intr,str(data/'images'))
args=SimpleNamespace(resolution=1,source_path=None,use_semantic_weight=False,data_device='cpu')
pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
rows=[];tiles=[]
with torch.no_grad():
 for i,(info,ref) in enumerate(zip(infos,manifest['views'])):
    # R stored by FastGS is camera-to-world rotation. Use local yaw/pitch,
    # and a random lateral displacement (no roll or forward translation).
    yaw,pitch=np.deg2rad(rng.uniform(-5,5,2))
    cy,sy=np.cos(yaw),np.sin(yaw);cx,sx=np.cos(pitch),np.sin(pitch)
    ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    theta=rng.uniform(0,2*np.pi)
    local=np.array([np.cos(theta),np.sin(theta),0])*translation
    center=-info.R @ info.T
    moved=center+info.R @ local
    rotation=info.R @ ry @ rx
    t=-rotation.T @ moved
    assert np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
    assert np.allclose(-rotation@t,moved,atol=1e-6)
    cam=loadCam(args,i,info._replace(R=rotation,T=t),1.0)
    rendered=render_fastgs(cam,model,pipe,torch.zeros(3,device='cuda'),.6)['render'].clamp(0,1).cpu()
    ri=Image.fromarray((rendered.numpy().transpose(1,2,0)*255).round().astype('uint8'))
    stem=Path(ref['comparison']).stem
    ri.save(out/'render'/f'{stem}.png')
    w,h=ri.size
    tiles.append(ri.copy())
    rows.append(dict(source_image=ref['image'],colmap_image_id=ids[i],yaw_deg=float(np.rad2deg(yaw)),pitch_deg=float(np.rad2deg(pitch)),translation_local=local.tolist(),original_camera_center=center.tolist(),perturbed_camera_center=moved.tolist(),original_R=info.R.tolist(),original_T=info.T.tolist(),perturbed_R=rotation.tolist(),perturbed_T=t.tolist(),width=w,height=h,fovx=float(info.FovX),fovy=float(info.FovY),render=f'render/{stem}.png'))
    print(f'[{i+1}/100] {ref["image"]}',flush=True)
# Preserve aspect ratios and all pixels; no cropping/stretching.
cell_w=max(im.width for im in tiles)
cell_h=max(im.height for im in tiles)
canvas=Image.new('RGB',(10*cell_w,10*cell_h),(20,20,20))
for i,im in enumerate(tiles):
    x=(i%10)*cell_w+(cell_w-im.width)//2
    y=(i//10)*cell_h+(cell_h-im.height)//2
    canvas.paste(im,(x,y))
canvas.save(out/'montage_100.jpg',quality=95,subsampling=0)
preview=canvas.copy();preview.thumbnail((2400,2400));preview.save(out/'preview.jpg',quality=90)
(out/'poses.json').write_text(json.dumps(dict(model=manifest['model'],seed=20260922,trajectory_radius=radius,translation_length=translation,translation_fraction=.01,yaw_pitch_range_deg=[-5,5],rotation_convention='R camera-to-world, T world-to-camera translation; R_new=R*Ry*Rx; center_new=center+R*local_translation',mult=.6,views=rows),indent=2))
(out/'index.html').write_text('<meta charset="utf-8"><h1>100 perturbed renders</h1><a href="montage_100.jpg"><img style="max-width:100%" src="preview.jpg"></a>')
(out/'README.md').write_text('100 renders in a 10x10 montage, row-major order matching poses.json. No GT images.\nSame seed and cameras as previous run; yaw/pitch uniform [-5,5] degrees, lateral translation 1% of trajectory radius.\nOriginal image sizes and aspect ratios preserved; dark padding for mixed orientations.\nCOLMAP units are not meters without calibration.\n')
print('Saved',out,'montage size:',canvas.size,flush=True)
