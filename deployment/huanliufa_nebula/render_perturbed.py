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
for name in ['render','comparison','overview_pages']: (out/name).mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
assert torch.cuda.is_available()
extr=read_extrinsics_binary(str(data/'sparse/0/images.bin'))
intr=read_intrinsics_binary(str(data/'sparse/0/cameras.bin'))
# Trajectory scale matches the normalization radius used by the training loader.
centers=np.array([-qvec2rotmat(e.qvec).T @ e.tvec for e in extr.values()])
radius=float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1)
translation=radius*.005
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
    yaw,pitch=np.deg2rad(rng.uniform(-2,2,2))
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
    # Include the unperturbed render to separate fitting error from viewpoint effects.
    with Image.open(base/'gt'/f'{stem}.png') as im: gi=im.copy()
    with Image.open(base/'render'/f'{stem}.png') as im: original=im.copy()
    w,h=ri.size
    panel=Image.new('RGB',(3*w,h+50),(25,25,25))
    for j,im in enumerate([gi,original,ri]): panel.paste(im,(j*w,50))
    draw=ImageDraw.Draw(panel)
    for j,title in enumerate(['Original GT (reference only)','Original-view render',f'Perturbed yaw={np.rad2deg(yaw):.2f}, pitch={np.rad2deg(pitch):.2f} deg']):draw.text((j*w+5,8),title,fill='white')
    panel.save(out/'comparison'/f'{stem}.jpg',quality=95)
    panel.thumbnail((1200,420));tiles.append(panel)
    rows.append(dict(source_image=ref['image'],colmap_image_id=ids[i],yaw_deg=float(np.rad2deg(yaw)),pitch_deg=float(np.rad2deg(pitch)),translation_local=local.tolist(),original_camera_center=center.tolist(),perturbed_camera_center=moved.tolist(),original_R=info.R.tolist(),original_T=info.T.tolist(),perturbed_R=rotation.tolist(),perturbed_T=t.tolist(),width=w,height=h,fovx=float(info.FovX),fovy=float(info.FovY),render=f'render/{stem}.png',comparison=f'comparison/{stem}.jpg'))
    print(f'[{i+1}/100] {ref["image"]}',flush=True)
for start in range(0,100,10):
    canvas=Image.new('RGB',(2400,2100),(35,35,35))
    for i,im in enumerate(tiles[start:start+10]):canvas.paste(im,((i%2)*1200,(i//2)*420))
    canvas.save(out/'overview_pages'/f'{start//10+1:02d}.jpg',quality=92)
(out/'poses.json').write_text(json.dumps(dict(model=manifest['model'],seed=20260922,trajectory_radius=radius,translation_length=translation,translation_fraction=.005,yaw_pitch_range_deg=[-2,2],rotation_convention='R camera-to-world, T world-to-camera translation; R_new=R*Ry*Rx; center_new=center+R*local_translation',mult=.6,views=rows),indent=2))
(out/'index.html').write_text('<meta charset="utf-8"><h1>100 perturbed views</h1><p>Left: original GT reference; middle: original-view render; right: perturbed render. No matching GT for perturbed views.</p>'+''.join(f'<p>{r["source_image"]}</p><img loading="lazy" style="max-width:100%" src="{r["comparison"]}">' for r in rows))
(out/'README.md').write_text('100 perturbed renders; same 100 source cameras as render_comparison_100.\nSeed 20260922. Independent yaw/pitch uniform [-2,2] degrees; no roll.\nRandom direction in local camera XY plane, translation length = 0.005 * training camera trajectory radius.\nCOLMAP units are not assumed to be meters. Intrinsics unchanged.\nComparison columns: original GT (reference only), original render, perturbed render.\nNo PSNR/SSIM: there is no ground truth for the perturbed pose.\nComplete poses and settings: poses.json.\n')
print('Saved',out,flush=True)
