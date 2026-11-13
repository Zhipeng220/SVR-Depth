import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, csv, json, bisect
from decimal import Decimal
from pathlib import Path
import cv2, numpy as np, torch
root=Path('/home/svrdepth/SVR-Depth/velodepth')
sys.path[:0]=[str(root),str(root/'experiments/p1')]
import gt2_fixed_clip_repair_train as gt2
import gt4a2_exp1_train_eval as exp1
out=root/'outputs/p5norm_eng09_velo_recovery'; out.mkdir(parents=True,exist_ok=True)
with (root/'outputs/p5norm_eng09_velo_raw_stagea/raw_manifest.csv').open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
assert len(rows)>34 and rows[19]['rgb_timestamp']=='1341846435.350127'
cfg=json.loads((Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c/config.json').read_text())
cs=cfg['data']['shape_constraints']; pad,padded=gt2.get_paddings((480,640),cs['ratio_bounds'])
_,net=gt2.get_resize_factor(padded,(float(cs['pixels_min']),float(cs['pixels_min'])+(float(cs['pixels_max'])-float(cs['pixels_min']))/10))
assert tuple(net)==(434,574)
shape={'original_shape':(480,640),'paddings':pad,'network_shape':net,'mean':torch.tensor(gt2.IMAGENET_DATASET_MEAN).view(1,3,1,1),'std':torch.tensor(gt2.IMAGENET_DATASET_STD).view(1,3,1,1)}
def save(name, data):
    with (out/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def loader(row, need):
    x=exp1.load_frame(row,shape,torch.device('cpu'),need_gt=need); img=x['img']
    assert torch.isfinite(img).all().item() and int(x['index'])==int(row['frame_index_zero_based'])
    assert abs(x['rgb_timestamp']-float(row['rgb_timestamp']))<1e-6
    g=x['gt_np']; n=int((np.isfinite(g)&(g>0)).sum()) if g is not None else ''
    return dict(status='PASS',shape=str(tuple(img.shape)),dtype=str(img.dtype),min=float(img.min()),max=float(img.max()),finite=True,valid_pixels=n,gt_shape=str(g.shape) if g is not None else '')
def safe(row,need):
    try: return loader(row,need)
    except Exception as e: return dict(status='FAIL: '+str(e),shape='',dtype='',min='',max='',finite='',valid_pixels='',gt_shape='')

checks=[]
for i in range(3,8):
 r=rows[i];x=loader(r,True);checks.append(dict(dataset="TUM_RAW",sequence="freiburg3_walking_halfsphere",index=i,rgb_timestamp=r["rgb_timestamp"],depth_timestamp=r["depth_timestamp"],rgb_path=r["rgb_path"],depth_path=r["depth_path"],depth_delta_ms=r["abs_delta_ms"],loader_status=x["status"],input_shape=x["shape"],gt_shape=x["gt_shape"],valid_pixels=x["valid_pixels"],hdf5_frame_mapping="UNKNOWN",camera="UNKNOWN"))
assert not torch.cuda.is_initialized()
with (root/"outputs/mnv02_raw_frame_recovery/MNV02_CPU_LOADER_VALIDATION.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=checks[0]);w.writeheader();w.writerows(checks)
print("5 formal CPU frames PASS; HDF5 identity UNKNOWN; CUDA uninitialized")
