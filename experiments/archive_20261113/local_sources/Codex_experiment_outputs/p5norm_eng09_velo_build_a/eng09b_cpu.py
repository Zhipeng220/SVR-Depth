import os,sys,csv,json,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1')
import torch,numpy as np
ROOT=Path('/home/svrdepth/SVR-Depth/velodepth');OUT=ROOT/'outputs/p5norm_eng09_velo_build_a';sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1')]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1
try:
 assert not torch.cuda.is_initialized()
 source=ROOT/'outputs/p5norm_eng09_velo_raw_stagea/raw_manifest.csv'
 with source.open(encoding='utf8',newline='') as f:orig=[next(csv.DictReader(f))] if False else list(csv.DictReader(f))[:3]
 assert len(orig)==3 and all(int(r['frame_index_zero_based'])==i for i,r in enumerate(orig))
 with (OUT/'prefix_manifest.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(orig[0]));w.writeheader();w.writerows(orig)
 rows=exp1.read_manifest(OUT/'prefix_manifest.csv',training=False);cfg=json.loads((Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c/config.json').read_text());cs=cfg['data']['shape_constraints'];pad,padded=gt2.get_paddings((480,640),cs['ratio_bounds']);_,net=gt2.get_resize_factor(padded,(float(cs['pixels_min']),float(cs['pixels_min'])+(float(cs['pixels_max'])-float(cs['pixels_min']))/10));assert tuple(net)==(434,574)
 shape={'original_shape':(480,640),'paddings':pad,'network_shape':net,'mean':torch.tensor(gt2.IMAGENET_DATASET_MEAN).view(1,3,1,1),'std':torch.tensor(gt2.IMAGENET_DATASET_STD).view(1,3,1,1)};result=[]
 for i,r in enumerate(rows):
  frame=exp1.load_frame(r,shape,torch.device('cpu'),need_gt=True);im=frame['img'];gt=frame['gt_np'];mask=np.isfinite(gt)&(gt>0);tum=np.isfinite(gt)&(gt>.001)&(gt<10);assert tuple(im.shape)==(1,3,434,574) and torch.isfinite(im).all() and np.isfinite(gt).all() and mask.any() and np.array_equal(mask,tum);assert str(frame['rgb_path'])==orig[i]['rgb_path'];result.append(dict(index=i,rgb_timestamp=orig[i]['rgb_timestamp'],depth_timestamp=orig[i]['depth_timestamp'],rgb_path=orig[i]['rgb_path'],depth_path=orig[i]['depth_path'],pose_association=orig[i]['pose_association'],shape=str(tuple(im.shape)),gt_shape=str(gt.shape),valid_pixels=int(mask.sum()),input_min=float(im.min()),input_max=float(im.max()),finite=True,status='PASS'))
 assert all(float(result[i]['rgb_timestamp'])<float(result[i+1]['rgb_timestamp']) for i in range(2)) and not torch.cuda.is_initialized()
 with (OUT/'P5NORM_ENG09_VELO_PREFIX_CPU.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 (OUT/'cpu_status.json').write_text(json.dumps({'status':'PASS','cuda_initialized':False,'model_instances':0,'frames':result},indent=2));print('B0 PASS: 3 frames, CUDA uninitialized',flush=True)
except Exception:
 (OUT/'cpu_status.json').write_text(json.dumps({'status':'FAIL','error':traceback.format_exc()},indent=2));print(traceback.format_exc());sys.exit(1)
