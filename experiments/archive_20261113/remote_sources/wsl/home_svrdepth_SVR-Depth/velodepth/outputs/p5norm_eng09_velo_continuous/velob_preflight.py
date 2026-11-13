import os,sys,json,csv,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1')
import torch,numpy as np
ROOT=Path('/home/svrdepth/SVR-Depth/velodepth');OUT=ROOT/'outputs/p5norm_eng09_velo_continuous';sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1')]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1
records=[];status={'status':'NO-GO','cuda_initialized':False,'model_instances':0}
try:
 assert not torch.cuda.is_initialized();src=ROOT/'outputs/p5norm_eng09_velo_raw_stagea/raw_manifest.csv'
 with src.open(encoding='utf8',newline='') as f:r=list(csv.DictReader(f))
 cfg=json.loads((Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c/config.json').read_text());cs=cfg['data']['shape_constraints'];pad,padded=gt2.get_paddings((480,640),cs['ratio_bounds']);_,net=gt2.get_resize_factor(padded,(float(cs['pixels_min']),float(cs['pixels_min'])+(float(cs['pixels_max'])-float(cs['pixels_min']))/10));assert tuple(net)==(434,574)
 shape={'original_shape':(480,640),'paddings':pad,'network_shape':net,'mean':torch.tensor(gt2.IMAGENET_DATASET_MEAN).view(1,3,1,1),'std':torch.tensor(gt2.IMAGENET_DATASET_STD).view(1,3,1,1)}
 for i in range(8,35):
  x=r[i];row={'source_index':i,'source_frame_id':x['source_frame_id'],'rgb_timestamp':x['rgb_timestamp'],'depth_timestamp':x['depth_timestamp'],'depth_mapping':x['gt_match_status'],'pose_timestamp':x['source_gt_timestamp'],'pose_association':x['pose_association']}
  try:
   assert int(x['frame_index_zero_based'])==i and float(x['rgb_timestamp'])>float(r[i-1]['rgb_timestamp']);assert x['pose_association']=='MATCHED' and x['source_gt_timestamp'];frame=exp1.load_frame(x,shape,torch.device('cpu'),need_gt=True);img=frame['img'];gt=frame['gt_np'];mask=np.isfinite(gt)&(gt>0);assert torch.isfinite(img).all() and np.isfinite(gt).all() and mask.any();row.update(status='PASS',input_shape=str(tuple(img.shape)),gt_shape=str(gt.shape),valid_pixels=int(mask.sum()),input_min=float(img.min()),input_max=float(img.max()))
  except Exception as e:
   row.update(status='FAIL',reason=str(e),exception=type(e).__name__);records.append(row);raise RuntimeError('source index '+str(i)+' failed formal Loader: '+str(e)) from e
  records.append(row)
 assert not torch.cuda.is_initialized();status.update(status='PASS')
except Exception:status['error']=traceback.format_exc()
finally:
 fields=list(dict.fromkeys(k for x in records for k in x))
 with (OUT/'P5NORM_ENG09_VELOB_CPU_PREFLIGHT.csv').open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
 (OUT/'cpu_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if status['status']!='PASS':sys.exit(1)
