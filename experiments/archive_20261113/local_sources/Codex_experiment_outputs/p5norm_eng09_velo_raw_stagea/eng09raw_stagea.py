import os,sys,csv,json,math,bisect,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','2')
import cv2,numpy as np
cv2.setNumThreads(2)
ROOT=Path('/home/svrdepth/SVR-Depth/velodepth');OUT=ROOT/'outputs/p5norm_eng09_velo_raw_stagea';DATA=OUT/'raw/extracted/rgbd_dataset_freiburg3_walking_halfsphere';sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1')]
import y11_data
def parse_stamp_file(path):
 rows=[]
 for line in path.read_text().splitlines():
  if not line.strip() or line.lstrip().startswith('#'):continue
  bits=line.split();rows.append((bits[0],float(bits[0]),bits[1:]))
 assert all(math.isfinite(x[1]) for x in rows) and all(a[1]<b[1] for a,b in zip(rows,rows[1:])),str(path)+' timestamp duplicate/order'
 return rows
def write(name,rows):
 fields=list(dict.fromkeys(k for x in rows for k in x))
 with (OUT/name).open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
rgbraw=parse_stamp_file(DATA/'rgb.txt');depthraw=parse_stamp_file(DATA/'depth.txt');pose=parse_stamp_file(DATA/'groundtruth.txt');assert all(len(x[2])==7 and all(math.isfinite(float(v)) for v in x[2]) for x in pose)
rgb=y11_data.timestamp_rows(DATA/'rgb.txt',DATA);depth=y11_data.timestamp_rows(DATA/'depth.txt',DATA);pairs=y11_data.associate(rgb,depth,max_ms=20.);assert len(pairs)==len(rgbraw)
raw_r={str((DATA/x[2][0]).resolve()):x[0] for x in rgbraw};raw_d={str((DATA/x[2][0]).resolve()):x[0] for x in depthraw};pt=[x[1] for x in pose];times=[];assoc=[]
for i,r in enumerate(pairs):
 ts=float(r['rgb_timestamp']);j=bisect.bisect_left(pt,ts);ix=min((k for k in (j-1,j) if 0<=k<len(pt)),key=lambda k:abs(pt[k]-ts));ok=abs(pt[ix]-ts)<=.020
 r['rgb_timestamp']=raw_r[r['rgb_path']];r['depth_timestamp']=raw_d[r['depth_path']] if r['depth_path'] else '';r['source_sequence']='freiburg3_walking_halfsphere';r['source_frame_id']=rgbraw[i][2][0];r['source_gt_timestamp']=pose[ix][0] if ok else '';r['gt_pose_values']=' '.join(pose[ix][2]) if ok else '';r['gt_pose_delta_ms']=abs(pt[ix]-ts)*1000 if ok else '';r['pose_association']='MATCHED' if ok else 'UNMATCHED';assoc.append(r)
 times.append(dict(source_sequence=r['source_sequence'],source_frame_id=r['source_frame_id'],rgb_timestamp=r['rgb_timestamp'],rgb_path=r['rgb_path'],depth_timestamp=r['depth_timestamp'],depth_path=r['depth_path'],pose_timestamp=r['source_gt_timestamp'],pose_association=r['pose_association'],hdf_index='NOT_USED'))
write('P5NORM_ENG09_VELO_RAW_TIMESTAMP.csv',times);write('P5NORM_ENG09_VELO_RAW_ASSOCIATION.csv',assoc)
rawinterval=np.diff([x[1] for x in rgbraw]);med=float(np.median(rawinterval));assert med>0;maxgap=1.5*med;usable=[];grays=[]
for i,r in enumerate(assoc):
 rgb_img=cv2.imread(r['rgb_path'],cv2.IMREAD_COLOR);gray=None;ok=rgb_img is not None and rgb_img.shape==(480,640,3)
 if ok:gray=cv2.cvtColor(cv2.resize(rgb_img,(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
 gt=cv2.imread(r['depth_path'],cv2.IMREAD_UNCHANGED) if r['gt_match_status']=='MATCHED' else None
 ok=ok and gt is not None and gt.shape==(480,640) and gt.dtype==np.uint16 and bool((gt>0).any()) and r['pose_association']=='MATCHED';usable.append(ok);grays.append(gray)
 if i%200==0:print('CPU decode',i,'/',len(assoc),flush=True)
motion=[]
for i,(a,b) in enumerate(zip(grays,grays[1:])):
 if a is None or b is None:motion.append(float('nan'));continue
 flow=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0);motion.append(float(np.median(np.linalg.norm(flow,axis=2))/math.hypot(320,240)))
valid=[]
for i in range(len(assoc)-4):
 if all(usable[i:i+5]) and np.all(rawinterval[i:i+4]<=maxgap) and np.isfinite(motion[i:i+4]).all():valid.append((i,float(np.median(motion[i:i+4]))))
assert valid,'no legally qualified windows';threshold=float(np.percentile([x[1] for x in valid],75,method='linear'));start,score=next(x for x in valid if x[1]>=threshold);print('WINDOW',start,'SCORE',score,'THRESHOLD',threshold,'N',len(valid),flush=True)
window=[]
for k in range(5):
 i=start+k;r=assoc[i];window.append(dict(window_offset=k,original_rgb_index=i,source_frame_id=r['source_frame_id'],rgb_timestamp=r['rgb_timestamp'],rgb_path=r['rgb_path'],depth_timestamp=r['depth_timestamp'],depth_path=r['depth_path'],pose_timestamp=r['source_gt_timestamp'],pose_delta_ms=r['gt_pose_delta_ms'],motion_to_next=motion[i] if k<4 else '',window_score=score,q75_threshold=threshold,valid_window_count=len(valid),median_rgb_gap_seconds=med,max_allowed_gap_seconds=maxgap))
write('P5NORM_ENG09_VELO_RAW_WINDOW.csv',window)
# The actual M05 raw-manifest reader and frame loader are called only for the five frozen frames.
import torch
assert not torch.cuda.is_initialized()
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1
assert not torch.cuda.is_initialized()
manifest=OUT/'raw_manifest.csv';fields=list(assoc[0])
with manifest.open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(assoc)
loaded=exp1.read_manifest(manifest,training=False);cfg=json.loads((Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c/config.json').read_text());cs=cfg['data']['shape_constraints'];paddings,padded=gt2.get_paddings((480,640),cs['ratio_bounds']);_,network=gt2.get_resize_factor(padded,(float(cs['pixels_min']),float(cs['pixels_min'])+(float(cs['pixels_max'])-float(cs['pixels_min']))/10));assert tuple(network)==(434,574)
shape={'original_shape':(480,640),'paddings':paddings,'network_shape':network,'mean':torch.tensor(gt2.IMAGENET_DATASET_MEAN).view(1,3,1,1),'std':torch.tensor(gt2.IMAGENET_DATASET_STD).view(1,3,1,1)};loader=[]
for k in range(5):
 i=start+k;frame=exp1.load_frame(loaded[i],shape,torch.device('cpu'),need_gt=True);image=frame['img'];gt=frame['gt_np'];mask=np.isfinite(gt)&(gt>0);assert torch.isfinite(image).all() and np.isfinite(gt).all() and mask.any();assert Path(frame['rgb_path'])==Path(assoc[i]['rgb_path']);loader.append(dict(window_offset=k,source_frame_id=assoc[i]['source_frame_id'],rgb_timestamp=assoc[i]['rgb_timestamp'],depth_timestamp=assoc[i]['depth_timestamp'],pose_timestamp=assoc[i]['source_gt_timestamp'],rgb_shape='(480,640,3)',model_input_shape=str(tuple(image.shape)),input_dtype=str(image.dtype),input_min=float(image.min()),input_max=float(image.max()),gt_shape=str(gt.shape),gt_dtype=str(gt.dtype),gt_min=float(gt.min()),gt_max=float(gt.max()),valid_pixels=int(mask.sum()),finite=True,status='PASS'))
write('P5NORM_ENG09_VELO_RAW_LOADER.csv',loader);assert not torch.cuda.is_initialized();(OUT/'stagea_result.json').write_text(json.dumps(dict(raw_rgb=len(rgbraw),raw_depth=len(depthraw),raw_poses=len(pose),matched=sum(r['gt_match_status']=='MATCHED' for r in assoc),pose_matched=sum(r['pose_association']=='MATCHED' for r in assoc),median_gap=med,max_gap=maxgap,eligible_windows=len(valid),start=start,score=score,q75=threshold,model_shape=list(network),cuda_initialized=False),indent=2));print('LOADER PASS CUDA INITIALIZED',torch.cuda.is_initialized(),flush=True)
