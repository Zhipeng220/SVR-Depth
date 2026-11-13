from pathlib import Path
import csv,json,glob,io
import numpy as np,cv2,h5py
from PIL import Image
root=Path('/home/svrdepth/SVR-Depth/velodepth'); base=root/'outputs'; pilot=base/'p5norm_v2_ac_pilot/sequence_selection'; out=base/'p5norm_v2_4_independent_replication'
out.mkdir(exist_ok=False)
read=lambda p:list(csv.DictReader(p.open(encoding='utf-8-sig')))
prior=read(pilot/'pilot_sequence_selection.csv'); selected={(r['dataset'],r['sequence_id']) for r in prior}
elig=read(pilot/'pilot_sequence_gt_eligibility.csv'); history={}
for f in glob.glob(str(base/'m05x_p3b_20261109_223610/*/capture_metadata.json')):
 d=json.load(open(f));history.setdefault((d['dataset'],d['sequence']),{})[int(d['anchor'])]=d['anchor_id']
for p in [base/'p5norm_v2_1_norm_capture/P5NORM_V2_1_NORM_FEATURES.csv',base/'p5norm_v2_3_single_commit/P5NORM_V2_3_SINGLE_COMMIT_RESULTS.csv',base/'p5norm_v2_2_analysis/P5NORM_V2_2_COMMON_PREFIX_ANALYSIS.csv']:
 for r in read(p):
  if 'dataset' in r and 'sequence_id' in r:selected.add((r['dataset'],r['sequence_id']))
records=[];anchors=[]
for row in elig:
 d=row['dataset'];s=row['sequence_id']
 if (d,s) in selected:continue
 z=dict(row);z.update(p5_history='GT inventory only; project-history exposed (P3-B anchor metadata)',eligible='False');h=None
 try:
  assert row['eligible'].lower()=='true'
  if d=='Bonn':
   rr=read(base/'y04_v4_manifest_candidate_20261108_190940'/(s+'_pairs_v4_candidate.csv'));rgbs=[x['rgb_path'] for x in rr];gts=[x['depth_path'] if x['gt_match_status']=='MATCHED' else '' for x in rr];timestamps=[x['rgb_timestamp'] for x in rr]
  else:
   h=h5py.File(Path.home()/'datasets/velodepth_evaluation'/(d+'.hdf5'),'r');md=json.loads(bytes(h['sequences.json' if d=='TUM' else 'sequences_.json'][:]).decode());rgbs=md[s]['image'];gts=md[s]['depth'];timestamps=['']*len(rgbs)
  assert len(rgbs)==len(gts) and len(rgbs)>8
  aa=history[(d,s)];assert {1,4}.issubset(aa), 'historic anchors 1 and 4 missing'
  temp=[]
  for a in [1,4]:
   target=a+4
   for i in [a,target]:
    if d=='Bonn': rgb=cv2.imread(rgbs[i],cv2.IMREAD_COLOR);dep=cv2.imread(gts[i],cv2.IMREAD_UNCHANGED) if gts[i] else None
    else: rgb=np.array(Image.open(io.BytesIO(h[rgbs[i]][:].tobytes())).convert('RGB'));dep=np.array(Image.open(io.BytesIO(h[gts[i]][:].tobytes())))
    if rgb is None:raise ValueError('RGB unreadable at '+str(i))
    if i==target:
     if dep is None:raise ValueError('GT unreadable')
     if dep.ndim==3:dep=dep[:,:,0].astype(np.int32)|(dep[:,:,1].astype(np.int32)<<8)|(dep[:,:,2].astype(np.int32)<<16)
     if dep.shape!=rgb.shape[:2]:raise ValueError('GT shape mismatch')
     gt=dep.astype(np.float32)/(5000 if d=='Bonn' else 1000);mask=np.isfinite(gt)&(gt>0) if d=='Bonn' else np.isfinite(gt)&(gt>(.001 if d=='TUM' else .01))&(gt<10.)
     if not mask.any():raise ValueError('no valid GT pixel')
   temp.append(dict(dataset=d,sequence_id=s,anchor_id=aa[a],anchor_idx=a,target_idx=target,anchor_timestamp=timestamps[a],target_timestamp=timestamps[target],rgb_path=rgbs[target],gt_path=gts[target],eligible=True))
  anchors.extend(temp);z.update(eligible='True',reason='eligible; historical GT check plus anchor/target recheck')
 except Exception as e:z.update(reason=str(e))
 finally:
  if h:h.close()
 records.append(z)
def write(name,rr,fields=None):
 with (out/name).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rr[0]),extrasaction='ignore');w.writeheader();w.writerows(rr)
write('P5NORM_V2_4_SEQUENCE_ELIGIBILITY.csv',records)
write('P5NORM_V2_4_FROZEN_SEQUENCE_LIST.csv',[r for r in records if r['eligible']=='True'])
write('P5NORM_V2_4_FROZEN_ANCHORS.csv',anchors)
print('frozen',len(records),len(anchors),flush=True)
