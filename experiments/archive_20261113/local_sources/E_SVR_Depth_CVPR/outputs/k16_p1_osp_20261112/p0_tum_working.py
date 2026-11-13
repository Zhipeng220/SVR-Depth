import csv, json, io
import h5py, numpy as np
from PIL import Image
from pathlib import Path
p=Path(r'D:\SVR-Depth-data\VeloDepth-Evaluation\TUM.hdf5')
hist=Path(r'C:\Users\pyk18\Documents\ChatGPT\SVRDEPTH\V5_K16_配对周期结果.csv')
with hist.open(encoding='utf-8-sig',newline='') as f: historical={r['sequence'] for r in csv.DictReader(f) if r['dataset']=='TUM'}
out=Path('outputs/k16_p1_osp_20261112/P0_TUM_working_inventory.csv')
with h5py.File(p,'r') as h, out.open('w',encoding='utf-8-sig',newline='') as f:
 s=json.loads(bytes(h['sequences.json'][:]).decode('ascii')); keys=[x.rsplit(' ',1)[0] for x in bytes(h['val.txt'][:]).decode('ascii').strip().splitlines()]
 fields=['physical_scene_id','sequence_id','domain','exposure_status','timestamp_status','rgb_gt_readability','gt_coverage','gt_complete_index_cycle_candidates','complete_cycle_count','historical_OP_available','eligibility','exclusion_reason','metadata_fields']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for k in keys:
  q=s[k]; n=len(q['image']); complete=0; valid=0; errors=[]
  for i in range(n):
   try:
    im=np.asarray(Image.open(io.BytesIO(h[q['image'][i]][:].tobytes())).convert('RGB'))
    a=np.asarray(Image.open(io.BytesIO(h[q['depth'][i]][:].tobytes())))
    if a.ndim==3: a=a[:,:,0].astype(np.int32)|(a[:,:,1].astype(np.int32)<<8)|(a[:,:,2].astype(np.int32)<<16)
    gt=a.astype(np.float32)/1000.; ok=im.shape[:2]==gt.shape and bool((np.isfinite(gt)&(gt>0.001)&(gt<10)).any())
    valid+=ok
    if i%16==0: cyc=True
    cyc=cyc and ok
    if i%16==15 and cyc: complete+=1
   except Exception as e:
    errors.append(str(e)); cyc=False
  short=k.rsplit('/',1)[-1]; native=short.removesuffix('_val'); seen=k in historical or short in historical or native in historical
  provisional='TUM_FR1_DESK_AREA_PROVISIONAL' if native in ('fr1_desk','fr1_desk2','fr1_room') else 'UNKNOWN'
  w.writerow(dict(physical_scene_id=provisional,sequence_id=k,domain='TUM',exposure_status='historical V5 O/P benchmark' if seen else 'UNKNOWN',timestamp_status='UNKNOWN: no source timestamps or verified conversion/synchronization',rgb_gt_readability='all decoded' if not errors else 'ERROR: '+errors[0],gt_coverage=f'{valid}/{n} frames have at least one valid GT pixel',gt_complete_index_cycle_candidates=complete,complete_cycle_count=0,historical_OP_available='YES: historical O/P aggregate cycle metrics; predictions not established; protocol compatibility unverified' if seen else 'UNKNOWN',eligibility='NO',exclusion_reason='UNKNOWN physical scene identity/independence and timestamp continuity / RGB-GT correspondence',metadata_fields=';'.join(q.keys())))
  print(short,valid,n,complete,seen,flush=True)

