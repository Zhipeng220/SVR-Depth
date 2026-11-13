"""Pre-result frozen candidate eligibility, CPU only; no hashes or predictions."""
from pathlib import Path
import csv,glob,json,io,collections
import numpy as np,cv2,h5py
from PIL import Image
ROOT=Path('/home/svrdepth/SVR-Depth/velodepth');DATA=Path('/home/svrdepth/datasets/velodepth_evaluation');out=ROOT/'outputs/p5norm_v2_ac_pilot/sequence_selection';out.mkdir(parents=True,exist_ok=False)
by=collections.defaultdict(lambda:collections.defaultdict(dict))
for f in glob.glob(str(ROOT/'outputs/m05x_p3b_20261109_223610/*/capture_metadata.json')):
 d=json.load(open(f));by[d['dataset']][d['sequence']][int(d['anchor'])]=d['anchor_id']
records=[];anchors=[]
for domain,seqs in sorted(by.items()):
 h=None;metadata=None
 if domain!='Bonn':
  h=h5py.File(DATA/(domain+'.hdf5'),'r');metadata=json.loads(bytes(h['sequences.json' if domain=='TUM' else 'sequences_.json'][:]).decode())
 for seq,ans in sorted(seqs.items()):
  entry={'dataset':domain,'sequence_id':seq,'eligible':False};valid=[];rgbs=[];gts=[];ts=[]
  try:
   if domain=='Bonn':
    f=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940'/(seq+'_pairs_v4_candidate.csv'); rows=list(csv.DictReader(f.open(encoding='utf-8-sig')));rgbs=[r['rgb_path'] for r in rows];gts=[r['depth_path'] if r['gt_match_status']=='MATCHED' else '' for r in rows];ts=[r['rgb_timestamp'] for r in rows]
   else:
    s=metadata[seq];rgbs=s['image'];gts=s['depth'];ts=['' for _ in rgbs]
   if len(rgbs)!=len(gts):raise ValueError('RGB/GT count differs')
   for i,(rp,gp) in enumerate(zip(rgbs,gts)):
    if domain=='Bonn':
     rgb=cv2.imread(rp,cv2.IMREAD_COLOR);dep=cv2.imread(gp,cv2.IMREAD_UNCHANGED) if gp else None
    else:
     rgb=np.array(Image.open(io.BytesIO(h[rp][:].tobytes())).convert('RGB'));dep=np.array(Image.open(io.BytesIO(h[gp][:].tobytes())))
    if rgb is None:raise ValueError('RGB missing or undecodable at '+str(i))
    if dep is None:valid.append(False);continue
    if dep.ndim==3:dep=dep[:,:,0].astype(np.int32)|(dep[:,:,1].astype(np.int32)<<8)|(dep[:,:,2].astype(np.int32)<<16)
    if dep.shape!=rgb.shape[:2]:raise ValueError('GT/RGB shape mismatch at '+str(i))
    gt=dep.astype(np.float32)/(5000 if domain=='Bonn' else 1000)
    mask=np.isfinite(gt)&(gt>0) if domain=='Bonn' else np.isfinite(gt)&(gt>(0.001 if domain=='TUM' else 0.01))&(gt<10.0)
    valid.append(bool(mask.any()))
   elig=[(a,aid) for a,aid in sorted(ans.items()) if a>0 and a+4<len(valid) and a//16==(a+4)//16 and valid[a+4]]
   entry.update(frame_count=len(rgbs),valid_gt_count=sum(valid),valid_gt_ratio=sum(valid)/len(valid),eligible_anchor_count=len(elig),first_valid_target=min((a+4 for a,_ in elig),default=''),last_valid_target=max((a+4 for a,_ in elig),default=''),eligible=bool(elig),reason='eligible' if elig else 'no frozen anchor with valid target')
   for a,aid in elig:anchors.append({'dataset':domain,'sequence_id':seq,'anchor_id':aid,'anchor_idx':a,'target_idx':a+4,'anchor_timestamp':ts[a],'target_timestamp':ts[a+4],'rgb_path':rgbs[a+4],'gt_path':gts[a+4],'eligible':True})
  except Exception as e:entry.update(frame_count=len(rgbs),valid_gt_count=sum(valid),valid_gt_ratio='',eligible_anchor_count=0,first_valid_target='',last_valid_target='',reason=str(e))
  records.append(entry);print(domain,seq,entry['reason'],flush=True)
 if h:h.close()
def write(n,rows,fields):
 with (out/n).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
write('pilot_sequence_gt_eligibility.csv',records,['dataset','sequence_id','frame_count','valid_gt_count','valid_gt_ratio','eligible_anchor_count','first_valid_target','last_valid_target','eligible','reason'])
chosen=[]
for domain in ['Bonn','TUM','ScanNet']:
 vals=[x for x in records if x['dataset']==domain and x['eligible']];vals.sort(key=(lambda x:x['sequence_id']) if domain=='TUM' else (lambda x:(x['frame_count'],x['sequence_id'])))
 indices=sorted(set([0,(len(vals)-1)//2,len(vals)-1])) if vals else []
 for i in indices:
  x=vals[i].copy();x['selection_reason']='GT eligible; pre-result '+('sequence_id equal spacing' if domain=='TUM' else 'length quantile; sequence_id tiebreak');chosen.append(x)
write('pilot_sequence_selection.csv',chosen,['dataset','sequence_id','frame_count','valid_gt_count','eligible_anchor_count','selection_reason'])
selected={(x['dataset'],x['sequence_id']) for x in chosen};write('pilot_anchor4_manifest.csv',[a for a in anchors if (a['dataset'],a['sequence_id']) in selected],['dataset','sequence_id','anchor_id','anchor_idx','target_idx','anchor_timestamp','target_timestamp','rgb_path','gt_path','eligible'])
print('SELECTION COMPLETE',len(chosen),'anchor_count',sum((x['dataset'],x['sequence_id']) in selected for x in anchors),flush=True)
