import hashlib,json
from pathlib import Path
root=Path.cwd();work=root/'m05x_work/p4ab_parallel_20261110';d=root/'docs/m05x/p3c1_formal';lock=json.loads((work/'protocol_lock.json').read_text(encoding='utf8'));a=json.loads((d/'anchor_manifest.json').read_text(encoding='utf8'))['formal_anchors'];files=json.loads((d/'evidence_manifest.json').read_text(encoding='utf8'))['files']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[]
for x in a:
 aid=x['anchor_id'];base=d/'runs'/aid;rec={'anchor_id':aid,'dataset':x['dataset'],'age':x['age'],'sequence':x['sequence']}
 for name in ['capture_metadata.json','evidence_arrays.npz','verification/invariant_validation.json']:
  p=base/name;h=sha(p);rel='runs/'+aid+'/'+name
  if h!=files[rel]['sha256']:raise RuntimeError('historical hash mismatch '+rel)
  rec[name]=h
 val=json.loads((base/'verification/invariant_validation.json').read_text(encoding='utf8'))
 if val['status']!='PASS' or not val['checks']['gt_and_metric_masks_equal'] or not val['checks']['prior_endpoints_exact']:raise RuntimeError('historical validity failed '+aid)
 rows.append(rec)
p=work/'capture_selection.json';v={'protocol_lock_sha256':sha(work/'protocol_lock.json'),'fixed_before_capture':True,'candidates':'V5-native only','reused_full_captures':['TUM_00_age4','ScanNet_00_age4'],'bonn_smoke_note':'prior smoke report lacks raw warp/S0 data needed for independent sample verification; recapture same fixed anchor','records':rows}
if p.exists() and json.loads(p.read_text(encoding='utf8'))!=v:raise RuntimeError('selection mismatch')
if not p.exists():p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('SELECTION_PASS',len(rows),'sha256',sha(p))

