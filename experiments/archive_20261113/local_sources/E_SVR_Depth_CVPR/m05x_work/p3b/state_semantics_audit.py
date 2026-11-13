import hashlib,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];p2=root/'docs/m05x/p2b';anchors=json.loads((p2/'anchor_manifest.json').read_text(encoding='utf8'))['anchors'];out=root/'docs/m05x/p3b';summary={};checks=[]
for entry in anchors:
 aid=entry['anchor_id'];d=p2/'runs'/aid;meta=json.loads((d/'capture_metadata.json').read_text(encoding='utf8'));alias=meta['array_aliases'];npz=d/'evidence_arrays.npz'
 h=hashlib.sha256()
 with npz.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 if h.hexdigest()!=meta['evidence_npz_sha256']:raise ValueError('digest mismatch '+aid)
 with np.load(npz,allow_pickle=False) as z:
  for prefix in ('committed_O','committed_T'):
   for i in range(4):
    key=prefix+'__feature'+str(i);a=z[alias.get(key,key)]
    if not np.isfinite(a).all():raise ValueError('nonfinite '+aid+' '+key)
    k=prefix+'_feature'+str(i);dct=summary.setdefault(k,{'min':float('inf'),'max':-float('inf'),'dtypes':{},'shapes':{}})
    dct['min']=min(dct['min'],float(a.min()));dct['max']=max(dct['max'],float(a.max()));dtype=str(a.dtype);shape=str(list(a.shape));dct['dtypes'][dtype]=dct['dtypes'].get(dtype,0)+1;dct['shapes'][shape]=dct['shapes'].get(shape,0)+1
 checks.append(aid)
out.mkdir(parents=True,exist_ok=True);(out/'state_semantics_ranges.json').write_text(json.dumps({'anchor_count':len(checks),'fields':summary,'source':'P2-B frozen O/T committed states; no inference'},ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('AUDITED',len(checks),json.dumps(summary),flush=True)
