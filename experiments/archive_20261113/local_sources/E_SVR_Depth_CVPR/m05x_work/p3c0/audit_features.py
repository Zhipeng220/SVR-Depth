"""CPU only: verify P3-B frozen feature state bytes and quantify existing update directions."""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'docs/m05x/p3b';OUT=ROOT/'docs/m05x/p3c0';ALPHAS=(0.,.25,.5,.75,1.)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def run():
 m=json.loads((P/'evidence_manifest.json').read_text(encoding='utf8'));a=json.loads((ROOT/'docs/m05x/p2b/anchor_manifest.json').read_text(encoding='utf8'))['anchors']
 if m['status']!='PASS' or len(a)!=42:raise ValueError('frozen evidence incomplete')
 ranges={};records=[];maxerr=0.;mincos=1.
 for n,item in enumerate(a,1):
  aid=item['anchor_id'];d=P/'runs'/aid;meta=json.loads((d/'capture_metadata.json').read_text(encoding='utf8'));f=d/'evidence_arrays.npz';expected=m['anchors'][aid]['files']
  if sha(f)!=expected['evidence_arrays.npz'] or sha(d/'capture_metadata.json')!=expected['capture_metadata.json']:raise ValueError('P3-B file digest mismatch '+aid)
  index=m['anchors'][aid]['arrays'];aliases=meta['array_aliases'];ss={alpha:{'norm2':0.,'dot':0.,'ref2':0.} for alpha in ALPHAS[1:]}
  with np.load(f,allow_pickle=False) as z:
   for level in range(4):
    vals={}
    for prefix in ['committed_O','committed_T']+[f'committed_A{j}' for j in range(5)]:
     key=f'{prefix}__feature{level}';ar=z[aliases.get(key,key)];rec=index[key]
     if list(ar.shape)!=rec['shape'] or str(ar.dtype)!=rec['dtype'] or arrsha(ar)!=rec['sha256'] or not np.isfinite(ar).all():raise ValueError('feature byte/shape mismatch '+aid+' '+key)
     if tuple(ar.shape)!=(1,512,31,41) or ar.dtype!=np.float32:raise ValueError('unexpected feature state shape/dtype')
     vals[prefix]=ar
     group='old' if prefix=='committed_O' else 'candidate' if prefix=='committed_T' else None
     if group:
      row=ranges.setdefault(group,{'min':float('inf'),'max':-float('inf'),'shape':list(ar.shape),'dtype':str(ar.dtype)});row['min']=min(row['min'],float(ar.min()));row['max']=max(row['max'],float(ar.max()))
    old=vals['committed_O'];cand=vals['committed_T'];delta=cand.astype('float64')-old.astype('float64')
    for j,alpha in enumerate(ALPHAS):
     actual=vals[f'committed_A{j}'];target=old if alpha==0 else cand if alpha==1 else old+np.float32(alpha)*(cand-old)
     maxerr=max(maxerr,float(np.max(np.abs(actual-target))))
     if alpha>0:
      da=actual.astype('float64')-old.astype('float64');ss[alpha]['norm2']+=float(np.sum(da*da));ss[alpha]['dot']+=float(np.sum(da*delta));ss[alpha]['ref2']+=float(np.sum(delta*delta))
  if maxerr>1e-6:raise ValueError('interpolation mismatch '+aid)
  cos={str(alpha):(r['dot']/math.sqrt(r['norm2']*r['ref2']) if r['norm2'] and r['ref2'] else None) for alpha,r in ss.items()}
  if any(v is None for v in cos.values()):raise ValueError('zero direction magnitude '+aid)
  mincos=min(mincos,*cos.values());records.append({'anchor_id':aid,'dataset':item['dataset'],'sequence':item['sequence'],'age':item['age'],'delta_reference_l2':math.sqrt(ss[1.]['ref2']),'alpha_update_cosine_vs_reference':cos})
  if n%10==0:print('FEATURE_AUDIT',n,'/42',flush=True)
 result={'status':'PASS','anchor_count':len(records),'state_ranges':ranges,'max_interpolation_abs_error':maxerr,'min_alpha_direction_cosine':mincos,'definitions':{'F_old':'P3-B committed_O: S0 after unrepaired current propagation, not previous parent','F_candidate':'P3-B committed_T: S1 after pre-readout feature repair','delta_F':'committed_T minus committed_O, four feature levels concatenated for L2','alpha_directions':'same delta scaled; alpha endpoints cloned exactly'},'dimension_semantics':'four learned feature levels with 512 channels, each at 31x41; individual channels have no validated human semantic assignment','source_based_candidates':'See candidate_direction_inventory.csv: same-frame keyframe path is plausible but has not met joint-state/flow, norm-match, noncollinearity and D1 invariants in an actual run.','anchors':records}
 (OUT/'feature_direction_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print('FEATURE_AUDIT_PASS',len(records),'mincos',mincos,'maxerr',maxerr,flush=True)
if __name__=='__main__':run()
