import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path.cwd();p=Path(sys.argv[1]);aid=sys.argv[2];out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True);checks={};details=[]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
try:
 meta=json.loads((p/'metrics/result.json').read_text(encoding='utf8'));npz=p/'capture/capture.npz';checks['file_sha']=sha(npz)==meta['capture_npz_sha256'];checks['anchor']=meta['anchor']==aid
 hist=root/'docs/m05x/p3c1_formal/runs'/aid;oldmeta=json.loads((hist/'capture_metadata.json').read_text(encoding='utf8'))
 with np.load(npz,allow_pickle=False) as z,np.load(hist/'evidence_arrays.npz',allow_pickle=False) as old:
  get=lambda k:old[oldmeta.get('array_aliases',{}).get(k,k)]
  checks['all_array_hashes']=set(z.files)==set(meta['arrays']) and all(hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()==v['sha256'] and list(z[k].shape)==v['shape'] and str(z[k].dtype)==v['dtype'] for k,v in meta['arrays'].items())
  checks['all_arrays_finite']=all(np.isfinite(z[k]).all() for k in z.files if z[k].dtype.kind in 'fc');checks['D1_historical_exact']=np.array_equal(z['base_D1'],z['historical_D1']) and np.array_equal(z['base_D1'],get('current_D1_O'))
  checks['baseline_diagnostic_exact']=all(np.array_equal(z['base_'+x],z['diag_'+x]) for x in ['D1','radius','confidence','rays_map']+[f'feature_{i}' for i in range(4)])
  checks['hook_counts']=meta['hook_counts']=={'small_encoder':1,'repair':1,'grid':4,'grid_sample':4,'fusion':4};checks['hooks_restored_recorded']=meta['hooks_restored'] is True
  for i in range(4):
   raw=z['raw_grid_'+str(i)];actual=z['actual_grid_'+str(i)];mask=z['mask_'+str(i)].astype(bool);d=z['delta_'+str(i)];w=z['warp_'+str(i)];s0=z['S0_'+str(i)];cand=z['candidate_'+str(i)];expected_mask=np.isfinite(raw).all(axis=-1)&(raw[...,0]>=-1)&(raw[...,0]<=1)&(raw[...,1]>=-1)&(raw[...,1]<=1);e=np.broadcast_to(mask[:,None,:,:],d.shape);x=d[e].astype(np.float64);y=w[e].astype(np.float64);value=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))) if x.size and np.isfinite(x).all() and np.isfinite(y).all() and np.linalg.norm(x)>0 and np.linalg.norm(y)>0 else None;row=meta['layers'][i]
   layer={'mask_derived_exact':np.array_equal(mask,expected_mask),'grid_is_actual_sample_input':np.array_equal(actual,z['sample_grid_'+str(i)]),'warp_is_fusion_input_and_sample_output':np.array_equal(w,z['sample_output_'+str(i)]),'sample_input_is_historical_parent':np.array_equal(z['sample_input_'+str(i)],get('parent_O__feature'+str(i))),'s0_is_historical':np.array_equal(s0,get('committed_O__feature'+str(i))),'candidate_is_historical':np.array_equal(cand,get('candidate_V5__feature'+str(i))),'delta_is_candidate_minus_s0':np.array_equal(d,cand.astype(np.float32)-s0.astype(np.float32)),'flow_interpolation_recorded':row['flow_interpolation_sequence_exact'] is True,'mask_stats_match':int(mask.sum())==row['mask_valid'] and mask.size==row['mask_total'],'c3_recomputed_matches':value is not None and row['c3'] is not None and abs(value-row['c3'])<=1e-9,'c3_cpu':value,'raw_grid_sha256':hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),'actual_grid_sha256':hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest()}
   details.append(layer);checks['layer_'+str(i)]=all(v for k,v in layer.items() if isinstance(v,bool))
 result={'status':'PASS' if all(checks.values()) else 'FAIL','anchor':aid,'checks':checks,'layers':details,'scope':['live pointer and hook restoration are runtime logs; saved arrays alone cannot prove them','C3 is geometric proxy; not occlusion reliability'],'c3_cpu_tolerance':1e-9}
except Exception as e:result={'status':'FAIL','anchor':aid,'checks':checks,'error':repr(e)}
(out/'independent_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['status']=='PASS' else 1)
