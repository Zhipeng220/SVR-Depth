"""Independent P3-B evidence reader, with historical P2-B endpoint comparison."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];FIELDS=('radius','confidence','rays_map','feature0','feature1','feature2','feature3');ALPHAS=(0.,.25,.5,.75,1.);BRANCHES=('O','T','S','A0','A1','A2','A3','A4')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(folder):
 folder=Path(folder);meta=json.loads((folder/'capture_metadata.json').read_text(encoding='utf8'));p=folder/'evidence_arrays.npz'
 if sha(p)!=meta['evidence_npz_sha256']:raise ValueError('NPZ hash mismatch')
 with np.load(p,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 for name,ref in meta.get('array_aliases',{}).items():
  if name in arrays or ref not in arrays:raise ValueError('invalid lossless deduplication reference')
  arrays[name]=arrays[ref]
 return meta,arrays
def verify(folder):
 meta,a=load(folder);aid=meta['anchor_id'];p2=ROOT/'docs/m05x/p2b/runs'/aid;oldmeta,old=load(p2);anchorlock=json.loads((ROOT/'docs/m05x/p2b/anchor_manifest.json').read_text(encoding='utf8'))['anchors'];lock=json.loads((ROOT/'docs/m05x/p3b/protocol_lock.json').read_text(encoding='utf8'));expected=next(x for x in anchorlock if x['anchor_id']==aid)
 checks={};detail={}
 def eq(x,y):return x in a and y in a and a[x].shape==a[y].shape and a[x].dtype==a[y].dtype and np.array_equal(a[x],a[y])
 checks['frozen_identity']=(meta['identity']['checkpoint']==lock['checkpoint_sha256'] and meta['identity']['backbone']==lock['backbone_sha256'] and meta['identity']['p3_lock']==sha(ROOT/'docs/m05x/p3b/protocol_lock.json') and meta['identity']['protocol']==lock['p2b_sequence_manifest_sha256'] and sha(ROOT/'docs/m05x/p2b/anchor_manifest.json')==lock['p2b_anchor_manifest_sha256'] and meta['identity']['capture_script']==sha(ROOT/'m05x_work/p3b/capture_p3b.py'))
 checks['registered_anchor']=all(meta[k]==expected[k] for k in ('dataset','sequence','sequence_index','age','cycle')) and (aid in lock['pilot_anchor_ids'] if meta['phase']=='pilot' else aid in lock['formal_anchor_ids'])
 checks['same_parent']=all(eq('parent_O__'+f,'parent_T__'+f) for f in FIELDS)
 checks['shared_d1']=all(eq('current_D1_O','current_D1_'+b) for b in BRANCHES)
 checks['nonfeature_state_equal']=all(eq('committed_O__'+f,'committed_'+b+'__'+f) for b in ('T','S','A0','A1','A2','A3','A4') for f in ('radius','confidence','rays_map'))
 checks['alpha_endpoints_features']=all(eq('committed_A0__feature'+str(i),'committed_O__feature'+str(i)) and eq('committed_A4__feature'+str(i),'committed_T__feature'+str(i)) for i in range(4))
 max_error=0.;feature_norms={}
 for j,alpha in enumerate(ALPHAS):
  ss=base=0.
  for i in range(4):
   f0=a[f'committed_O__feature{i}'];f1=a[f'committed_T__feature{i}'];fa=a[f'committed_A{j}__feature{i}'];target=f0 if alpha==0 else f1 if alpha==1 else f0+np.float32(alpha)*(f1-f0)
   max_error=max(max_error,float(np.max(np.abs(fa-target))));delta=fa.astype(np.float64)-f0.astype(np.float64);ss+=float(np.sum(delta*delta));base+=float(np.sum(f0.astype(np.float64)**2))
  feature_norms[str(alpha)]={'delta_l2':math.sqrt(ss),'relative_l2':math.sqrt(ss)/(math.sqrt(base)+1e-12)}
 checks['interpolation_formula']=max_error<=1e-6;detail['max_interpolation_error']=max_error;detail['feature_norms']=feature_norms
 ptr=meta['storage_ptrs'];sets=[set(ptr[b].values()) for b in BRANCHES];checks['storage_no_cross_branch_alias']=all(len(s)==len(FIELDS) and 0 not in s for s in sets) and all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i))
 checks['sham_state_and_predictions']=all(eq('committed_O__'+f,'committed_S__'+f) for f in FIELDS) and all(eq('pred_O_'+str(h),'pred_S_'+str(h)) for h in range(1,5))
 checks['alpha0_equals_O']=all(eq('pred_A0_'+str(h),'pred_O_'+str(h)) for h in range(1,5));checks['alpha1_equals_T']=all(eq('pred_A4_'+str(h),'pred_T_'+str(h)) for h in range(1,5))
 rng=[k.removeprefix('replay_rng_O_') for k in a if k.startswith('replay_rng_O_')]
 static=('flow_norm','grid_sample','image_residual','validity_mask')
 checks['same_inputs_rng_cache']=bool(rng) and all(eq('replay_rng_O_'+k,'replay_rng_'+b+'_'+k) for k in rng for b in BRANCHES) and all(eq('input_O_anchor','input_'+b+'_anchor') and all(eq(f'input_O_{h}',f'input_{b}_{h}') for h in range(1,5)) for b in BRANCHES) and all(eq(f'static_O_{h}__{k}',f'static_{b}_{h}__{k}') and eq(f'static_{b}_{h}__{k}',f'static_after_{b}_{h}__{k}') for b in BRANCHES for h in range(1,5) for k in static)
 checks['repair_off_no_refresh']=meta['repair_replay_mode']=='off' and meta['K']==16 and meta['age']+4<16 and meta['horizons']==[1,2,3,4]
 checks['gt_loaded_after_replay_documented']=meta['gt_loaded_after_replays'] is True
 checks['finite_all_float_arrays']=all(np.isfinite(x).all() for x in a.values() if x.dtype.kind in 'fc')
 checks['gt_masks_same']=all(eq(f'gt_valid_O_{h}',f'gt_valid_{b}_{h}') and eq(f'metric_valid_{b}_{h}',f'gt_valid_{b}_{h}') and np.array_equal(a[f'gt_valid_{b}_{h}'],np.isfinite(a[f'gt_{h}'])&(a[f'gt_{h}']>0)) for b in BRANCHES for h in range(1,5))
 prior=json.loads((ROOT/'docs/m05x/p2b/evidence_manifest.json').read_text(encoding='utf8'))['anchors'][aid]['files_sha256']
 def prior_sha(name):
  values=[v for k,v in prior.items() if k.replace('\\','/').endswith('/'+name)]
  if len(values)!=1:raise ValueError('P2-B provenance missing '+name)
  return values[0]
 checks['saved_p2b_reference_identity']=(oldmeta['identity']['checkpoint']==lock['checkpoint_sha256'] and oldmeta['sequence']==meta['sequence'] and oldmeta['age']==meta['age'] and sha(p2/'evidence_arrays.npz')==prior_sha('evidence_arrays.npz') and sha(p2/'capture_metadata.json')==prior_sha('capture_metadata.json'))
 source_keys=('backbone','model_config','exp1','gt2','m05','external_eval','pair_csv') + (('sequence_metadata',) if meta['dataset']!='Bonn' else ())
 checks['source_and_input_match_p2b']=all(meta['identity'].get(k) is not None and meta['identity'][k]==oldmeta['identity'].get(k) for k in source_keys) and meta.get('inputs')==oldmeta.get('inputs') and bool(meta.get('inputs'))
 checks['saved_p2b_reference_exact']=all(np.array_equal(a[f'pred_{new}_{h}'],old[f'pred_{prev}_{h}']) for h in range(1,5) for new,prev in [('A0','O'),('A4','T')]) and np.array_equal(a['current_D1_O'],old['current_D1_O']) and all(np.array_equal(a[f'gt_valid_O_{h}'],old[f'gt_valid_O_{h}']) for h in range(1,5))
 metrics={};counts={}
 for j,alpha in enumerate(ALPHAS):
  for h in (1,4):
   gt=a[f'gt_{h}'];pred=a[f'pred_A{j}_{h}'];mask=a[f'metric_valid_A{j}_{h}'];g=gt[mask].astype(np.float64);p=pred[mask].astype(np.float64)
   if not len(g):raise ValueError('empty GT mask')
   metrics[f'A{j}_H{h}']=float(np.mean(abs(g-p)/g));counts[f'A{j}_H{h}']=len(g)
   checks[f'metric_A{j}_H{h}_matches']=abs(metrics[f'A{j}_H{h}']-meta['metrics'][f'A{j}_H{h}']['AbsRel'])<1e-12
 for k in checks:checks[k]=bool(checks[k])
 array_manifest={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k,v in a.items()}
 result={'status':'PASS' if all(checks.values()) else 'INVALID','anchor_id':aid,'dataset':meta['dataset'],'sequence':meta['sequence'],'age':meta['age'],'phase':meta['phase'],'checks':checks,'details':detail,'metrics_absrel':metrics,'valid_pixels':counts,'limitations':['D1 is one shared prediction, not independent forwards.','GT load order and live storage pointers require instrumentation/source audit.','P2-B endpoints are compared exactly; no historical M05 Formal claim.']}
 return result,array_manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output-dir',required=True,type=Path);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
 try:r,index=verify(args.folder)
 except (OSError,KeyError,StopIteration,TypeError,ValueError) as e:r,index={'status':'BLOCKED','reason':str(e)},{}
 (args.output_dir/'invariant_validation.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf8');(args.output_dir/'evidence_manifest.json').write_text(json.dumps({'arrays':index,'files':{p.name:sha(p) for p in args.folder.iterdir() if p.is_file()}},indent=2)+'\n',encoding='utf8');print(json.dumps(r,ensure_ascii=False));return 0 if r['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
