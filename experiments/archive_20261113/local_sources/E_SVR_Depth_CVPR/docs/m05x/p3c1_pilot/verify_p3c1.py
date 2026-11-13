"""Independent matched-norm pilot verifier; only reads saved arrays, no model import."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BRANCHES=('O','V5','V4M','V4N','S');FIELDS=('radius','confidence','rays_map','feature0','feature1','feature2','feature3')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(folder):
 folder=Path(folder);meta=json.loads((folder/'capture_metadata.json').read_text(encoding='utf8'));f=folder/'evidence_arrays.npz'
 if sha(f)!=meta['evidence_npz_sha256']:raise ValueError('NPZ digest mismatch')
 with np.load(f,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 for name,ref in meta.get('array_aliases',{}).items():
  if name in a or ref not in a:raise ValueError('invalid deduplication reference')
  a[name]=a[ref]
 return meta,a
def verify(folder):
 meta,a=load(folder);lockpath=ROOT/'docs/m05x/p3c1_pilot/pilot_protocol.json';lock=json.loads(lockpath.read_text(encoding='utf8'));aid=meta['anchor_id'];anchors=json.loads((ROOT/'docs/m05x/p2b/anchor_manifest.json').read_text(encoding='utf8'))['anchors'];expected=next(x for x in anchors if x['anchor_id']==aid)
 p3=ROOT/'docs/m05x/p3b/runs'/aid;prior,old=load(p3);p3manifest=json.loads((ROOT/'docs/m05x/p3b/evidence_manifest.json').read_text(encoding='utf8'))['anchors'][aid]['files']
 checks={};detail={}
 def eq(x,y):return x in a and y in a and a[x].shape==a[y].shape and a[x].dtype==a[y].dtype and np.array_equal(a[x],a[y])
 ident=meta['identity'];checks['frozen_identity']=(ident['checkpoint']==lock['checkpoint_sha256']['V5'] and ident['checkpoint_v4']==lock['checkpoint_sha256']['V4'] and ident['backbone']==lock['backbone_sha256'] and ident['decoder']==lock['decoder_sha256'] and ident['repair_head']==lock['repair_source_sha256'] and ident['protocol']==lock['p2b_sequence_manifest_sha256'] and ident['c1_lock']==sha(lockpath) and sha(ROOT/'docs/m05x/p2b/anchor_manifest.json')==lock['p2b_anchor_manifest_sha256'] and ident['capture_script']==sha(ROOT/'m05x_work/p3c1_pilot/capture_p3c1.py'))
 checks['registered_anchor']=aid in lock['anchor_ids'] and all(meta[k]==expected[k] for k in ('dataset','sequence','sequence_index','age','cycle'))
 checks['prior_reference_integrity']=(sha(p3/'evidence_arrays.npz')==p3manifest['evidence_arrays.npz'] and sha(p3/'capture_metadata.json')==p3manifest['capture_metadata.json'] and prior['identity']['checkpoint']==lock['checkpoint_sha256']['V5'])
 source_keys=('backbone','model_config','exp1','gt2','m05','external_eval','pair_csv')+(('sequence_metadata',) if meta['dataset']!='Bonn' else ())
 checks['source_and_inputs_match_prior']=all(ident.get(k) is not None and ident[k]==prior['identity'].get(k) for k in source_keys) and bool(meta.get('inputs')) and meta['inputs']==prior['inputs']
 checks['runtime_and_precision_match_prior']=all(bool(meta.get(k)) and meta.get(k)==prior.get(k) for k in ('device','torch','cuda')) and meta.get('head_precision')=='cuda autocast float16'
 checks['same_parent']=all(eq('parent_O__'+f,'parent_T__'+f) for f in FIELDS)
 checks['same_s0_for_both_heads']=all(eq('head_input_V4__feature'+str(i),'head_input_V5__feature'+str(i)) and eq('head_input_V4__feature'+str(i),'head_input_after__feature'+str(i)) and eq('head_input_V4__feature'+str(i),'committed_O__feature'+str(i)) for i in range(4))
 checks['shared_current_d1']=all(eq('current_D1_O','current_D1_'+b) for b in BRANCHES)
 checks['candidate_and_commits_match']=all(eq('candidate_V5__feature'+str(i),'committed_V5__feature'+str(i)) and eq('candidate_V4__feature'+str(i),'committed_V4N__feature'+str(i)) and eq('candidate_V5__feature'+str(i),'committed_T__feature'+str(i)) for i in range(4))
 checks['nonfeature_state_equal']=all(eq('committed_O__'+f,'committed_'+b+'__'+f) for b in BRANCHES for f in ('radius','confidence','rays_map'))
 pointers=meta['storage_ptrs'];sets=[set(pointers[b].values()) for b in BRANCHES];checks['state_storage_independent']=all(len(s)==len(FIELDS) and 0 not in s for s in sets) and all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i))
 sums={'v4':0.,'v5':0.,'v4m':0.,'dot':0.};layers=[];maxerr=0.
 for i in range(4):
  f0=a[f'committed_O__feature{i}'];f4=a[f'candidate_V4__feature{i}'];f5=a[f'candidate_V5__feature{i}'];fm=a[f'committed_V4M__feature{i}']
  d4=(f4-f0).astype(np.float64);d5=(f5-f0).astype(np.float64);dm=(fm-f0).astype(np.float64)
  v4=float(np.sum(d4*d4));v5=float(np.sum(d5*d5));v4m=float(np.sum(dm*dm));dot=float(np.sum(d4*d5))
  for k,v in [('v4',v4),('v5',v5),('v4m',v4m),('dot',dot)]:sums[k]+=v
  layers.append({'layer':i,'v4_native_l2':math.sqrt(v4),'v5_native_l2':math.sqrt(v5),'v4_matched_l2':math.sqrt(v4m),'matched_minus_v5_l2':math.sqrt(v4m)-math.sqrt(v5),'cosine_v4_v5':dot/math.sqrt(v4*v5) if v4 and v5 else None})
  expected_m=f0+np.float32(meta['v4_scale_gpu'])*(f4-f0);maxerr=max(maxerr,float(np.max(np.abs(fm-expected_m))))
 n4,n5,nm=(math.sqrt(sums[k]) for k in ('v4','v5','v4m'));cos=sums['dot']/(n4*n5) if n4*n5 else None;relative=abs(nm-n5)/n5 if n5 else float('inf');scale=n5/n4 if n4 else None
 checks['finite_nonzero_norms']=n4>lock['norm_minimum'] and n5>lock['norm_minimum'] and all(math.isfinite(x) for x in (n4,n5,nm))
 checks['scale_recomputed']=scale is not None and math.isclose(scale,meta['v4_scale_gpu'],rel_tol=1e-9,abs_tol=1e-12) and lock['scale_minimum']<=scale<=lock['scale_maximum']
 checks['norm_match']=relative<=lock['norm_match_relative_tolerance'];checks['directions_noncollinear']=cos is not None and abs(cos)<lock['noncollinearity_max_abs_cosine'];checks['matched_formula']=maxerr<=lock['formula_abs_tolerance']
 detail['geometry']={'v4_native_l2':n4,'v5_native_l2':n5,'v4_matched_l2':nm,'matched_norm_relative_error':relative,'v4_v5_cosine':cos,'scale':scale,'formula_max_abs_error':maxerr,'layers':layers}
 checks['sham_matches_O']=all(eq('committed_O__'+f,'committed_S__'+f) for f in FIELDS) and all(eq(f'pred_O_{h}',f'pred_S_{h}') for h in range(1,5))
 rng=[k.removeprefix('replay_rng_O_') for k in a if k.startswith('replay_rng_O_')];static=('flow_norm','grid_sample','image_residual','validity_mask')
 checks['same_inputs_rng_cache']=bool(rng) and all(eq('replay_rng_O_'+k,'replay_rng_'+b+'_'+k) for b in BRANCHES for k in rng) and all(eq('input_O_anchor','input_'+b+'_anchor') and all(eq(f'input_O_{h}',f'input_{b}_{h}') for h in range(1,5)) for b in BRANCHES) and all(eq(f'static_O_{h}__{k}',f'static_{b}_{h}__{k}') and eq(f'static_{b}_{h}__{k}',f'static_after_{b}_{h}__{k}') for b in BRANCHES for h in range(1,5) for k in static)
 checks['repair_off_no_refresh']=meta['repair_replay_mode']=='off' and meta['K']==16 and meta['age']+4<16 and meta['horizons']==[1,2,3,4]
 checks['gt_loaded_after_replay_documented']=meta['gt_loaded_after_replays'] is True
 checks['finite_all_float_arrays']=all(np.isfinite(v).all() for v in a.values() if v.dtype.kind in 'fc')
 checks['gt_and_metric_masks_equal']=all(eq(f'gt_valid_O_{h}',f'gt_valid_{b}_{h}') and eq(f'metric_valid_{b}_{h}',f'gt_valid_{b}_{h}') and np.array_equal(a[f'gt_valid_{b}_{h}'],np.isfinite(a[f'gt_{h}'])&(a[f'gt_{h}']>0)) for b in BRANCHES for h in range(1,5))
 checks['prior_endpoints_exact']=all(np.array_equal(a[f'pred_{new}_{h}'],old[f'pred_{prev}_{h}']) for h in range(1,5) for new,prev in [('O','A0'),('V5','A4')]) and np.array_equal(a['current_D1_O'],old['current_D1_O']) and all(np.array_equal(a[f'gt_valid_O_{h}'],old[f'gt_valid_O_{h}']) for h in range(1,5))
 metrics={};counts={}
 for b in BRANCHES:
  for h in (1,4):
   gt=a[f'gt_{h}'];pred=a[f'pred_{b}_{h}'];mask=a[f'metric_valid_{b}_{h}'];g=gt[mask].astype(np.float64);p=pred[mask].astype(np.float64)
   if not len(g):raise ValueError('empty GT mask')
   metrics[f'{b}_H{h}']=float(np.mean(abs(g-p)/g));counts[f'{b}_H{h}']=len(g);checks[f'metric_{b}_H{h}_matches']=abs(metrics[f'{b}_H{h}']-meta['metrics'][f'{b}_H{h}']['AbsRel'])<1e-12
 for h in (1,4):metrics[f'direction_delta_H{h}']=metrics[f'V4M_H{h}']-metrics[f'V5_H{h}']
 for k in checks:checks[k]=bool(checks[k])
 index={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k,v in a.items()}
 return {'status':'PASS' if all(checks.values()) else 'INVALID','anchor_id':aid,'dataset':meta['dataset'],'sequence':meta['sequence'],'checks':checks,'details':detail,'metrics_absrel':metrics,'valid_pixels':counts,'limitations':['D1 is a single shared prediction, not independent forward passes.','GT order and live storage pointers require source and runtime audit.','Matched V4 is a scaled state and not native V4 output.']},index
def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
 try:r,index=verify(args.folder)
 except (OSError,KeyError,StopIteration,TypeError,ValueError) as e:r,index={'status':'BLOCKED','reason':str(e)},{}
 (args.output_dir/'invariant_validation.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf8');(args.output_dir/'evidence_manifest.json').write_text(json.dumps({'arrays':index,'files':{p.name:sha(p) for p in args.folder.iterdir() if p.is_file()}},indent=2)+'\n',encoding='utf8');print(json.dumps(r,ensure_ascii=False));return 0 if r['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
