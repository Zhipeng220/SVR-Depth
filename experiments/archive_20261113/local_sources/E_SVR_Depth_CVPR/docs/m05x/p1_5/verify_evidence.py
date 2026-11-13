"""Independent read-only validation of P1.5 NPZ; no model or capture code imports."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

FIELDS=('radius','confidence','rays_map','feature0','feature1','feature2','feature3')
def sha(b): return hashlib.sha256(b).hexdigest()
def verify(folder):
    folder=Path(folder);meta=json.loads((folder/'capture_metadata.json').read_text(encoding='utf-8'))
    with np.load(folder/'evidence_arrays.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    checks={};details={};index={}
    for k,a in arrays.items():
        index[k]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':sha(np.ascontiguousarray(a).tobytes())}
    def equal(a,b):
        if a not in arrays or b not in arrays:return False
        x,y=arrays[a],arrays[b]
        return x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x,y)
    checks['same_parent_state']=all(equal('parent_O__'+f,'parent_T__'+f) for f in FIELDS)
    checks['current_D1_equal']=equal('current_D1_O','current_D1_T') and np.isfinite(arrays['current_D1_O']).all()
    checks['valid_gt_masks_equal']=all(equal(f'gt_valid_O_{h}',f'gt_valid_T_{h}') and np.array_equal(arrays[f'gt_valid_O_{h}'],np.isfinite(arrays[f'gt_{h}'])&(arrays[f'gt_{h}']>0)) and arrays[f'gt_valid_O_{h}'].any() for h in (1,2,3,4))
    checks['metric_valid_masks_equal']=all(equal(f'metric_valid_O_{h}',f'metric_valid_T_{h}') for h in (1,2,3,4))
    # The runtime pointer observations alone are not sufficient: verify actual one-field mutation and sibling-before/after arrays too.
    for prefix,tag,metadata_key in [('alias','alias','alias_runtime'),('commit_alias','commit_alias','committed_alias_runtime')]:
        per={}
        for f in FIELDS:
            rt=meta.get(metadata_key,{}).get(f,{})
            per[f]=(bool(rt.get('o_ptr')) and bool(rt.get('t_ptr')) and rt.get('o_ptr')!=rt.get('t_ptr')
                    and equal(f'{tag}_before_T__{f}',f'{tag}_after_T__{f}')
                    and f'{prefix}_O_before__{f}' in arrays and f'{prefix}_O_mutated__{f}' in arrays
                    and not np.array_equal(arrays[f'{prefix}_O_before__{f}'],arrays[f'{prefix}_O_mutated__{f}']))
        details[metadata_key]=per
    def cross_disjoint(key):
        records=meta.get(key,{})
        return len(records)==len(FIELDS) and not ({records[f]['o_ptr'] for f in FIELDS}&{records[f]['t_ptr'] for f in FIELDS})
    checks['state_storage_independent']=all(details['alias_runtime'].values()) and all(details['committed_alias_runtime'].values()) and cross_disjoint('alias_runtime') and cross_disjoint('committed_alias_runtime')
    checks['nonfeature_state_unchanged']=all(equal('committed_O__'+f,'committed_T__'+f) for f in ('radius','confidence','rays_map'))
    feature_changes={f:not equal('committed_O__'+f,'committed_T__'+f) for f in FIELDS if f.startswith('feature')};details['feature_changes']=feature_changes
    rng=[k.removeprefix('replay_rng_O_') for k in arrays if k.startswith('replay_rng_O_')]
    rng_ok=bool(rng) and all(equal('replay_rng_O_'+k,'replay_rng_T_'+k) for k in rng)
    input_ok=equal('input_O_anchor','input_T_anchor') and all(equal(f'input_O_{h}',f'input_T_{h}') for h in (1,2,3,4))
    static_keys=('flow_norm','grid_sample','image_residual','validity_mask')
    static_ok=all(equal(f'static_O_{h}__{k}',f'static_T_{h}__{k}') and all(equal(f'static_{b}_{h}__{k}',f'static_after_{b}_{h}__{k}') for b in ('O','T')) for h in (1,2,3,4) for k in static_keys)
    checks['same_future_protocol']=bool(rng_ok and input_ok and static_ok and meta.get('repair_replay_mode')=='off' and meta.get('horizons')==[1,2,3,4])
    checks['gt_load_order_documented']=meta.get('gt_loaded_after_replays') is True
    checks['frozen_model_hashes_match']=meta.get('identity',{}).get('checkpoint')=='2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251' and meta.get('identity',{}).get('backbone')=='615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e'
    sources=list(Path(__file__).parent.glob('capture_invariants*.py'))
    checks['capture_source_sha_matches']=any(sha(p.read_bytes())==meta.get('identity',{}).get('capture_script') for p in sources)
    metrics={}
    for h in (1,4):
        for b in ('O','T'):
            gt=arrays[f'gt_{h}'];pred=arrays[f'pred_{b}_{h}'];valid=(gt>0)&np.isfinite(gt)&np.isfinite(pred)&(pred>0)
            checks[f'metric_mask_{b}_H{h}']=np.array_equal(valid,arrays[f'metric_valid_{b}_{h}']) and valid.any()
            metrics[f'{b}_H{h}']=float(np.mean(np.abs(gt[valid].astype('float64')-pred[valid].astype('float64'))/gt[valid].astype('float64'))) if valid.any() else None
            checks[f'metric_recomputed_{b}_H{h}']=metrics[f'{b}_H{h}'] is not None and abs(metrics[f'{b}_H{h}']-meta['metrics'][f'{b}_H{h}']['AbsRel'])<1e-12
    for h in (1,4):metrics[f'T_minus_O_H{h}']=metrics[f'T_H{h}']-metrics[f'O_H{h}']
    for key in list(checks): checks[key]=bool(checks[key])
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'details':details,'metrics_absrel':metrics,'scope':'one new Bonn development anchor only; not evidence of historical run','limits':['D1 arrays are two saved copies of a shared computation; not independent forward passes.','Pointer metadata and mutation snapshots observe this instrumented run, not historical execution.','GT usage is additionally a capture-code and control-flow audit, not independently proven by stored arrays alone.']},index

def main():
    p=argparse.ArgumentParser();p.add_argument('evidence_dir',type=Path);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    try: result,index=verify(a.evidence_dir)
    except (OSError,KeyError,ValueError,TypeError) as exc: result,index={'status':'BLOCKED','reason':str(exc)},{}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'invariant_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    files={str(p):sha(p.read_bytes()) for p in sorted(a.evidence_dir.iterdir()) if p.is_file()}
    manifest={'source_scope':'P1.5 new run only','files_sha256':files,'arrays':index}
    (a.output_dir/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
