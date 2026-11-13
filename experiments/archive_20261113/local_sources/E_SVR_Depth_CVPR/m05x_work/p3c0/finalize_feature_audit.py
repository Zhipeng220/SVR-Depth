import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'docs/m05x/p3c0';p=out/'feature_direction_audit.json';data=json.loads(p.read_text(encoding='utf8'));heads=json.loads((out/'head_parameter_audit.json').read_text(encoding='utf8'));geometry=json.loads((out/'candidate_cpu_geometry.json').read_text(encoding='utf8'))
assert heads['same_keys'] and heads['same_shapes'] and not heads['exact_equal'] and geometry['device']=='cpu' and not geometry['state_commit_performed'] and not geometry['gt_read'] and not geometry['future_replay_performed']
assert len(geometry['results'])==3 and all(r['v4_delta_l2_cpu']>0 and r['v5_delta_l2_saved']>0 and abs(r['cos_v4_cpu_vs_v5_saved'])<.999 and r['matched_norm_relative_error']<1e-6 for r in geometry['results'])
data['source_based_candidates']='The frozen V4 and V5 heads are model-derived plausible directions. CPU geometry on three saved shared S0 inputs demonstrates nonzero and non-collinear directions and near-matched float32 norms; D1/state invariants and mixed-precision GPU reproducibility remain untested.'
data['candidate_pair_cpu_audit']={'status':'GEOMETRY_PASS_CAUSAL_PILOT_NOT_RUN','head_same_keys':heads['same_keys'],'head_same_shapes':heads['same_shapes'],'head_parameter_delta_l2':heads['parameter_delta_l2'],'head_parameter_cosine':heads['parameter_cosine'],'results':geometry['results']}
p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print('candidate geometry conditional PASS')
