import csv,hashlib,json
from pathlib import Path
root=Path.cwd();work=root/'m05x_work/p4ab_parallel_20261110';work.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
anchors=root/'docs/m05x/p3c1_formal/anchor_manifest.json'
a=json.loads(anchors.read_text(encoding='utf8'))['formal_anchors']
if len(a)!=39 or len({x['anchor_id'] for x in a})!=39:raise RuntimeError('anchor manifest invalid')
p={'version':'P4AB-Parallel-1.0','frozen_before_new_results':True,'candidate':'V5-native only; O matched to same parent and mask','target':'U_H4=O_H4_AbsRel-V5_H4_AbsRel','anchor_ids':[x['anchor_id'] for x in a],'anchor_manifest_sha256':sha(anchors),'checkpoint_sha256':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251','norm_features':['global_l2','l0_l2','l1_l2','l2_l2','l3_l2'],'c3_features':['c3_l0','c3_l1','c3_l2','c3_l3'],'ridge_alpha':1.0,'scale':'training-fold standardization; zero variance -> scale 1; unpenalized intercept','split':'leave-one-certified-physical-recording-out; never split candidates/anchors sharing physical source','identity_gate':'unresolved physical source means no LOPO or grouped inference; preserve all unknowns and report BLOCKED, no guessed identity','unit':'equal physical-group weight after sample aggregation','holm':'four layer p values per family and per domain','association':'within-domain physical-group averages and group-level paired Spearman; residualized rank analysis on four norm covariates (descriptive), group permutation; no frame-level independence','seed':20261110,'bootstrap_resamples':10000,'permutations':10000,'P4A_GO':'match evidence, certified groups, Holm-adjusted p<0.05 for at least one layer in at least two domains, effect signs and sample counts reported','P4B_GO':'M2_MAE<M1_MAE with paired group-bootstrap 95% CI for M1−M2 lower bound >0, no single domain or group solely drives direction, and no leakage','pilot_gate':'the 3 smoke anchors cannot alone support training or generalization','negative_zero_tie':'U>0 beneficial; U<0 harmful; U=0 no change, no post-hoc epsilon','dataset_role':'development only; not independent test'}
p['P4B_GO']='M2_MAE<M1_MAE; paired group-bootstrap 95% CI of M1−M2 has lower bound >0; per-domain improvements nonnegative in all three and positive in at least two; dropping any one physical group does not reverse overall improvement; no leakage'
f=work/'protocol_lock.json'
if f.exists() and json.loads(f.read_text(encoding='utf8'))!=p:raise RuntimeError('frozen protocol mismatch')
if not f.exists():f.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('FROZEN',len(a),'anchors',sha(f))

