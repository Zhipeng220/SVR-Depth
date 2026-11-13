"""P3-B recording-level aggregation. No model inference."""
import csv,json,statistics,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/m05x/p3b';ALPHAS=(0.,.25,.5,.75,1.)
def summarize(rows,expected,reps=5000,seed=20261025):
 seen=set();groups=defaultdict(list)
 for row in rows:
  key=(row['dataset'],row['sequence'],row['age'],row['alpha'])
  if key in seen or row['status']!='PASS':raise ValueError('duplicate or invalid dose observation '+str(key))
  seen.add(key);groups[(row['dataset'],row['sequence'],row['alpha'])].append(row)
 if seen!=expected:raise ValueError(f'coverage mismatch missing {len(expected-seen)} extra {len(seen-expected)}')
 seq=[]
 for (dataset,sequence,alpha),r in sorted(groups.items()):
  if sorted(x['age'] for x in r)!=[1,4]:raise ValueError('missing age pair')
  seq.append({'dataset':dataset,'sequence':sequence,'alpha':alpha,'n_anchors':2,**{name:statistics.mean(x[name] for x in r) for name in ('H1_AbsRel','H4_AbsRel','H1_delta_vs_alpha0','H4_delta_vs_alpha0','feature_delta_l2','feature_relative_l2')},'pilot_p2a_exposed':r[0]['pilot_p2a_exposed']})
 for domain,seqid in sorted({(s['dataset'],s['sequence']) for s in seq}):
  curve=sorted([x for x in seq if x['dataset']==domain and x['sequence']==seqid],key=lambda x:x['alpha']);values=np.array([x['H4_AbsRel'] for x in curve]);diff=np.diff(values);nonmonotonic=bool(np.any(diff>1e-10) and np.any(diff< -1e-10));best=curve[int(np.argmin(values))]['alpha']
  for x in curve:x['oracle_best_alpha_H4']=best;x['H4_nonmonotonic']=nonmonotonic
 dataset=[]
 for domain in ('Bonn','TUM','ScanNet'):
  for alpha in ALPHAS:
   group=[x for x in seq if x['dataset']==domain and x['alpha']==alpha];n=len(group)
   if n<2:raise ValueError('too few recordings for '+domain)
   row={'dataset':domain,'alpha':alpha,'n_sequences':n,'n_anchors':2*n}
   for key in ('H1_AbsRel','H4_AbsRel','H1_delta_vs_alpha0','H4_delta_vs_alpha0','feature_delta_l2','feature_relative_l2'):
    v=np.array([x[key] for x in group]);row[key+'_mean']=float(np.mean(v));row[key+'_median']=float(np.median(v));row[key+'_std_ddof1']=float(np.std(v,ddof=1))
   vals=np.array([x['H4_delta_vs_alpha0'] for x in group]);rng=np.random.default_rng(seed);boot=vals[rng.integers(0,n,size=(reps,n))].mean(axis=1);row['H4_delta_mean_bootstrap95_low']=float(np.quantile(boot,.025));row['H4_delta_mean_bootstrap95_high']=float(np.quantile(boot,.975));row['H4_positive_sequences']=int((vals>0).sum());row['H4_negative_sequences']=int((vals<0).sum());dataset.append(row)
 return seq,dataset
def write(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 anchors=json.loads((ROOT/'docs/m05x/p2b/anchor_manifest.json').read_text(encoding='utf8'))['anchors'];lock=json.loads((OUT/'protocol_lock.json').read_text(encoding='utf8'));progress=json.loads((OUT/'execution_progress.json').read_text(encoding='utf8'))
 if progress['stopped_early'] or len(progress['completed'])!=42 or any(x['status']!='PASS' for x in progress['completed']):raise ValueError('formal coverage or pilot gate incomplete')
 rows=[];validations={};manifests={}
 for x in anchors:
  d=OUT/'runs'/x['anchor_id'];r=json.loads((d/'verification/invariant_validation.json').read_text(encoding='utf8'))
  if r['status']!='PASS' or (r['anchor_id'],r['dataset'],r['sequence'],r['age'])!=(x['anchor_id'],x['dataset'],x['sequence'],x['age']):raise ValueError('invalid anchor '+x['anchor_id'])
  validations[x['anchor_id']]=r;manifests[x['anchor_id']]=json.loads((d/'verification/evidence_manifest.json').read_text(encoding='utf8'))
  for j,alpha in enumerate(ALPHAS):
   norm=r['details']['feature_norms'][str(alpha)];m=r['metrics_absrel'];rows.append({'anchor_id':x['anchor_id'],'dataset':x['dataset'],'sequence':x['sequence'],'age':x['age'],'alpha':alpha,'status':r['status'],'H1_AbsRel':m[f'A{j}_H1'],'H4_AbsRel':m[f'A{j}_H4'],'H1_delta_vs_alpha0':m[f'A{j}_H1']-m['A0_H1'],'H4_delta_vs_alpha0':m[f'A{j}_H4']-m['A0_H4'],'feature_delta_l2':norm['delta_l2'],'feature_relative_l2':norm['relative_l2'],'pilot_p2a_exposed':x['pilot_p2a_exposed']})
 expected={(x['dataset'],x['sequence'],x['age'],alpha) for x in anchors for alpha in ALPHAS};seq,dataset=summarize(rows,expected)
 write(OUT/'dose_response_anchor.csv',rows);write(OUT/'dose_response_sequence.csv',seq);write(OUT/'dose_response_dataset.csv',dataset)
 endpoint={k:{x:v for x,v in r['checks'].items() if x in ('alpha_endpoints_features','alpha0_equals_O','alpha1_equals_T','saved_p2b_reference_exact','saved_p2b_reference_identity')} for k,r in validations.items()}
 (OUT/'alpha_endpoint_validation.json').write_text(json.dumps({'status':'PASS','anchors':endpoint},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 (OUT/'invariant_validation.json').write_text(json.dumps({'status':'PASS','anchors':validations},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 (OUT/'evidence_manifest.json').write_text(json.dumps({'status':'PASS','protocol_sha256':hashlib.sha256((OUT/'protocol_lock.json').read_bytes()).hexdigest(),'anchors':manifests},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
 for i,domain in enumerate(('Bonn','TUM','ScanNet')):
  ds=[r for r in dataset if r['dataset']==domain];x=np.array([r['alpha'] for r in ds]);y=np.array([r['H4_delta_vs_alpha0_mean'] for r in ds]);lo=np.array([r['H4_delta_mean_bootstrap95_low'] for r in ds]);hi=np.array([r['H4_delta_mean_bootstrap95_high'] for r in ds]);axes[i].plot(x,y,marker='o');axes[i].fill_between(x,lo,hi,alpha=.2);axes[i].axhline(0,c='gray',lw=.8);axes[i].set_title(f"{domain} (n={ds[0]['n_sequences']} recordings)");axes[i].set_xlabel('Feature commit alpha');axes[i].set_ylabel('H4 AbsRel relative to alpha=0')
 fig.suptitle('P3-B development dose response; bootstrap by recording');fig.savefig(OUT/'dose_response_curves.png',dpi=180);fig.savefig(OUT/'dose_response_curves.pdf');plt.close(fig)
 print('P3B_STATS_COMPLETE',len(rows),'dose observations',len(seq),'recording-dose rows',flush=True)
if __name__=='__main__':main()
