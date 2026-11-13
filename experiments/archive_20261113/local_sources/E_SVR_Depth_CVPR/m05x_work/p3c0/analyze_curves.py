"""Offline sequence-level 5-point dose shape; no GPU and no input mutation."""
import csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'docs/m05x/p3b';OUT=ROOT/'docs/m05x/p3c0';ALPHA=np.array([0,.25,.5,.75,1.]);TOL=1e-10
def curve_stats(a,y):
 a=np.asarray(a,dtype=float);y=np.asarray(y,dtype=float)
 if a.shape!=(5,) or y.shape!=(5,) or not np.array_equal(a,ALPHA) or not np.isfinite(y).all():raise ValueError('invalid alpha curve')
 slope=float(np.dot(a-a.mean(),y-y.mean())/np.dot(a-a.mean(),a-a.mean()));intercept=float(y.mean()-slope*a.mean());dd=np.diff(y,n=2);d=np.diff(y)
 return {'linear_slope':slope,'linear_intercept':intercept,'curvature_dd_025':float(dd[0]),'curvature_dd_050':float(dd[1]),'curvature_dd_075':float(dd[2]),'max_abs_second_diff':float(np.abs(dd).max()),'nonmonotonic':bool(np.any(d>TOL) and np.any(d< -TOL)),'oracle_best_alpha':float(a[np.argmin(y)]),'alpha1_minus_alpha0':float(y[-1]-y[0])}
def run():
 m=json.loads((SRC/'evidence_manifest.json').read_text(encoding='utf8'));source=SRC/'dose_response_sequence.csv'
 if hashlib.sha256(source.read_bytes()).hexdigest()!=m['p3b_output_sha256']['dose_response_sequence.csv']:raise ValueError('source digest mismatch')
 groups=defaultdict(list)
 with source.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):groups[(r['dataset'],r['sequence'])].append(r)
 rows=[]
 for (domain,sequence),values in sorted(groups.items()):
  values.sort(key=lambda r:float(r['alpha']));y=[float(r['H4_AbsRel']) for r in values];x=[float(r['alpha']) for r in values];s=curve_stats(x,y)
  if any(float(r['oracle_best_alpha_H4'])!=s['oracle_best_alpha'] or (r['H4_nonmonotonic']=='True')!=s['nonmonotonic'] for r in values):raise ValueError('P3-B curve oracle mismatch')
  norm=float(values[-1]['feature_relative_l2']);rows.append({'dataset':domain,'sequence':sequence,'n_anchors':2,'relative_update_alpha1':norm,**{f'H4_alpha_{a:g}':y[i] for i,a in enumerate(ALPHA)},**s})
 if len(rows)!=21:raise ValueError('expected 21 curves')
 with (OUT/'dose_response_shape_analysis.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={}
 for domain in ('Bonn','TUM','ScanNet'):
  g=[x for x in rows if x['dataset']==domain];summary[domain]={'n_sequences':len(g),'slope_mean':statistics.mean(x['linear_slope'] for x in g),'slope_min':min(x['linear_slope'] for x in g),'slope_max':max(x['linear_slope'] for x in g),'nonmonotonic':sum(x['nonmonotonic'] for x in g),'oracle_counts':{str(a):int(sum(x['oracle_best_alpha']==a for x in g)) for a in ALPHA},'max_abs_second_diff':max(x['max_abs_second_diff'] for x in g)}
  pairs=[(abs(a['relative_update_alpha1']-b['relative_update_alpha1']),a,b) for i,a in enumerate(g) for b in g[i+1:] if a['alpha1_minus_alpha0']*b['alpha1_minus_alpha0']<0]
  if pairs:
   _,a,b=min(pairs,key=lambda x:x[0]);summary[domain]['nearest_opposite_pair']={'a':a['sequence'],'b':b['sequence'],'R_a':a['relative_update_alpha1'],'R_b':b['relative_update_alpha1'],'delta_a':a['alpha1_minus_alpha0'],'delta_b':b['alpha1_minus_alpha0'],'relative_difference':abs(a['relative_update_alpha1']-b['relative_update_alpha1'])/((a['relative_update_alpha1']+b['relative_update_alpha1'])/2)}
 (OUT/'dose_shape_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':run()
