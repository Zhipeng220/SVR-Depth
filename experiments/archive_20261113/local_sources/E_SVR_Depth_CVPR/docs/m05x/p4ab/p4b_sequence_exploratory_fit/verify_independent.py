"""Recompute all 21 OOF folds with augmented least squares, not original solver."""
import csv,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
root=Path.cwd();p=root/'docs/m05x/p4ab/p4b_sequence_exploratory_fit'
def load(f):
 with f.open(encoding='utf-8-sig',newline='') as x:return list(csv.DictReader(x))
rows=load(root/'docs/m05x/p4ab/P4A_PAIRED_DATA_AUDIT.csv');oof=load(p/'P4BE_OOF_PREDICTIONS.csv');seq=load(p/'P4BE_SEQUENCE_METRICS.csv');folds=load(p/'P4BE_FOLD_DIAGNOSTICS.csv');reports=load(p/'P4BE_MODEL_COMPARISON.csv');conf=json.loads((p/'P4BE_CONFIG.json').read_text(encoding='utf8'))
assert len(rows)==len(oof)==39 and len(seq)==len(folds)==21 and len({(r['dataset'],r['sequence_id']) for r in seq})==21
o={r['anchor_id']:r for r in oof};assert set(o)=={r['anchor_id'] for r in rows}
ks=conf['models']['M2'];x=np.array([[float(r[k]) for k in ks] for r in rows]);y=np.array([float(r['U_H4']) for r in rows]);maxdiff=0.0
for f in folds:
 te=np.array([(r['dataset'],r['sequence_id'])==(f['dataset'],f['sequence_id']) for r in rows]); tr=~te
 assert te.sum()==int(f['test_n']) and tr.sum()==int(f['train_n']) and te.sum()>0
 for m,pdim in [('M0',0),('M1',5),('M2',9)]:
  if pdim==0:pred=np.full(te.sum(),y[tr].mean())
  else:
   a=x[tr,:pdim];b=x[te,:pdim];mu=a.mean(0);sd=a.std(0);sd[sd==0]=1;z=(a-mu)/sd;zt=(b-mu)/sd
   design=np.vstack([z,np.eye(pdim)]);target=np.concatenate([y[tr]-y[tr].mean(),np.zeros(pdim)]);beta=np.linalg.lstsq(design,target,rcond=None)[0];pred=y[tr].mean()+zt@beta
  for idx,value in zip(np.flatnonzero(te),pred):maxdiff=max(maxdiff,abs(float(value)-float(o[rows[idx]['anchor_id']][m+'_prediction'])))
assert maxdiff<1e-12
for s in seq:
 group=[o[r['anchor_id']] for r in rows if (r['dataset'],r['sequence_id'])==(s['dataset'],s['sequence_id'])];assert len(group)==int(s['anchor_n'])
 for m in ['M0','M1','M2']:
  assert abs(np.mean([float(q[m+'_prediction']) for q in group])-float(s[m+'_prediction_mean']))<1e-12
  assert abs(abs(float(s[m+'_prediction_mean'])-float(s['U_H4_mean']))-float(s[m+'_abs_error']))<1e-12
for unit in ['sequence','anchor']:
 r=next(v for v in reports if v['unit']==unit)
 for m in ['M0','M1','M2']:
  ar=np.array([float(s[m+'_error']) for s in (seq if unit=='sequence' else oof)]);assert abs(np.mean(abs(ar))-float(r[m+'_MAE']))<1e-12;assert abs(np.sqrt(np.mean(ar**2))-float(r[m+'_RMSE']))<1e-12
print('PASS 39 OOF predictions, 21 disjoint folds, alternate least-squares, aggregation and MAE/RMSE; max difference',maxdiff)
