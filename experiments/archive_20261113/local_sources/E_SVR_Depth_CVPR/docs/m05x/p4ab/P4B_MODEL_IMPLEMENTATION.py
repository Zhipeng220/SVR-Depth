"""P4AB-1.1: synthetic-only validation. Real-data fitting remains BLOCKED.
Do not call low-level ridge_predict on real labels; it is a numerical primitive.
Physical group certification is mandatory and IDs are never inferred from sequence names."""
"""Development-only P4-B Ridge implementation; requires certified physical groups."""
import numpy as np
NORM=['global_l2','l0_l2','l1_l2','l2_l2','l3_l2']
C3=['c3_l0','c3_l1','c3_l2','c3_l3']
def validate(rows):
 if not rows:raise ValueError('no data')
 keys=set(); anchor_groups={}; sequence_groups={}
 for r in rows:
  anchor=r.get('anchor_id'); group=r.get('physical_group_id'); sequence=(r.get('dataset'),r.get('sequence_id')) if r.get('sequence_id') else None
  if anchor in anchor_groups and anchor_groups[anchor]!=group: raise ValueError('anchor spans groups')
  anchor_groups[anchor]=group
  if sequence and sequence in sequence_groups and sequence_groups[sequence]!=group: raise ValueError('sequence spans groups')
  if sequence: sequence_groups[sequence]=group
  key=(r.get('anchor_id'),r.get('candidate'))
  if not all(key) or key in keys:raise ValueError('missing or duplicate anchor/candidate')
  keys.add(key)
  if r.get('physical_group_status')!='CERTIFIED' or not r.get('physical_group_id'):raise ValueError('physical source group not certified')
  if r.get('matched_evidence')!='PASS' or r.get('candidate')!='V5-native':raise ValueError('sample evidence or candidate mismatch')
  vals=[float(r[k]) for k in NORM+C3+['U_H4']]
  if not np.isfinite(vals).all():raise ValueError('nonfinite feature or label')
 if len({r['physical_group_id'] for r in rows})<3:raise ValueError('fewer than three physical groups')
def ridge_predict(train_x,train_y,test_x,alpha=1.0, *, synthetic_only=False):
 if synthetic_only is not True: raise PermissionError('Real-label fitting is BLOCKED')
 train_x=np.asarray(train_x,float);test_x=np.asarray(test_x,float);train_y=np.asarray(train_y,float)
 mean=train_x.mean(0);scale=train_x.std(0);scale=np.where(scale==0,1,scale)
 X=(train_x-mean)/scale;T=(test_x-mean)/scale
 ybar=train_y.mean();coef=np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@(train_y-ybar))
 return ybar+T@coef,{'train_mean':mean,'train_scale':scale,'y_mean':ybar,'coef':coef}
def rank(x):
 x=np.asarray(x,float);order=np.argsort(x,kind='stable');result=np.empty(len(x),float);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  result[order[i:j]]=(i+j-1)/2;i=j
 return result
def spearman(a,b):
 a,b=rank(a),rank(b)
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
 return float(np.corrcoef(a,b)[0,1])
def evaluate(rows, *, synthetic_only=False):
 if synthetic_only is not True or any(r.get('synthetic_marker') != 'SYNTHETIC_TEST_ONLY' for r in rows): raise PermissionError('Real-label fitting is BLOCKED; synthetic tests only')
 validate(rows);groups=np.array([r['physical_group_id'] for r in rows]);y=np.array([float(r['U_H4']) for r in rows]);features=np.array([[float(r[k]) for k in NORM+C3] for r in rows]);pred={m:np.empty(len(rows)) for m in ['M0','M1','M2']};splits=[]
 for group in sorted(set(groups)):
  tr=groups!=group;te=~tr
  pred['M0'][te]=y[tr].mean();pred['M1'][te]=ridge_predict(features[tr,:5],y[tr],features[te,:5],synthetic_only=True)[0];pred['M2'][te]=ridge_predict(features[tr],y[tr],features[te],synthetic_only=True)[0];splits.append({'test_group':group,'train_groups':sorted(set(groups[tr])),'test_indices':np.flatnonzero(te).tolist(),'train_indices':np.flatnonzero(tr).tolist()})
 group_rows=[]
 for group in sorted(set(groups)):
  ix=np.flatnonzero(groups==group);dataset={rows[i]['dataset'] for i in ix}
  if len(dataset)!=1:raise ValueError('group crosses domains; report grouping explicitly before analysis')
  r={'physical_group_id':group,'dataset':next(iter(dataset)),'n':len(ix),'truth':float(y[ix].mean())}
  for m in pred:r[m]=float(pred[m][ix].mean())
  group_rows.append(r)
 return {'sample_predictions':{k:v.tolist() for k,v in pred.items()},'groups':group_rows,'splits':splits}
def aggregate(groups):
 y=np.array([g['truth'] for g in groups]);out={}
 for m in ['M0','M1','M2']:
  q=np.array([g[m] for g in groups]);e=q-y;out[m]={'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'Spearman':spearman(y,q)}
 out['mae_improvement_rate']=(out['M1']['MAE']-out['M2']['MAE'])/out['M1']['MAE'] if out['M1']['MAE']>0 else None
 return out
def bootstrap_delta(groups,seed=20261110,resamples=10000):
 rng=np.random.default_rng(seed);a=np.array([abs(g['truth']-g['M1'])-abs(g['truth']-g['M2']) for g in groups]);draw=a[rng.integers(0,len(a),size=(resamples,len(a)))].mean(axis=1)
 return {'mean':float(a.mean()),'lower':float(np.quantile(draw,.025)),'upper':float(np.quantile(draw,.975)),'unit':'physical_group','resamples':resamples,'seed':seed}

