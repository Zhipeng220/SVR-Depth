import csv,json,math,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]); names=['global_l2','l0_l2','l1_l2','l2_l2','l3_l2']
def read(n):
 with (p/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
data=read('input.csv');oof={r['anchor_id']:r for r in read('oof.csv')}; config=json.loads((p/'config.json').read_text(encoding='utf8')); index={}
for f in sorted(p.glob('fold_*.json')):
 m=json.loads(f.read_text()); key=(m['heldout_dataset'],m['heldout_sequence_id']);assert key not in index;index[key]=f
assert len(index)==21 and len(data)==39
def load(dataset,sequence,filename=None,feature_names=names):
 key=(dataset,sequence)
 if key not in index:raise ValueError('unknown sequence')
 f=Path(filename) if filename else index[key]
 if not f.is_file():raise FileNotFoundError('model missing')
 m=json.loads(f.read_text())
 if (m['heldout_dataset'],m['heldout_sequence_id'])!=key:raise ValueError('wrong fold')
 if feature_names!=names or m['feature_names']!=names:raise ValueError('feature order/count')
 if m['alpha']!=1 or m['source_data_hash']!=hashlib.sha256((p/'input.csv').read_bytes()).hexdigest() or m['model_config_hash']!=hashlib.sha256((p/'config.json').read_bytes()).hexdigest():raise ValueError('identity')
 for k in ['scaler_mean','scaler_scale','ridge_coef']:
  if len(m[k])!=5 or any(not math.isfinite(x) for x in m[k]):raise ValueError('parameter')
 if min(m['scaler_scale'])<=0 or not math.isfinite(m['ridge_intercept']):raise ValueError('scale')
 return m
def predict(m,vals):
 if len(vals)!=5 or any(not math.isfinite(float(v)) for v in vals):raise ValueError('invalid input')
 return m['ridge_intercept']+math.fsum((float(v)-u)/s*b for v,u,s,b in zip(vals,m['scaler_mean'],m['scaler_scale'],m['ridge_coef']))
pred=[];checks=[]
for key,f in index.items():
 m=load(*key); subset=[r for r in data if (r['dataset'],r['sequence_id'])==key];assert subset
 train=[r for r in data if (r['dataset'],r['sequence_id'])!=key];assert {r['anchor_id'] for r in train}==set(m['train_anchor_ids']) and {r['anchor_id'] for r in subset}==set(m['heldout_anchor_ids'])
 for r in subset:
  v=predict(m,[r[k] for k in names]);ref=float(oof[r['anchor_id']]['M1_prediction']);err=abs(v-ref);assert err<=1e-10;pred.append({'anchor_id':r['anchor_id'],'dataset':key[0],'sequence_id':key[1],'fold_id':m['fold_id'],'prediction':v,'historical_oof':ref,'absolute_difference':err})
 checks.append({'case':'valid_fold','dataset':key[0],'sequence_id':key[1],'fold_id':m['fold_id'],'status':'PASS','detail':'identity and finite parameters validated'})
assert len(pred)==39
key=list(index)[0];other=list(index)[1]
def reject(case,fn):
 try:fn()
 except (ValueError,FileNotFoundError,KeyError,TypeError) as e:checks.append({'case':case,'dataset':'','sequence_id':'','fold_id':'','status':'PASS','detail':type(e).__name__});return
 raise AssertionError(case+' not rejected')
reject('wrong_fold',lambda:load(*key,filename=index[other]));reject('unknown',lambda:load('UNKNOWN','UNKNOWN'));reject('missing',lambda:load(*key,filename=p/'missing.json'));reject('feature_order',lambda:load(*key,feature_names=names[::-1]));reject('feature_count',lambda:load(*key,feature_names=names[:-1]));m=load(*key);reject('nonfinite',lambda:predict(m,[1,2,3,4,float('nan')]))
for filename,rows in [('P5NORM_M1_SERVER_LOADER_CHECK.csv',checks),('P5NORM_M1_39_PREDICTION_RECHECK.csv',pred)]:
 with (p/filename).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('PASS server CPU 21 folds 39 predictions max_error',max(x['absolute_difference'] for x in pred),'six rejection cases')
