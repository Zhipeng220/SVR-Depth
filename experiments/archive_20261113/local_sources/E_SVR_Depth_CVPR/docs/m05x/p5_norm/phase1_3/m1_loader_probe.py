import csv,json,hashlib,math,tempfile,shutil
from pathlib import Path
root=Path.cwd(); out=root/'docs/m05x/p5_norm/phase1_3'; mr=root/'docs/m05x/p5_norm/phase1_1/model_recovery'; historic=root/'docs/m05x/p4ab/p4b_sequence_exploratory_fit';source=root/'docs/m05x/p4ab/P4A_PAIRED_DATA_AUDIT.csv'
keys=['global_l2','l0_l2','l1_l2','l2_l2','l3_l2']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
data=read(source);old={r['anchor_id']:r for r in read(historic/'P4BE_OOF_PREDICTIONS.csv')}; config_sha=sha(historic/'P4BE_CONFIG.json'); data_sha=sha(source)
def load_guard(dataset,sequence,fold_path,feature_names):
 if list(feature_names)!=keys:raise ValueError('feature order mismatch')
 if not fold_path.is_file():raise FileNotFoundError('model missing')
 m=json.loads(fold_path.read_text(encoding='utf8'))
 if (m['heldout_dataset'],m['heldout_sequence_id'])!=(dataset,sequence):raise ValueError('fold/sequence mismatch')
 if m['feature_names']!=keys or m['alpha']!=1.0 or m['source_data_hash']!=data_sha or m['model_config_hash']!=config_sha:raise ValueError('model identity/config mismatch')
 for k in ['scaler_mean','scaler_scale','ridge_coef']:
  if len(m[k])!=5 or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in m[k]):raise ValueError('bad parameters')
 if any(v<=0 for v in m['scaler_scale']) or not math.isfinite(m['ridge_intercept']):raise ValueError('bad scale/intercept')
 return m
def predict(m,features):
 if len(features)!=5 or any(not math.isfinite(float(v)) for v in features):raise ValueError('bad input')
 return m['ridge_intercept']+math.fsum((float(x)-mu)/scale*b for x,mu,scale,b in zip(features,m['scaler_mean'],m['scaler_scale'],m['ridge_coef']))
models=sorted(mr.glob('fold_*.json'));assert len(models)==21
checks=[];seen=set();maxerr=0
for f in models:
 m=json.loads(f.read_text(encoding='utf8'));ds,seq=m['heldout_dataset'],m['heldout_sequence_id'];assert (ds,seq) not in seen;seen.add((ds,seq));m=load_guard(ds,seq,f,keys);rr=[r for r in data if (r['dataset'],r['sequence_id'])==(ds,seq)];assert rr
 training=[r for r in data if (r['dataset'],r['sequence_id'])!=(ds,seq)]
 assert {r['anchor_id'] for r in training}==set(m['train_anchor_ids']) and {r['anchor_id'] for r in rr}==set(m['heldout_anchor_ids'])
 for i,k in enumerate(keys):
  vals=[float(r[k]) for r in training];mu=math.fsum(vals)/len(vals);sd=math.sqrt(math.fsum((v-mu)**2 for v in vals)/len(vals));sd=sd or 1.0
  assert math.isclose(mu,m['scaler_mean'][i],rel_tol=1e-12,abs_tol=1e-12) and math.isclose(sd,m['scaler_scale'][i],rel_tol=1e-12,abs_tol=1e-12)
 for r in rr:
  actual=predict(m,[r[k] for k in keys]);err=abs(actual-float(old[r['anchor_id']]['M1_prediction']));maxerr=max(maxerr,err);assert err<=1e-10
 checks.append({'check':'valid_fold','dataset':ds,'sequence_id':seq,'fold':m['fold_id'],'samples':len(rr),'max_abs_difference':max(abs(predict(m,[r[k] for k in keys])-float(old[r['anchor_id']]['M1_prediction'])) for r in rr),'status':'PASS'})
f0=models[0]; m0=json.loads(f0.read_text(encoding='utf8'))
def must_reject(name,func):
 try:func()
 except (ValueError,FileNotFoundError,KeyError,TypeError) as e:checks.append({'check':name,'dataset':'','sequence_id':'','fold':'','samples':'','max_abs_difference':'','status':'PASS:'+type(e).__name__});return
 raise AssertionError(name+' did not reject')
must_reject('wrong_fold',lambda:load_guard(m0['heldout_dataset'],m0['heldout_sequence_id'],models[1],keys))
must_reject('unknown_sequence',lambda:load_guard('UNKNOWN','UNKNOWN',f0,keys))
must_reject('missing_model',lambda:load_guard(m0['heldout_dataset'],m0['heldout_sequence_id'],mr/'does_not_exist.json',keys))
must_reject('wrong_feature_order',lambda:load_guard(m0['heldout_dataset'],m0['heldout_sequence_id'],f0,keys[::-1]))
must_reject('nonfinite_input',lambda:predict(m0,[0,0,0,0,float('nan')]))
with (out/'P5NORM_M1_LOADER_GUARD_TEST.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
print('PASS models=21 predictions=39 max_abs_difference=',maxerr,'guard cases=5')
