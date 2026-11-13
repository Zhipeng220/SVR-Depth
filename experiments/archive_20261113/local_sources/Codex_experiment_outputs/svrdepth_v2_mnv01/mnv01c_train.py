"""MNV-01C scene-held-out exploratory logistic baselines; no GPU.

Input: outputs/svrdepth_v2_mnv01/MNV01_HISTORY_FEATURES.csv (or argv[1]).
Required: physical_scene_id and label (HARMFUL/BENEFICIAL/NEUTRAL), or
U_absrel/utility_absrel, where -0.001/+0.001 bounds define labels. Optional
numeric feature fields are listed below. Missing values stay missing in the
input; medians and scaling are estimated only on each training fold. This
script deliberately never substitutes dataset/scene identifiers as features.
"""
from pathlib import Path
import csv,sys,math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT=Path('outputs/svrdepth_v2_mnv01')
source=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'MNV01_HISTORY_FEATURES.csv'
GROUPS={
 'M1':['overall_update_l2'],
 'M2':['layer1_update_l2','layer2_update_l2','layer3_update_l2','layer4_update_l2'],
}
CTX=[f'{field}_{stat}' for field in ('radius','confidence','rays') for stat in ('mean','std','p10','p50','p90')]
HIST=['history_full_count','recent5_commit_rate','frames_since_last_full','history_hold_count','recent_update_trend']
GROUPS['M3']=GROUPS['M2']+CTX
GROUPS['M4']=GROUPS['M3']+HIST
ALIASES={'layer1_update_l2':['update_l1'],'layer2_update_l2':['update_l2'],'layer3_update_l2':['update_l3'],'layer4_update_l2':['update_l4'],
         'radius_mean':['radius'],'confidence_mean':['confidence'],'rays_mean':['rays'],
         'history_full_count':['historical_full_commit_count','full_commit_count'], 'history_hold_count':['historical_hold_count','hold_count'],
         'frames_since_last_full':['time_since_last_full_commit','time_since_previous_commit'], 'recent5_commit_rate':['recent_5_commit_rate'], 'recent_update_trend':['recent_norm_trend']}
def val(row,name):
 for k in [name]+ALIASES.get(name,[]):
  try:
   v=float(row.get(k,''))
   if math.isfinite(v):return v
  except (ValueError,TypeError):pass
 return np.nan
def label(r):
 x=str(r.get('label',r.get('risk_label',''))).strip().upper()
 if x in ['HARMFUL','BENEFICIAL','NEUTRAL']:return x
 u=val(r,'U_absrel')
 if not np.isfinite(u):u=val(r,'utility_absrel')
 if not np.isfinite(u):return 'UNKNOWN'
 return 'HARMFUL' if u < -.001 else 'BENEFICIAL' if u > .001 else 'NEUTRAL'
def auc(y,p):
 pos=p[y==1];neg=p[y==0]
 return float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:]))) if len(pos) and len(neg) else None
def ap(y,p):
 if not np.any(y==1):return None
 order=np.argsort(-p,kind='stable');y=y[order];p=p[order]
 ends=np.r_[np.where(p[:-1]!=p[1:])[0],len(p)-1];tp=np.cumsum(y)[ends];fp=(ends+1)-tp
 return float(np.sum((tp-np.r_[0,tp[:-1]])/tp[-1]*tp/(tp+fp)))
def metrics(y,p):
 pred=p>=.5;pos=y==1;neg=y==0;recall=float(np.mean(pred[pos])) if pos.any() else None;tnr=float(np.mean(~pred[neg])) if neg.any() else None
 return dict(auroc=auc(y,p),auprc=ap(y,p),balanced_accuracy=(recall+tnr)/2 if recall is not None and tnr is not None else None,brier=float(np.mean((p-y)**2)),harmful_recall=recall,beneficial_false_blocking_rate=float(np.mean(pred[neg])) if neg.any() else None)
def fit(X,y):
 # sklearn-style summed negative log likelihood + 1/(2*C)*||coef||^2; C=1.
 def objective(z):
  v=X@z[:-1]+z[-1];loss=np.logaddexp(0,v).sum()-np.dot(y,v)+.5*np.dot(z[:-1],z[:-1]);d=expit(v)-y
  return loss,np.r_[X.T@d+z[:-1],d.sum()]
 opt=minimize(objective,np.zeros(X.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':1000})
 if not opt.success:raise RuntimeError('optimizer: '+str(opt.message))
 return opt.x
with source.open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
for r in original:r['_label']=label(r);r['_scene']=str(r.get('physical_scene_id','')).strip()
neutral=sum(r['_label']=='NEUTRAL' for r in original);unknown=sum(r['_label']=='UNKNOWN' or r['_scene'].upper() in ('','UNKNOWN','NA','N/A','NOT_OBSERVED') for r in original)
eligible=[r for r in original if r['_label'] in ('HARMFUL','BENEFICIAL') and r['_scene'].upper() not in ('','UNKNOWN','NA','N/A','NOT_OBSERVED')]
scenes=sorted(set(r['_scene'] for r in eligible));out=[]
for scene in scenes:
 train=[r for r in eligible if r['_scene']!=scene];test=[r for r in eligible if r['_scene']==scene]
 y=np.array([int(r['_label']=='HARMFUL') for r in train]);yt=np.array([int(r['_label']=='HARMFUL') for r in test])
 for model in ['M0','M1','M2','M3','M4']:
  result={'heldout_scene':scene,'model':model,'n_train':len(train),'n_test':len(test),'train_harmful':int(y.sum()),'test_harmful':int(yt.sum()),'status':'NOT_OBSERVED','features_used':'','reason':''}
  if len(y)==0 or (model!='M0' and len(np.unique(y))<2):result['reason']='training fold has fewer than two classes';out.append(result);continue
  try:
   if model=='M0':p=np.full(len(test),float(y.mean()))
   else:
    features=GROUPS[model];a=np.array([[val(r,k) for k in features] for r in train]);b=np.array([[val(r,k) for k in features] for r in test]);keep=np.isfinite(a).any(axis=0)
    if not keep.any():raise ValueError('no observed features in training fold')
    a=a[:,keep];b=b[:,keep];used=[k for k,q in zip(features,keep) if q];median=np.nanmedian(a,axis=0);a=np.where(np.isfinite(a),a,median);b=np.where(np.isfinite(b),b,median)
    mean=a.mean(axis=0);std=a.std(axis=0);std=np.where(std>0,std,1);a=(a-mean)/std;b=(b-mean)/std;z=fit(a,y);p=expit(b@z[:-1]+z[-1]);result['features_used']=';'.join(used)
   result.update(metrics(yt,p));result['status']='EXPLORATORY' if len(np.unique(yt))==2 else 'EXPLORATORY_SINGLE_CLASS_TEST';result['reason']='AUROC/other class-specific metrics unavailable when absent from heldout scene' if len(np.unique(yt))<2 else ''
  except Exception as e:result['reason']=str(e)
  out.append(result)
fields=['heldout_scene','model','n_train','n_test','train_harmful','test_harmful','status','features_used','reason','auroc','auprc','balanced_accuracy','brier','harmful_recall','beneficial_false_blocking_rate']
ROOT.mkdir(parents=True,exist_ok=True)
with (ROOT/'MNV01_SCENE_SPLIT_RESULTS.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
summary=[]
for model in ['M0','M1','M2','M3','M4']:
 rr=[r for r in out if r['model']==model and r['status'].startswith('EXPLORATORY')];z={'model':model,'folds_observed':len(rr),'folds_total':len(scenes),'status':'EXPLORATORY_ONLY','eligible':len(eligible),'neutral':neutral,'unknown_or_scene_missing':unknown,'scene_count':len(scenes)}
 for k in fields[-6:]:
  x=[r[k] for r in rr if r.get(k) is not None];z[k]=float(np.mean(x)) if x else None
 summary.append(z)
with (ROOT/'MNV01_PREDICTIVE_BASELINES.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('n',len(eligible),'harmful',sum(r['_label']=='HARMFUL' for r in eligible),'beneficial',sum(r['_label']=='BENEFICIAL' for r in eligible),'neutral',neutral,'unknown_or_scene_missing',unknown,'scenes',len(scenes))
print('Exploratory only; means are unweighted held-out scene averages. Missing metrics omitted, not zero.')
