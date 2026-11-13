"""Sequence-grouped development analysis; rejects unverified physical grouping."""
import numpy as np
import ridge
def holm(values):
 order=sorted((p,i) for i,p in enumerate(values) if p is not None);adjusted=[None]*len(values);prior=0.
 for k,(p,i) in enumerate(order):
  prior=max(prior,min(1.,(len(values)-k)*p));adjusted[i]=prior
 return adjusted
def perm_spearman(x,y,rng,draws):
 obs=ridge.spearman(x,y)
 if obs is None:return None,None
 extreme=0
 for _ in range(draws):
  v=ridge.spearman(x,rng.permutation(y))
  if v is not None and abs(v)>=abs(obs)-1e-15:extreme+=1
 return obs,(extreme+1)/(draws+1)
def residualize(x,controls):
 c=np.column_stack([np.ones(len(x)),np.column_stack([ridge.rank(a) for a in controls.T])])
 z=ridge.rank(x);return z-c@np.linalg.lstsq(c,z,rcond=None)[0]
def perm_pearson(x,y,rng,draws):
 if np.std(x)<1e-10 or np.std(y)<1e-10:return None,None
 obs=float(np.corrcoef(x,y)[0,1]);extreme=0
 for _ in range(draws):
  v=float(np.corrcoef(x,rng.permutation(y))[0,1]);extreme+=abs(v)>=abs(obs)-1e-15
 return obs,(extreme+1)/(draws+1)
def analyze(rows,seed=20261110,permutations=10000):
 ridge.validate(rows);rng=np.random.default_rng(seed);result={}
 for ds in ('Bonn','TUM','ScanNet'):
  subset=[r for r in rows if r['dataset']==ds];groups=sorted({r['physical_group_id'] for r in subset});summ=[]
  for g in groups:
   q=[r for r in subset if r['physical_group_id']==g]
   summ.append({'g':g,'y':np.mean([float(r['U_H4']) for r in q]),'features':np.mean([[float(r[k]) for k in ridge.NORM+ridge.C3] for r in q],axis=0)})
  if len(summ)<5:result[ds]={'status':'BLOCKED_INSUFFICIENT_CERTIFIED_GROUPS','n_groups':len(summ)};continue
  y=np.array([q['y'] for q in summ]);X=np.stack([q['features'] for q in summ]);layer=[];norm=[]
  for i in range(4):
   rho,p=perm_spearman(X[:,i+5],y,rng,permutations)
   # Rank residualization is descriptive; permutation uses fixed residuals.
   if len(summ)<=7:partial,pp=None,None
   else:
    rx=residualize(X[:,i+5],X[:,:5]);ry=residualize(y,X[:,:5])
    partial,pp=perm_pearson(rx,ry,rng,permutations)
   layer.append({'rho':rho,'p':p,'partial_rank_rho':partial,'partial_permutation_p':pp})
   nr,npv=perm_spearman(X[:,i+1],y,rng,permutations);norm.append({'rho':nr,'p':npv})
  for name in ('p','partial_permutation_p'):
   adj=holm([x[name] for x in layer])
   for row,v in zip(layer,adj):row[name+'_holm']=v
  for row,v in zip(norm,holm([x['p'] for x in norm])):row['p_holm']=v
  def dist(select):
   return [{'layer':i,'median':float(np.median(X[select,i+5])),'min':float(np.min(X[select,i+5])),'max':float(np.max(X[select,i+5]))} for i in range(4)] if select.any() else []
  result[ds]={'status':'EXPLORATORY','n_groups':len(groups),'c3':layer,'norm':norm,'positive_groups':int((y>0).sum()),'negative_groups':int((y<0).sum()),'zero_groups':int((y==0).sum()),'beneficial_C3':dist(y>0),'harmful_C3':dist(y<0),'unit':'certified physical group'}
 return result

