"""Read-only P2-B tensor audit and predeclared sequence-level P3-A analysis."""
import csv,hashlib,json,math,zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'docs/m05x/p2b';OUT=ROOT/'docs/m05x/p3a';EPS=1e-12;REPS=5000;SEED=20261025
def relative_from_sums(delta_squared,base_squared):return math.sqrt(delta_squared)/(math.sqrt(base_squared)+EPS)
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rank(a):
 a=np.asarray(a,dtype=float);order=np.argsort(a,kind='mergesort');out=np.empty(len(a));i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[order[j]]==a[order[i]]:j+=1
  out[order[i:j]]=(i+j-1)/2+1;i=j
 return out
def rho(x,y):
 x=rank(x);y=rank(y);x-=x.mean();y-=y.mean();den=np.sqrt(np.dot(x,x)*np.dot(y,y));return float(np.dot(x,y)/den) if den else None
def bootstrap(x,y,seed=SEED,reps=REPS):
 rng=np.random.default_rng(seed);n=len(x);vals=[]
 for _ in range(reps):
  i=rng.integers(0,n,n);v=rho(np.asarray(x)[i],np.asarray(y)[i])
  if v is not None:vals.append(v)
 return {'low':float(np.quantile(vals,.025)) if len(vals)>=.8*reps else None,'high':float(np.quantile(vals,.975)) if len(vals)>=.8*reps else None,'valid_replicates':len(vals),'undefined_replicates':reps-len(vals),'reps':reps,'seed':seed}
def centered(groups):
 x=[];y=[]
 for group in groups:
  a=np.array([v['R'] for v in group]);b=np.array([v['H4'] for v in group]);x.extend(a-a.mean());y.extend(b-b.mean())
 return rho(x,y)
def centered_boot(groups):
 rng=np.random.default_rng(SEED);values=[]
 for _ in range(REPS):
  sample=[[g[i] for i in rng.integers(0,len(g),len(g))] for g in groups];v=centered(sample)
  if v is not None:values.append(v)
 return {'low':float(np.quantile(values,.025)) if len(values)>=.8*REPS else None,'high':float(np.quantile(values,.975)) if len(values)>=.8*REPS else None,'valid_replicates':len(values),'undefined_replicates':REPS-len(values),'reps':REPS,'seed':SEED,'resampling':'within dataset, recenter every draw'}
def write_csv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def compute():
 manifest=json.loads((SRC/'evidence_manifest.json').read_text(encoding='utf8'));validation=json.loads((SRC/'invariant_validation.json').read_text(encoding='utf8'))
 if manifest['status']!='PASS' or validation['status']!='PASS':raise ValueError('P2-B hard validation is not PASS')
 with (SRC/'paired_anchor_results.csv').open(encoding='utf-8-sig',newline='') as f:anchor_rows=list(csv.DictReader(f))
 with (SRC/'sequence_level_results.csv').open(encoding='utf-8-sig',newline='') as f:prior_seq=list(csv.DictReader(f))
 if len(anchor_rows)!=42 or len(prior_seq)!=21:raise ValueError('unexpected sample counts')
 paired=[];hashes={};groups=defaultdict(list)
 for n,row in enumerate(anchor_rows,1):
  aid=row['anchor_id'];folder=SRC/'runs'/aid;meta=json.loads((folder/'capture_metadata.json').read_text(encoding='utf8'));npz=folder/'evidence_arrays.npz';actual=file_sha(npz)
  file_records=manifest['anchors'][aid]['files_sha256']
  def expected_file(name):
   matches=[v for k,v in file_records.items() if k.replace('\\','/').endswith('/'+name)]
   if len(matches)!=1:raise ValueError('missing or ambiguous P2-B file manifest '+aid+' '+name)
   return matches[0]
  if actual!=meta['evidence_npz_sha256'] or actual!=expected_file('evidence_arrays.npz'):raise ValueError('file digest mismatch '+aid)
  if file_sha(folder/'capture_metadata.json')!=expected_file('capture_metadata.json'):raise ValueError('metadata digest mismatch '+aid)
  record=manifest['anchors'][aid]['arrays'];aliases=meta['array_aliases'];hashes[aid]={'npz_sha256':actual,'metadata_sha256':file_sha(folder/'capture_metadata.json')}
  sum_d=sum_o=sum_t=sum_p=sum_tp=0.;layer=[];shapes=[]
  with np.load(npz,allow_pickle=False) as z:
   for i in range(4):
    values=[]
    for prefix in ('committed_O','committed_T','parent_O'):
     key=f'{prefix}__feature{i}';ref=aliases.get(key,key);a=z[ref]
     if key not in record or list(a.shape)!=record[key]['shape'] or str(a.dtype)!=record[key]['dtype'] or array_sha(a)!=record[key]['sha256']:raise ValueError('array digest/shape mismatch '+aid+' '+key)
     values.append(a.astype(np.float64));shapes.append({'key':key,'shape':list(a.shape),'dtype':str(a.dtype)})
    base,target,parent=values
    if base.shape!=target.shape or base.shape!=parent.shape:raise ValueError('state shape mismatch '+aid)
    delta=target-base;d=float(np.sum(delta*delta));o=float(np.sum(base*base));t=float(np.sum(target*target));p=float(np.sum(parent*parent));tp=float(np.sum(target*parent))
    sum_d+=d;sum_o+=o;sum_t+=t;sum_p+=p;sum_tp+=tp
    old=json.loads(row['feature_update_norms'])[f'feature{i}']['relative_l2'];computed=math.sqrt(d)/math.sqrt(o) if o else None
    if computed is None or old is None or not math.isclose(computed,float(old),rel_tol=1e-10,abs_tol=1e-12):raise ValueError('P2-B layer relative norm mismatch '+aid)
    layer.append(computed)
  cos=sum_tp/math.sqrt(sum_t*sum_p) if sum_t and sum_p else None
  if cos is None:raise ValueError('zero norm for cosine '+aid)
  x={'anchor_id':aid,'dataset':row['dataset'],'sequence':row['sequence'],'age':int(row['age']),'R':relative_from_sums(sum_d,sum_o),'delta_l2':math.sqrt(sum_d),'base_l2':math.sqrt(sum_o),'cos_candidate_parent':cos,'H1':float(row['T_minus_O_H1']),'H4':float(row['T_minus_O_H4']),'layer_relative':layer}
  paired.append(x);groups[(x['dataset'],x['sequence'])].append(x);hashes[aid]['feature_arrays']=shapes
  if n%10==0:print('AUDITED',n,'/42',flush=True)
 seq=[];prior={(r['dataset'],r['sequence']):r for r in prior_seq}
 for key,rows in sorted(groups.items()):
  if sorted(r['age'] for r in rows)!=[1,4]:raise ValueError('wrong anchor coverage '+str(key))
  q={'dataset':key[0],'sequence':key[1],'n_anchors':2}
  for field in ('R','delta_l2','base_l2','cos_candidate_parent','H1','H4'):q[field]=float(np.mean([r[field] for r in rows]))
  for i in range(4):q[f'layer{i}_relative_mean']=float(np.mean([r['layer_relative'][i] for r in rows]))
  p=prior[key]
  for horizon in ('H1','H4'):
   if not math.isclose(q[horizon],float(p[horizon+'_T_minus_O']),rel_tol=1e-12,abs_tol=1e-14):raise ValueError('prior sequence mismatch '+str(key))
  seq.append(q)
 if len(seq)!=21:raise ValueError('expected 21 sequences')
 return paired,seq,hashes

def outputs(paired,seq,hashes):
 summary={'definition':'R uses concatenated four levels committed_T vs committed_O; epsilon=1e-12','seed':SEED,'sample_unit':'recording','domain':{},'centered':{},'opposite_sign_pairs':{}}
 loo=[];distribution=[]
 for domain in ('Bonn','TUM','ScanNet'):
  g=[x for x in seq if x['dataset']==domain];x=np.array([r['R'] for r in g]);y=np.array([r['H4'] for r in g]);y1=np.array([r['H1'] for r in g]);full=rho(x,y)
  summary['domain'][domain]={'n':len(g),'rho_H4':full,'rho_H4_bootstrap95':bootstrap(x,y),'rho_H1':rho(x,y1),'rho_H1_bootstrap95':bootstrap(x,y1),'R_min':float(x.min()),'R_median':float(np.median(x)),'R_max':float(x.max()),'positive_H4':int((y>0).sum()),'negative_H4':int((y<0).sum()),'rho_cosine_H4_exploratory':rho([r['cos_candidate_parent'] for r in g],y)}
  for i,r in enumerate(g):
   v=rho(np.delete(x,i),np.delete(y,i));loo.append({'dataset':domain,'omitted_sequence':r['sequence'],'n_remaining':len(g)-1,'rho_full':full,'rho_leave_one_out':v,'sign_flips':bool(v is not None and full is not None and np.sign(v)!=np.sign(full))})
  pairs=[(abs(a['R']-b['R']),a,b) for i,a in enumerate(g) for b in g[i+1:] if a['H4']*b['H4']<0]
  if pairs:
   _,a,b=min(pairs,key=lambda item:item[0]);summary['opposite_sign_pairs'][domain]={'sequence_a':a['sequence'],'sequence_b':b['sequence'],'R_a':a['R'],'R_b':b['R'],'H4_a':a['H4'],'H4_b':b['H4'],'relative_R_difference':abs(a['R']-b['R'])/((a['R']+b['R'])/2)}
  else:summary['opposite_sign_pairs'][domain]=None
  for r in g:distribution.append({'dataset':domain,'sequence':r['sequence'],'R':r['R'],'delta_l2':r['delta_l2'],'base_l2':r['base_l2'],'H4':r['H4']})
 gg=[[x for x in seq if x['dataset']==d] for d in ('Bonn','TUM','ScanNet')]
 summary['centered']={'rho_H4':centered(gg),'bootstrap95':centered_boot(gg),'meaning':'Spearman on dataset-mean-centered R and H4; stratified sequence resampling'}
 write_csv(OUT/'sequence_feature_effect.csv',seq);write_csv(OUT/'feature_norm_distribution.csv',distribution);write_csv(OUT/'leave_one_sequence_out.csv',loo)
 (OUT/'correlation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
 (OUT/'anchor_feature_audit.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
 inputs=['P2B_REPORT.md','sequence_level_results.csv','paired_anchor_results.csv','dataset_summary.json','confidence_intervals.json','evidence_manifest.json','invariant_validation.json','sequence_manifest.json']
 evidence={'input_files':{str(SRC/p):file_sha(SRC/p) for p in inputs},'analysis_plan_sha256':file_sha(OUT/'ANALYSIS_PLAN.md'),'analysis_script_sha256':file_sha(Path(__file__)),'anchor_sources':hashes,'npz_and_12_feature_arrays_per_anchor_verified':True}
 (OUT/'evidence_manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
 for j,domain in enumerate(('Bonn','TUM','ScanNet')):
  g=[r for r in seq if r['dataset']==domain];x=[r['R'] for r in g];y=[r['H4'] for r in g]
  axes[0,j].scatter(x,y,c=['#cc554a' if v>0 else '#3776a6' for v in y],s=40);axes[0,j].axhline(0,color='gray',lw=.8);axes[0,j].set_title(f"{domain} (n={len(g)}), rho={summary['domain'][domain]['rho_H4']:.3f}");axes[0,j].set_xlabel('Relative feature update R');axes[0,j].set_ylabel('H4 AbsRel(T) − AbsRel(O)')
  axes[1,j].hist(x,bins=min(5,len(g)),color='#6d8ca4',edgecolor='white');axes[1,j].set_xlabel('Relative feature update R');axes[1,j].set_ylabel('Recordings')
 fig.suptitle('P3-A: recording-level exploratory associations; development data',fontsize=13);fig.savefig(OUT/'feature_update_scatter.png',dpi=180);fig.savefig(OUT/'feature_update_scatter.pdf');plt.close(fig)
 print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':outputs(*compute())
