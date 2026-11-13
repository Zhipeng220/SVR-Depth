"""P4BE-1.0 sequence-level exploratory fit; does not modify original P4-B."""
import csv,json,hashlib,math,sys
from pathlib import Path
import numpy as np
root=Path.cwd(); here=root/'docs/m05x/p4ab/p4b_sequence_exploratory_fit';source=root/'docs/m05x/p4ab/P4A_PAIRED_DATA_AUDIT.csv';model=root/'docs/m05x/p4ab/P4B_MODEL_IMPLEMENTATION.py'; manifest=root/'docs/m05x/p3c1_formal/anchor_manifest.json'
norm=['global_l2','l0_l2','l1_l2','l2_l2','l3_l2'];c3=['c3_l0','c3_l1','c3_l2','c3_l3']; features=norm+c3
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writejson(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def writecsv(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if sys.argv[1:] == ['freeze']:
 path=here/'P4BE_CONFIG.json'
 if path.exists():raise SystemExit('config exists; refusing overwrite')
 writejson(path,{'version':'P4BE-1.0','purpose':'exploratory, sequence labels are not certified independent physical recordings','data':str(source.relative_to(root)),'data_sha256':sha(source),'formal_manifest_sha256':sha(manifest),'original_model_sha256':sha(model),'fold_key':['dataset','sequence_id'],'expected_anchors':39,'expected_sequences':21,'expected_by_domain':{'Bonn':[9,5],'TUM':[15,8],'ScanNet':[15,8]},'models':{'M0':'training-fold anchor mean','M1':norm,'M2':features},'ridge_alpha':1.0,'standardization':'training-fold anchor mean and population standard deviation; zero sd -> 1; unpenalized intercept','main_metric_unit':'sequence: mean anchor truth and mean anchor OOF prediction; each sequence equal weight','secondary_metric_unit':'anchor; each anchor equal weight','bootstrap':{'seed':20261110,'replicates':10000,'method':'percentile resampling sequence-level fixed OOF errors; not model retraining; paired M1-M2','physical_inference':False},'coefficient_analysis':'mean absolute training-fold standardized coefficients, exploratory only; correlated features and fold-specific scaling preclude attribution','history':'pre-outcome for this run, inputs already used in prior development','physical_certification':'UNKNOWN; must not claim physical independence','no_gpu':True})
 print('CONFIG FROZEN',sha(path));sys.exit(0)
config=json.loads((here/'P4BE_CONFIG.json').read_text(encoding='utf8'))
assert config['data_sha256']==sha(source) and config['formal_manifest_sha256']==sha(manifest) and config['original_model_sha256']==sha(model)
with source.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
fm=json.loads(manifest.read_text(encoding='utf8'))['formal_anchors']; expected={r['anchor_id']:(r['dataset'],r['sequence']) for r in fm}
assert len(rows)==len(expected)==39 and len({r['anchor_id'] for r in rows})==39 and set(expected)=={r['anchor_id'] for r in rows}
for r in rows:
 assert (r['dataset'],r['sequence_id'])==expected[r['anchor_id']] and r['matched_evidence']=='PASS' and r['candidate']=='V5-native'
 assert all(math.isfinite(float(r[k])) for k in features+['U_H4','O_H4_AbsRel','V5_H4_AbsRel']) and abs(float(r['U_H4'])-(float(r['O_H4_AbsRel'])-float(r['V5_H4_AbsRel'])))<1e-15
groups=sorted({(r['dataset'],r['sequence_id']) for r in rows});assert len(groups)==21
for d,counts in config['expected_by_domain'].items():assert [sum(r['dataset']==d for r in rows),sum(g[0]==d for g in groups)]==counts

def ridge(x,y,test):
 mu=x.mean(0); sd=x.std(0);sd=np.where(sd==0,1,sd);X=(x-mu)/sd;Z=(test-mu)/sd; intercept=y.mean();beta=np.linalg.solve(X.T@X+np.eye(x.shape[1]),X.T@(y-intercept));return intercept+Z@beta,intercept+X@beta,beta
def rank(x):
 order=np.argsort(x,kind='stable');out=np.empty(len(x));i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[i]]==x[order[j]]:j+=1
  out[order[i:j]]=(i+j-1)/2;i=j
 return out
def metrics(y,p):
 y=np.array(y,float);p=np.array(p,float);e=p-y;a=rank(y);b=rank(p);corr=float(np.corrcoef(a,b)[0,1]) if len(y)>=3 and a.std()>0 and b.std()>0 else None
 return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'Spearman':corr}

# Verify numerical equivalence on synthetic data against the original primitive.
sys.path.insert(0,str(model.parent));import P4B_MODEL_IMPLEMENTATION as original
rg=np.random.default_rng(734); sx=rg.normal(size=(13,9));sy=rg.normal(size=13);st=rg.normal(size=(3,9));p,_,_=ridge(sx,sy,st);q,_=original.ridge_predict(sx,sy,st,synthetic_only=True);assert np.allclose(p,q,rtol=0,atol=1e-12)
X=np.array([[float(r[k]) for k in features] for r in rows]);y=np.array([float(r['U_H4']) for r in rows]); pred={m:np.full(len(rows),np.nan) for m in ['M0','M1','M2']};folds=[]
for fold,(d,s) in enumerate(groups):
 te=np.array([(r['dataset'],r['sequence_id'])==(d,s) for r in rows]);tr=~te
 assert not ({(rows[i]['dataset'],rows[i]['sequence_id']) for i in np.flatnonzero(tr)} & {(d,s)})
 pred['M0'][te]=y[tr].mean();p1,t1,b1=ridge(X[tr,:5],y[tr],X[te,:5]);p2,t2,b2=ridge(X[tr],y[tr],X[te]);pred['M1'][te]=p1;pred['M2'][te]=p2
 folds.append({'fold':fold,'dataset':d,'sequence_id':s,'train_n':int(tr.sum()),'test_n':int(te.sum()),'train_mean_y':float(y[tr].mean()),'train_M1_RMSE':float(np.sqrt(np.mean((t1-y[tr])**2))),'train_M2_RMSE':float(np.sqrt(np.mean((t2-y[tr])**2))),**{f'M2_coef_{k}':float(b2[i]) for i,k in enumerate(features)}})
assert all(np.isfinite(v).all() for v in pred.values())
oof=[]
for i,r in enumerate(rows):
 q={'anchor_id':r['anchor_id'],'dataset':r['dataset'],'sequence_id':r['sequence_id'],'U_H4':float(y[i]),'O_H4':float(r['O_H4_AbsRel']),'V5_H4':float(r['V5_H4_AbsRel'])}
 for m in pred:q[m+'_prediction']=float(pred[m][i]);q[m+'_error']=float(pred[m][i]-y[i]);q[m+'_abs_error']=float(abs(pred[m][i]-y[i]))
 oof.append(q)
seq=[]
for d,s in groups:
 ix=[i for i,r in enumerate(rows) if (r['dataset'],r['sequence_id'])==(d,s)];q={'dataset':d,'sequence_id':s,'anchor_n':len(ix),'U_H4_mean':float(y[ix].mean())}
 for m in pred:
  v=float(pred[m][ix].mean());q[m+'_prediction_mean']=v;q[m+'_error']=v-q['U_H4_mean'];q[m+'_abs_error']=abs(q[m+'_error'])
 q['paired_abs_error_M1_minus_M2']=q['M1_abs_error']-q['M2_abs_error'];seq.append(q)
def comp(rs,unit,domain):
 truth=[r['U_H4_mean'] if unit=='sequence' else r['U_H4'] for r in rs];ms={}
 for m in pred:
  key=m+'_prediction_mean' if unit=='sequence' else m+'_prediction';ms[m]=metrics(truth,[r[key] for r in rs])
 result={'domain':domain,'unit':unit,'n':len(rs)}
 for m in ms:
  for k,v in ms[m].items():result[m+'_'+k]=v
 result['Delta_MAE_M0_minus_M1']=ms['M0']['MAE']-ms['M1']['MAE'];result['Delta_MAE_M1_minus_M2']=ms['M1']['MAE']-ms['M2']['MAE'];result['Delta_RMSE_M1_minus_M2']=ms['M1']['RMSE']-ms['M2']['RMSE'];result['M1_vs_M0_MAE_improvement_rate']=(ms['M0']['MAE']-ms['M1']['MAE'])/ms['M0']['MAE'] if ms['M0']['MAE'] else None;result['M2_vs_M1_MAE_improvement_rate']=(ms['M1']['MAE']-ms['M2']['MAE'])/ms['M1']['MAE'] if ms['M1']['MAE'] else None
 return result
comparison=[comp(seq,'sequence','ALL'),comp(oof,'anchor','ALL')];domains=[]
for d in ['Bonn','TUM','ScanNet']:
 for unit,rs in [('sequence',seq),('anchor',oof)]:
  x=comp([r for r in rs if r['dataset']==d],unit,d);x['anchor_n']=sum(r['dataset']==d for r in oof);x['sequence_n']=sum(r['dataset']==d for r in seq);domains.append(x)
boots=[];rng=np.random.default_rng(config['bootstrap']['seed'])
for d in ['ALL','Bonn','TUM','ScanNet']:
 q=[r for r in seq if d=='ALL' or r['dataset']==d]; arr=np.array([r['paired_abs_error_M1_minus_M2'] for r in q]); draws=arr[rng.integers(0,len(arr),size=(10000,len(arr)))].mean(axis=1)
 boots.append({'domain':d,'sequence_n':len(q),'statistic':'MAE(M1)-MAE(M2), sequence means','estimate':float(arr.mean()),'lower_2_5_percentile':float(np.quantile(draws,.025)),'upper_97_5_percentile':float(np.quantile(draws,.975)),'resamples':10000,'seed':20261110,'interpretation':'exploratory fixed-OOF sequence bootstrap; sequence not proven independent; no refit'})
writecsv(here/'P4BE_OOF_PREDICTIONS.csv',oof);writecsv(here/'P4BE_SEQUENCE_METRICS.csv',seq);writecsv(here/'P4BE_DOMAIN_METRICS.csv',domains);writecsv(here/'P4BE_MODEL_COMPARISON.csv',comparison);writecsv(here/'P4BE_BOOTSTRAP_RESULTS.csv',boots);writecsv(here/'P4BE_FOLD_DIAGNOSTICS.csv',folds)

coefs={k:float(np.mean([abs(f['M2_coef_'+k]) for f in folds])) for k in c3};overall=comparison[0];anchor=comparison[1]
def f(v):return '未定义' if v is None else f'{v:.6f}'
lines=['# P4BE-1.0 序列分组探索性拟合报告','','**本轮按研究负责人明确授权，使用真实标签完成 21 折 sequence 留一拟合；序列物理独立性未经认证，结果仅供开发探索。** 这不解除原实现的合成数据保护，也不构成 P5 许可。','','## 数据和复现','','- 按冻结 CSV 与 Formal manifest 独立核验：39 anchors、21 sequence；Bonn 9/5、TUM 15/8、ScanNet 15/8。U = O−V5，候选均为 V5-native。','- 以 (dataset, sequence_id) 作为组键；每折完整留出该序列，训练折 anchor 的均值/总体标准差标准化；0 方差设 1。M0 为训练折标签均值；M1 五个 norm，M2 再加四层 C3；alpha=1、截距不惩罚。','- 与原实现同一数学公式的独立 CPU 实验脚本保存在本目录；原实现只允许合成拟合并强制认证物理组，未修改、未将 unknown 伪装为 certified。在固定合成数据上与原 ridge_predict 最大绝对误差 ≤1e−12。','- 主指标：每条 sequence 先对真实标签和 OOF 预测分别取 anchor 均值，再计算误差，各 sequence 等权；另报 anchor 等权结果。保留完整 39 个 anchor 预测与 21 个 sequence 结果。','', '## 汇总比较（sequence 等权）','','| 模型 | MAE | RMSE | Spearman |','|---|---:|---:|---:|']
for m in pred:lines.append(f"| {m} | {f(overall[m+'_MAE'])} | {f(overall[m+'_RMSE'])} | {f(overall[m+'_Spearman'])} |")
lines += ['',f"M1 相对 M0 的 MAE 改善：{f(overall['Delta_MAE_M0_minus_M1'])}（比例 {f(overall['M1_vs_M0_MAE_improvement_rate'])}）；M2 相对 M1：ΔMAE = {f(overall['Delta_MAE_M1_minus_M2'])}，ΔRMSE = {f(overall['Delta_RMSE_M1_minus_M2'])}，MAE 改善比例 {f(overall['M2_vs_M1_MAE_improvement_rate'])}。正值代表后者更优。",f"anchor 等权 ΔMAE(M1−M2) = {f(anchor['Delta_MAE_M1_minus_M2'])}，与 sequence 等权口径不同。",'M0 的 OOF 均值来自不同训练折；留出的高标签会降低其训练均值，留出的低标签则会提高训练均值，因此接近 −1 的 M0 Spearman 是留一均值基线的机械现象，不表示该常数基线学到了反向预测关系。','', '## 数据域比较','','| 域 | anchor / sequence | M0 MAE / RMSE | M1 MAE / RMSE | M2 MAE / RMSE | ΔMAE M1−M2 |','|---|---:|---|---|---|---:|']
for r in domains:
 if r['unit']=='sequence':lines.append(f"| {r['domain']} | {r['anchor_n']} / {r['sequence_n']} | {f(r['M0_MAE'])} / {f(r['M0_RMSE'])} | {f(r['M1_MAE'])} / {f(r['M1_RMSE'])} | {f(r['M2_MAE'])} / {f(r['M2_RMSE'])} | {f(r['Delta_MAE_M1_minus_M2'])} |")
lines+=['','各域 Spearman、anchor 等权指标与全部单序列配对误差在 CSV 中。', '', '## 序列稳定性与 Bootstrap', '',f"M2 比 M1 的序列绝对误差更小：{sum(r['paired_abs_error_M1_minus_M2']>0 for r in seq)}/21；更大：{sum(r['paired_abs_error_M1_minus_M2']<0 for r in seq)}/21；相同：{sum(r['paired_abs_error_M1_minus_M2']==0 for r in seq)}/21。"]
for b in boots:lines.append(f"- {b['domain']}：sequence 重采样 ΔMAE {b['estimate']:+.6f}，探索性百分位区间 [{b['lower_2_5_percentile']:+.6f}, {b['upper_97_5_percentile']:+.6f}]，n={b['sequence_n']}。")
lines += ['','区间只重采样固定 OOF 序列误差，不重新训练模型；各 OOF 预测的训练集相互重叠，而且 sequence 不是已认证的独立物理录像，因此该区间不能解释为物理场景或总体推断。','','## C3 层与过拟合线索','','M2 训练折的标准化 C3 系数绝对值均值：'+ '；'.join(f'{k}={v:.6g}' for k,v in coefs.items())+'。这只是系数尺度的描述；训练折标准差不同、特征相关且样本少，不可将最大系数解释为可识别的层贡献或因果归因。未做逐层消融。',f"训练折平均 RMSE：M1={np.mean([q['train_M1_RMSE'] for q in folds]):.6f}，M2={np.mean([q['train_M2_RMSE'] for q in folds]):.6f}；这是训练样本平均和折平均，与 OOF 序列平均 RMSE 的单位及权重不同，只是过拟合线索，不能作严格的泛化差距估计。",'', '## 判定与停止条件','','- M1 相对 M0：'+('本次 sequence 等权 MAE 改善。' if overall['Delta_MAE_M0_minus_M1']>0 else '本次 sequence 等权 MAE 未改善。'),'- M2 相对 M1：'+('本次 sequence 等权 MAE 改善。' if overall['Delta_MAE_M1_minus_M2']>0 else '本次 sequence 等权 MAE 未改善。'),'- C3 增量只能描述为此数据/此分折的联合 M2 与 M1 差异，不能证明独立物理场景或因果增量；域间和单序列差异见上表与结果表。','- 39 anchor 的物理来源、场景相关性和完整历史仍未认证，无法支持可靠的跨场景稳定性或闭环收益结论。即使某个开发指标有利，也不具备自动进入 P5 的证据；等待研究负责人审批。','- 没有训练 VeloDepth、修改 checkpoint、启动 GPU 或更改历史结果。']
(here/'P4BE_EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
files=['P4BE_OOF_PREDICTIONS.csv','P4BE_SEQUENCE_METRICS.csv','P4BE_DOMAIN_METRICS.csv','P4BE_MODEL_COMPARISON.csv','P4BE_BOOTSTRAP_RESULTS.csv','P4BE_FOLD_DIAGNOSTICS.csv','P4BE_EXPERIMENT_REPORT.md','P4BE_CONFIG.json','experiment.py','verify_independent.py'];(here/'SHA256SUMS.txt').write_text('\n'.join(sha(here/f)+'  '+f for f in files)+'\n',encoding='utf8')
print('Completed 21 folds; overall',json.dumps(overall))
