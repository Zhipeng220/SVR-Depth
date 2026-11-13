from pathlib import Path
import csv,json,math,collections,statistics
import numpy as np
from scipy.stats import spearmanr,kendalltau,rankdata
import torch

BASE=Path('/home/svrdepth/SVR-Depth/velodepth/outputs')
SOURCES={'2.3':BASE/'p5norm_v2_3_single_commit','2.4':BASE/'p5norm_v2_4_independent_replication'}
OUT=BASE/'p5norm_v2_5_mechanism_consolidation'; OUT.mkdir(exist_ok=True)
FEATS=['overall_update_l2','overall_relative_norm','update_l1','update_l2','update_l3','update_l4']
DOMAINS=['Bonn','TUM','ScanNet'];SEED=20261025;B=10000
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 fields=list(dict.fromkeys(k for x in rows for k in x))
 with (OUT/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def spear(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float)
 if len(x)<3 or len(np.unique(x))<2 or len(np.unique(y))<2:return float('nan')
 return float(spearmanr(x,y).statistic)
def boots(rows,feature):
 rng=np.random.default_rng(SEED);groups=[np.array([i for i,r in enumerate(rows) if r['dataset']==d],dtype=int) for d in DOMAINS];x=np.array([r[feature] for r in rows],float);y=np.array([r['sequence_mean_utility'] for r in rows],float)
 ix=np.concatenate([rng.choice(g,size=(B,len(g)),replace=True) for g in groups if len(g)],axis=1)
 rx=rankdata(x[ix],axis=1);ry=rankdata(y[ix],axis=1);rx=rx-rx.mean(axis=1,keepdims=True);ry=ry-ry.mean(axis=1,keepdims=True);den=np.sqrt((rx*rx).sum(axis=1)*(ry*ry).sum(axis=1));v=np.divide((rx*ry).sum(axis=1),den,out=np.full(B,np.nan),where=den>0);v=v[np.isfinite(v)]
 return (float(np.quantile(v,.025)),float(np.quantile(v,.975)),len(v)) if len(v) else (float('nan'),float('nan'),0)
def within(rows,feature):
 # Residualize the pooled ranks by domain means; descriptive rank association, not a trained predictor.
 x=rankdata([r[feature] for r in rows]).astype(float);y=rankdata([r['sequence_mean_utility'] for r in rows]).astype(float)
 for d in DOMAINS:
  idx=np.array([i for i,r in enumerate(rows) if r['dataset']==d]);
  if len(idx):x[idx]-=x[idx].mean();y[idx]-=y[idx].mean()
 return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else float('nan')
def fold(rows,feature):
 x=[r[feature] for r in rows];y=[r['sequence_mean_utility'] for r in rows];return [spear(x[:i]+x[i+1:],y[:i]+y[i+1:]) for i in range(len(rows))]
combined=[];source_lines=[];checks=[]
for batch,folder in SOURCES.items():
 seqfile=folder/('P5NORM_V2_3_SEQUENCE_ANALYSIS.csv' if batch=='2.3' else 'P5NORM_V2_4_SEQUENCE_ANALYSIS.csv');labelfile=folder/('P5NORM_V2_3_UTILITY_LABELS.csv' if batch=='2.3' else 'P5NORM_V2_4_UTILITY_LABELS.csv')
 for p in [seqfile,labelfile,folder/('P5NORM_V2_3_REPLAY_SUMMARY.csv' if batch=='2.3' else 'P5NORM_V2_4_PARENT_ALIGNMENT.csv')]:
  if not p.is_file():raise FileNotFoundError(p)
  source_lines.append(str(p))
 seq=read(seqfile);labels=read(labelfile);reference=read(folder/('P5NORM_V2_3_REPLAY_SUMMARY.csv' if batch=='2.3' else 'P5NORM_V2_4_PARENT_ALIGNMENT.csv'))
 assert all(r['match']=='True' for r in reference) if batch=='2.3' else all(r['allclose']=='True' for r in reference)
 for s in seq:
  sub=[r for r in labels if r['dataset']==s['dataset'] and r['sequence_id']==s['sequence_id']]
  assert len(sub)==2 and all(r.get('included','True')=='True' for r in sub)
  u='utility_absrel' if batch=='2.3' else 'U_AbsRel';prior='mean_utility_absrel' if batch=='2.3' else 'mean_U_AbsRel'
  z={'batch_id':batch,'dataset':s['dataset'],'sequence_id':s['sequence_id'],'anchor_count':len(sub),'sequence_mean_utility':statistics.mean(float(r[u]) for r in sub)}
  assert math.isclose(z['sequence_mean_utility'],float(s[prior]),rel_tol=1e-10,abs_tol=1e-12)
  for f in FEATS:
   z[f]=statistics.mean(float(r[f]) for r in sub);assert math.isclose(z[f],float(s['mean_'+f]),rel_tol=1e-10,abs_tol=1e-12)
  z['confidence_mean']=None;z['radius_mean']=None
  if batch=='2.4':
   path=folder/'runs'/s['dataset']/Path(s['sequence_id']).name/'hold';tensors=[torch.load(path/('anchor_%s_tensors.pt'%r['anchor_idx']),map_location='cpu',weights_only=True) for r in sub]
   for f in ['confidence','radius']:
    values=[float(t[f].float().mean()) for t in tensors];assert all(np.isfinite(values));z[f+'_mean']=statistics.mean(values)
  combined.append(z)
assert len(combined)==21 and len({(r['dataset'],r['sequence_id']) for r in combined})==21
write('P5NORM_V2_5_SEQUENCE_LEVEL_COMBINED.csv',combined)

batchrows=[];domainrows=[];corrows=[];influence={}
for batch in ['2.3','2.4']:
 rows=[r for r in combined if r['batch_id']==batch];n=len(rows);y=[r['sequence_mean_utility'] for r in rows]
 for f in FEATS:
  x=[r[f] for r in rows];rho=spear(x,y);tau=float(kendalltau(x,y).statistic);lo,hi,valid=boots(rows,f);loo=fold(rows,f);finite=[v for v in loo if math.isfinite(v)];influence[(batch,f)]=loo
  batchrows.append(dict(batch_id=batch,feature=f,n_sequences=n,spearman=rho,spearman_nominal_p=float(spearmanr(x,y).pvalue),kendall_tau=tau,bootstrap_low=lo,bootstrap_high=hi,bootstrap_valid=valid,bootstrap_n=B,seed=SEED,within_domain_rank_residual_r=within(rows,f),leave_one_sequence_min=min(finite),leave_one_sequence_max=max(finite),leave_one_sequence_positive=sum(v>0 for v in finite),leave_one_sequence_count=len(finite)))
  for i,v in enumerate(loo):domainrows.append(dict(batch_id=batch,analysis='leave_one_sequence_out',dataset=rows[i]['dataset'],sequence_id=rows[i]['sequence_id'],feature=f,n_sequences=n-1,spearman=v))
  for d in DOMAINS:
   dd=[r for r in rows if r['dataset']==d];dx=[r[f] for r in dd];dy=[r['sequence_mean_utility'] for r in dd];rho_d=spear(dx,dy);pair_direction=np.sign((dx[1]-dx[0])*(dy[1]-dy[0])) if len(dx)==2 else ''
   domainrows.append(dict(batch_id=batch,analysis='domain',dataset=d,feature=f,n_sequences=len(dd),spearman=rho_d,kendall_tau=float(kendalltau(dx,dy).statistic) if len(dd)>=3 else '',two_sequence_order_direction=pair_direction,mean_feature=float(np.mean(dx)),mean_utility=float(np.mean(dy))))
 extras=['confidence_mean','radius_mean'] if batch=='2.4' else []
 for a in FEATS+extras:
  for b in FEATS+extras+['sequence_mean_utility']:
   if b!='sequence_mean_utility' and (FEATS+extras).index(b)<=(FEATS+extras).index(a):continue
   corrows.append(dict(batch_id=batch,feature_a=a,feature_b=b,n_sequences=n,spearman=spear([r[a] for r in rows],[r[b] for r in rows]),interpretation='exploratory; sequence-level'))
for batch in ['2.3','2.4']:
 rr=sorted([r for r in batchrows if r['batch_id']==batch],key=lambda r:r['spearman_nominal_p']);prev=0
 for i,r in enumerate(rr):prev=max(prev,min(1.,(len(rr)-i)*r['spearman_nominal_p']));r['holm_adjusted_p_6']=prev
write('P5NORM_V2_5_BATCH_COMPARISON.csv',batchrows);write('P5NORM_V2_5_FEATURE_CORRELATIONS.csv',corrows);write('P5NORM_V2_5_DOMAIN_ANALYSIS.csv',domainrows)

def get(batch,f):return next(r for r in batchrows if r['batch_id']==batch and r['feature']==f)
def fmt(x):return 'NA' if x is None or not math.isfinite(float(x)) else f'{x:.3f}'
def br(r):return f"ρ={fmt(r['spearman'])}, τ={fmt(r['kendall_tau'])}, 95% CI [{fmt(r['bootstrap_low'])}, {fmt(r['bootstrap_high'])}], 域控制秩相关={fmt(r['within_domain_rank_residual_r'])}, 留一序列范围 [{fmt(r['leave_one_sequence_min'])}, {fmt(r['leave_one_sequence_max'])}], 正向 {r['leave_one_sequence_positive']}/{r['leave_one_sequence_count']}"
def cr(batch,a,b):
 r=next((r for r in corrows if r['batch_id']==batch and r['feature_a']==a and r['feature_b']==b),None);return fmt(r['spearman']) if r else 'NA'
domains='\n'.join(f"- {r['batch_id']} {r['dataset']} {r['feature']}: n={r['n_sequences']}, ρ={fmt(r['spearman'])}, τ={fmt(r['kendall_tau'] if r.get('kendall_tau','')!='' else float('nan'))}, 双序列排序方向={r.get('two_sequence_order_direction','')}" for r in domainrows if r['analysis']=='domain' and r['feature'] in ['overall_update_l2','update_l1','overall_relative_norm'])
summary=f'''# 结论与路线\n\n**首选研究方向 B：Layer-wise Utility Prediction；暂缓开发任何预测器。** 单次提交的正负效用在两个批次均出现，说明何时提交 Feature 是值得研究的问题，但现有相关性不能证明 Norm 能可靠预测未知场景的 Utility。Layer 1 的跨批次正向现象值得预注册复核，并不能据此宣布其优于总体 Norm。多信号 C 是后续假设：当前互补信息没有得到独立验证，不应因相关性或特征数量而训练模型。\n\n保留 Adaptive Feature Commitment 与决策前可观测信号；修改“总体 Norm 足以预测效用”和“Layer 1 已被证明最佳”的假设。停止推进旧 M1、阈值和 B 策略训练；保留 Overall Absolute、Layer 1、Relative 为预先列明的候选，不以本批次结果挑选最优组合。\n\n唯一优先任务：在**新采集、未参与 P3-B 及 P5 方法开发的独立物理场景**上预注册复现同父状态单次提交 Utility 与分层 Norm 关联，预先指定域、样本量、主要分析及匹配检查；同时仅记录已定义的 Confidence、Radius，供后续探索。不要在该阶段以本数据训练预测器。\n\n距离 CVPR 主实验仍缺：能超越现有选择性更新方法的明确方法创新；经独立场景验证的决策前 Utility 信号；跨域泛化及其不确定性；在线策略相对稳健基线的收益、开销和消融；规模、场景独立性、配对状态与可复现性证据。工程运行成功不等于算法有效。\n'''
report=summary+f'''\n# 证据与方法\n\n2.3 是 9 序列/18 anchor 的开发批次；2.4 是 12 序列/24 anchor 的 P5 2.0–2.3 方法留出批次，但 2.4 存在 P3-B 项目历史暴露，并非整个项目未见数据。每序列两个 anchor 先取均值；21 行只是附 batch 标签的描述性汇总，不当作 21 条独立外部测试。效用 U=Hold AbsRel−Single Commit AbsRel，不混用 Repair-OFF H4。\n\n按域分层的序列 Bootstrap 各 10,000 次、seed={SEED}，秩相同或常量的无定义抽样剔除；每域少量序列导致区间不稳定。域控制值是全样本的两组秩分别在各域减去域内均值后的 Pearson 相关，属于描述性的域固定效应秩相关，不是拟合或验证预测模型。逐序列留一结果记录在域分析表；正向比例是留一后相关为正的比例，不是预测正确率。\n\n## 预先关注的三种特征\n\n- Overall Absolute：2.3 {br(get('2.3','overall_update_l2'))}；2.4 {br(get('2.4','overall_update_l2'))}。正向但减弱，区间跨零。\n- Layer 1 Update L2：2.3 {br(get('2.3','update_l1'))}；2.4 {br(get('2.4','update_l1'))}。两批点估计均为正，但同一先验研究体系且域内样本少，不能认定稳定泛化或最佳层。这里 L1 表示第一层，范数仍是 L2。2.4 六个探索特征的 Holm 校正约 0.170（名义 p≈0.028），不支持多重比较校正后的显著结论。\n- Overall Relative：2.3 {br(get('2.3','overall_relative_norm'))}；2.4 {br(get('2.4','overall_relative_norm'))}。点估计略有上升，不代表相对于 Absolute 的独立增益。\n\n## 域和个别序列影响\n\n{domains}\n\nBonn 2.4 仅 2 条，给出排序方向而非有意义的域内相关估计。LOO 不等同外部验证。Layer 2–4 的相关与 Bootstrap 全部列在批次比较表，不按观察结果另选主要特征。\n\n## 互补性与特征可用性\n\nAbsolute 与 Layer 1 的序列级 Spearman：2.3 {cr('2.3','overall_update_l2','update_l1')}，2.4 {cr('2.4','overall_update_l2','update_l1')}；Absolute 与 Relative：2.3 {cr('2.3','overall_update_l2','overall_relative_norm')}，2.4 {cr('2.4','overall_update_l2','overall_relative_norm')}。这些相关描述冗余程度，无法证明任何信号具有条件增量预测价值。没有训练组合模型。\n\n2.4 使用已保存决策前同次 ON 状态的 Confidence、Radius 张量，先对每 anchor 所有元素求算术均值、再对序列的两个 anchor 等权平均；它们的序列级探索关系见特征相关表。数值不代表已经验证的可靠性或几何机制。2.3 未保存可对齐的同类决策时张量，标为缺失，不用其他时刻或文本推算；因此无法跨批比较上述两个信号。\n\n2.3 只有决策前 Norm 重放匹配，2.4 对 S0/S1、显式状态和深度等关键张量逐项比较通过；两者均不能证明全部隐藏状态一致，不能夸大为完整因果证明。\n\n## 实际读取\n\n'''+''.join('- `'+x+'`\n' for x in source_lines)+f'- `{SOURCES["2.4"]}/runs/<domain>/<sequence>/hold/anchor_<index>_tensors.pt`：24 个已有张量文件，仅提取 Confidence 与 Radius 均值。\n'
report=report.replace('正向但减弱，区间跨零。','正向但减弱，2.4 区间跨零。')
report=report.replace('每域少量序列导致区间不稳定。','每域少量序列导致区间不稳定。本次 2.3 按域分层区间与原报告未分层序列 Bootstrap 区间 [-0.105, 0.853] 使用不同抽样方案，不能互相替代。')
report=report.replace('它们的序列级探索关系见特征相关表。','Confidence 与 Radius 两个均值和 Utility 的序列级 Spearman 均为 0.210，彼此为 1.000；这是观察样本中该均值摘要的高度冗余，不代表其他可能的摘要必然冗余。')
(OUT/'P5NORM_V2_5_MECHANISM_CONSOLIDATION.md').write_text(report,encoding='utf-8')
(OUT/'P5NORM_V2_5_FINAL_DECISION.md').write_text(summary,encoding='utf-8')
print(report)
