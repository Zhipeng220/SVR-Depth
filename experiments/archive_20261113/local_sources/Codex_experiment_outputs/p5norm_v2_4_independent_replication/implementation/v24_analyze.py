from pathlib import Path
import csv,json,io,math,statistics
import numpy as np,cv2,h5py
from PIL import Image
from scipy.stats import spearmanr,pearsonr
root=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_v2_4_independent_replication');R=lambda n:list(csv.DictReader((root/n).open()));seqs=R('P5NORM_V2_4_FROZEN_SEQUENCE_LIST.csv');anchors=R('P5NORM_V2_4_FROZEN_ANCHORS.csv')
def save(name,rows,fields=None):
 with (root/name).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def metric(a,p):
 d=a['dataset'];pred=np.squeeze(np.load(p))
 if d=='Bonn':raw=cv2.imread(a['gt_path'],cv2.IMREAD_UNCHANGED);gt=raw.astype(np.float32)/5000.;mask=np.isfinite(gt)&(gt>0)
 else:
  with h5py.File(Path.home()/'datasets/velodepth_evaluation'/(d+'.hdf5'),'r') as h:raw=np.array(Image.open(io.BytesIO(h[a['gt_path']][:].tobytes())))
  if raw.ndim==3:raw=raw[:,:,0].astype(np.int32)|(raw[:,:,1].astype(np.int32)<<8)|(raw[:,:,2].astype(np.int32)<<16)
  gt=raw.astype(np.float32)/1000.;mask=np.isfinite(gt)&(gt>(.001 if d=='TUM' else .01))&(gt<10.)
 if pred.shape!=gt.shape or not mask.any() or not (np.isfinite(pred[mask])&(pred[mask]>0)).all():raise RuntimeError('invalid prediction/GT')
 g=gt[mask].astype(np.float64);p=pred[mask].astype(np.float64);return dict(AbsRel=float(np.mean(abs(g-p)/g)),RMSE=float(np.sqrt(np.mean((g-p)**2))),delta1=float(np.mean(np.maximum(g/p,p/g)<1.25)),valid_pixels=int(mask.sum()))
holds=[];branches=[];alignment=[];labels=[]
for a in anchors:
 d=a['dataset'];s=a['sequence_id'];i=int(a['anchor_idx']);t=int(a['target_idx']);base=root/'runs'/d/Path(s).name;hp=base/'hold';bp=base/('anchor_'+str(i))
 for x in [hp,bp]:
  if json.loads((x/'status.json').read_text())['status']!='PASS':raise RuntimeError('incomplete worker')
 hn=next(x for x in json.loads((hp/'norm_records.json').read_text()) if x['anchor_idx']==i);bn=json.loads((bp/'norm_records.json').read_text())[0];hs=metric(a,hp/'predictions'/('frame_%06d.npy'%t));bs=metric(a,bp/'predictions'/('frame_%06d.npy'%t));assert hs['valid_pixels']==bs['valid_pixels'];checks=json.loads((bp/'alignment.json').read_text());ok=bool(checks) and all(x['allclose'] for x in checks);alignment+=checks
 r=dict(hn);r.update({('hold_'+k):v for k,v in hs.items()});holds.append(r.copy())
 z=dict(dataset=d,sequence_id=s,anchor_idx=i,target_idx=t,**{('commit_'+k):v for k,v in bs.items()},parent_alignment_pass=ok);branches.append(z)
 r.update(z);r.update(U_AbsRel=hs['AbsRel']-bs['AbsRel'],U_RMSE=hs['RMSE']-bs['RMSE'],U_Delta1=bs['delta1']-hs['delta1'],included=ok);labels.append(r)
save('P5NORM_V2_4_HOLD_BASELINE.csv',holds);save('P5NORM_V2_4_SINGLE_COMMIT_RESULTS.csv',branches);save('P5NORM_V2_4_PARENT_ALIGNMENT.csv',alignment);save('P5NORM_V2_4_UTILITY_LABELS.csv',labels)
features=['overall_update_l2','overall_relative_norm','update_l1','update_l2','update_l3','update_l4'];seq=[]
for s in seqs:
 a=[x for x in labels if x['dataset']==s['dataset'] and x['sequence_id']==s['sequence_id'] and x['included']]
 if not a:continue
 z=dict(dataset=s['dataset'],sequence_id=s['sequence_id'],anchors=len(a),mean_U_AbsRel=statistics.mean(x['U_AbsRel'] for x in a),beneficial=sum(x['U_AbsRel']>0 for x in a),harmful=sum(x['U_AbsRel']<0 for x in a))
 for f in features:z['mean_'+f]=statistics.mean(x[f] for x in a if x[f] is not None)
 seq.append(z)
save('P5NORM_V2_4_SEQUENCE_ANALYSIS.csv',seq)
def corr(x,y,kind='spearman'):
 if len(x)<3 or len(set(x))<2 or len(set(y))<2:return None
 return float((spearmanr if kind=='spearman' else pearsonr)(x,y).statistic)
domains=[]
for d in ['Bonn','TUM','ScanNet']:
 ss=[x for x in seq if x['dataset']==d];aa=[x for x in labels if x['dataset']==d and x['included']];domains.append(dict(dataset=d,sequences=len(ss),anchors=len(aa),beneficial=sum(x['U_AbsRel']>0 for x in aa),harmful=sum(x['U_AbsRel']<0 for x in aa),mean_U_AbsRel=statistics.mean(x['mean_U_AbsRel'] for x in ss) if ss else None,spearman=corr([x['mean_overall_update_l2'] for x in ss],[x['mean_U_AbsRel'] for x in ss])))
save('P5NORM_V2_4_DOMAIN_ANALYSIS.csv',domains)
x=np.array([z['mean_overall_update_l2'] for z in seq]);y=np.array([z['mean_U_AbsRel'] for z in seq]);rho=corr(x,y);pear=corr(x,y,'pearson');rng=np.random.default_rng(20261025);groups=[np.array([i for i,z in enumerate(seq) if z['dataset']==d],dtype=int) for d in ['Bonn','TUM','ScanNet']];boot=[]
for j in range(10000):
 idx=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups if len(g)]);v=corr(x[idx],y[idx]);
 if v is not None and math.isfinite(v):boot.append(v)
ci=np.quantile(boot,[.025,.975]).tolist() if boot else [None,None];xc=x.copy();yc=y.copy()
for g in groups:
 if len(g):xc[g]-=x[g].mean();yc[g]-=y[g].mean()
center=corr(xc,yc);loo={d:corr([z['mean_overall_update_l2'] for z in seq if z['dataset']!=d],[z['mean_U_AbsRel'] for z in seq if z['dataset']!=d]) for d in ['Bonn','TUM','ScanNet']};aux={f:corr([z['mean_'+f] for z in seq],y) for f in features};good=sum(z['included'] for z in labels);positive=sum(z['included'] and z['U_AbsRel']>0 for z in labels);negative=sum(z['included'] and z['U_AbsRel']<0 for z in labels)
txt=f'''# P5-Norm 2.4 独立序列复现\n\n序列留出 {len(seqs)}，实际有效序列 {len(seq)}；Hold {len(seqs)}，单次 Commit {len(labels)}；关键可观测张量通过 {good}/{len(labels)}；正/负效用 {positive}/{negative}。独立单位为序列，各序列的合格 anchor 先取平均。所有候选曾在 P3-B 历史实验中出现，属于项目历史已暴露、P5-Norm 2.0–2.3 开发留出，不能称为项目未见。\n\n主要指标 U=Hold AbsRel − single-commit AbsRel：Spearman {rho}；Pearson {pear}；固定种子 20261025 的 10000 次按域分层序列 Bootstrap，95% 区间 {ci}，有效重采样 {len(boot)}。域内中心化 Spearman {center}；逐域剔除相关 {loo}；辅助特征探索 {aux}。\n\n各域：{domains}。GT 离线计算并使用相同 mask。所有关键张量按 atol=rtol=1e-5 逐元素比较，比较通过不证明全部原生隐藏状态一致；P3-B 历史暴露和小样本均限制外推。相关不等同于决策效用的跨域预测能力或完整因果证明。无阈值搜索、M1 训练或 B 策略。\n'''
(root/'P5NORM_V2_4_NORM_UTILITY_ANALYSIS.md').write_text(txt,encoding='utf-8')
if rho is None:decision='有效样本不足，无法判断复现。'
elif rho<0:decision='主要假设方向受到反证；不应按原方向训练 M1。'
elif ci[0] is not None and ci[0]>0 and all(v is not None and v>0 for v in loo.values()) and center is not None and center>0:decision='正向关系复现，且本次统计检查未显示由单一域完全驱动；仅支持继续研究，不代表泛化或部署验收。'
elif rho<0.1:decision='相关接近零，未明确复现。'
else:decision='方向为正但统计或跨域证据不足；保留候选特征，暂不训练 M1。'
(root/'P5NORM_V2_4_FINAL_DECISION.md').write_text('# 研究决策\n\n'+decision+'\n\n'+txt,encoding='utf-8');print(txt);print('DECISION',decision)
