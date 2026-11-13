"""Offline evaluation of pre-frozen online anchor+4 predictions."""
from pathlib import Path
import csv,json,io,collections
import numpy as np,cv2,h5py
from PIL import Image
root=Path('/home/svrdepth/SVR-Depth/velodepth/outputs/p5norm_v2_ac_pilot');anchors=list(csv.DictReader((root/'sequence_selection/pilot_anchor4_manifest.csv').open()));records=[]
for a in anchors:
 dom,seq,idx=a['dataset'],a['sequence_id'],int(a['target_idx']);assert idx==int(a['anchor_idx'])+4;folder=root/dom/Path(seq).name
 if dom=='Bonn':raw=cv2.imread(a['gt_path'],cv2.IMREAD_UNCHANGED);gt=raw.astype(np.float32)/5000.;mask=np.isfinite(gt)&(gt>0)
 else:
  with h5py.File(Path.home()/'datasets/velodepth_evaluation'/(dom+'.hdf5'),'r') as h:raw=np.array(Image.open(io.BytesIO(h[a['gt_path']][:].tobytes())))
  if raw.ndim==3:raw=raw[:,:,0].astype(np.int32)|(raw[:,:,1].astype(np.int32)<<8)|(raw[:,:,2].astype(np.int32)<<16)
  gt=raw.astype(np.float32)/1000.;mask=np.isfinite(gt)&(gt>(.001 if dom=='TUM' else .01))&(gt<10.)
 if not mask.any():raise RuntimeError('no GT valid '+str(a))
 preds={s:np.load(folder/s/'predictions'/('frame_%06d.npy'%idx)) for s in ['A','C']}
 for s,p in preds.items():
  p=np.squeeze(p)
  if p.shape!=gt.shape or not (np.isfinite(p[mask])&(p[mask]>0)).all():raise RuntimeError('invalid prediction '+str((a,s,p.shape,gt.shape)))
  g=gt[mask].astype(np.float64);v=p[mask].astype(np.float64);records.append(dict(dataset=dom,sequence_id=seq,anchor_id=a['anchor_id'],anchor_idx=a['anchor_idx'],target_idx=idx,strategy=s,valid_pixels=int(mask.sum()),AbsRel=float(np.mean(abs(g-v)/g)),RMSE=float(np.sqrt(np.mean((g-v)**2))),delta1=float(np.mean(np.maximum(g/v,v/g)<1.25))))
out=root/'results';out.mkdir(exist_ok=False)
def write(n,rows):
 with (out/n).open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
write('P5NORM_V2_ONLINE_ANCHOR4_RESULTS.csv',records)
groups=collections.defaultdict(list)
for x in records:groups[(x['dataset'],x['sequence_id'],x['strategy'])].append(x)
seq=[]
for d,s in sorted({(x['dataset'],x['sequence_id']) for x in records}):
 a=groups[(d,s,'A')];c=groups[(d,s,'C')];ma=float(np.mean([x['AbsRel'] for x in a]));mc=float(np.mean([x['AbsRel'] for x in c]));seq.append(dict(dataset=d,sequence_id=s,anchors=len(a),A_AbsRel=ma,C_AbsRel=mc,A_RMSE=float(np.mean([x['RMSE'] for x in a])),C_RMSE=float(np.mean([x['RMSE'] for x in c])),A_delta1=float(np.mean([x['delta1'] for x in a])),C_delta1=float(np.mean([x['delta1'] for x in c])),delta=mc-ma,relative_pct=100*(ma-mc)/ma))
write('P5NORM_V2_AC_PAIRED_DELTAS.csv',seq)
domains=[]
for d in ['Bonn','TUM','ScanNet']:
 z=[x for x in seq if x['dataset']==d]
 if z:domains.append(dict(dataset=d,sequences=len(z),A_AbsRel=float(np.mean([x['A_AbsRel'] for x in z])),C_AbsRel=float(np.mean([x['C_AbsRel'] for x in z])),A_RMSE=float(np.mean([x['A_RMSE'] for x in z])),C_RMSE=float(np.mean([x['C_RMSE'] for x in z])),A_delta1=float(np.mean([x['A_delta1'] for x in z])),C_delta1=float(np.mean([x['C_delta1'] for x in z])),delta=float(np.mean([x['delta'] for x in z])),C_better=sum(x['delta']<0 for x in z),A_better=sum(x['delta']>0 for x in z),exact_ties=sum(x['delta']==0 for x in z)))
write('P5NORM_V2_DOMAIN_COMPARISON.csv',domains)
rng=np.random.default_rng(20261025);samples=[]
for _ in range(10000):
 ds=[]
 for d in ['Bonn','TUM','ScanNet']:
  z=[x['delta'] for x in seq if x['dataset']==d];ds.append(float(np.mean(rng.choice(z,size=len(z),replace=True))))
 samples.append(np.mean(ds))
(out/'P5NORM_V2_AC_PILOT_ANALYSIS.md').write_text('# A/C 在线 anchor+4 Pilot\n\nDelta=C−A；每序列先平均 anchor，再对域等权，域间宏平均。配对 Bootstrap 在每个域内重采样序列，10000 次、seed 20261025；仅 3 条/域，探索性。\n\n'+str(domains)+'\n\n宏均值 '+str(float(np.mean([x['delta'] for x in domains])))+'；Bootstrap 95% '+str(np.quantile(samples,[.025,.975]).tolist())+'\n\n此指标不是历史 Repair OFF 的单次反事实 H4；不直接判断 Norm 的预测能力。\n')
(out/'P5NORM_V2_HISTORICAL_H4_SEPARATE_REPORT.md').write_text('# 历史 H4 与本轮分离\n\n历史 H4 在单次干预后 Repair OFF 回放至 t+4；本轮采用每帧策略持续生效的在线轨迹 t+4，两者未混合。\n')
print('EVALUATED',len(records),'anchor-strategy records',flush=True)
