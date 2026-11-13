from pathlib import Path
import csv,json,io
import numpy as np,cv2,h5py
from PIL import Image
base=Path('/home/svrdepth/SVR-Depth/velodepth/outputs');pilot=base/'p5norm_v2_ac_pilot';root=base/'p5norm_v2_3_single_commit'
def evaluate(a,folder):
 d=a['dataset'];seq=a['sequence_id'];idx=int(a['target_idx']);pred=np.squeeze(np.load(folder/'predictions'/('frame_%06d.npy'%idx)))
 if d=='Bonn':raw=cv2.imread(a['gt_path'],cv2.IMREAD_UNCHANGED);gt=raw.astype(np.float32)/5000.;mask=np.isfinite(gt)&(gt>0)
 else:
  with h5py.File(Path.home()/'datasets/velodepth_evaluation'/(d+'.hdf5'),'r') as h:raw=np.array(Image.open(io.BytesIO(h[a['gt_path']][:].tobytes())))
  if raw.ndim==3:raw=raw[:,:,0].astype(np.int32)|(raw[:,:,1].astype(np.int32)<<8)|(raw[:,:,2].astype(np.int32)<<16)
  gt=raw.astype(np.float32)/1000.;mask=np.isfinite(gt)&(gt>(.001 if d=='TUM' else .01))&(gt<10.)
 if pred.shape!=gt.shape or not mask.any() or not (np.isfinite(pred[mask])&(pred[mask]>0)).all():raise RuntimeError('invalid prediction/GT')
 g=gt[mask].astype(np.float64);p=pred[mask].astype(np.float64);metric={'AbsRel':float(np.mean(abs(g-p)/g)),'RMSE':float(np.sqrt(np.mean((g-p)**2))),'delta1':float(np.mean(np.maximum(g/p,p/g)<1.25))}
 old=[x for x in csv.DictReader((pilot/'results/P5NORM_V2_ONLINE_ANCHOR4_RESULTS.csv').open()) if x['dataset']==d and x['sequence_id']==seq and x['anchor_idx']==a['anchor_idx'] and x['target_idx']==a['target_idx'] and x['strategy']=='A'];assert len(old)==1 and int(old[0]['valid_pixels'])==int(mask.sum())
 r=json.loads((folder/'norm_records.json').read_text())[0];status=json.loads((folder/'status.json').read_text());r.update({'hold_absrel':float(old[0]['AbsRel']),'single_commit_absrel':metric['AbsRel'],'utility_absrel':float(old[0]['AbsRel'])-metric['AbsRel'],'hold_rmse':float(old[0]['RMSE']),'single_commit_rmse':metric['RMSE'],'utility_rmse':float(old[0]['RMSE'])-metric['RMSE'],'hold_delta1':float(old[0]['delta1']),'single_commit_delta1':metric['delta1'],'utility_delta1':metric['delta1']-float(old[0]['delta1']),'commit_count':sum(z['choice']=='COMMIT' for z in status['choices']),'subsequent_hold_count':sum(z['choice']=='HOLD' and z['frame']>int(a['anchor_idx']) for z in status['choices']),'runtime_seconds':status['elapsed_s']})
 return r
