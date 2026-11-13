import os,sys,json,csv
from pathlib import Path
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'; os.environ['CUDA_VISIBLE_DEVICES']=''; sys.path.insert(0,str(Path.cwd()))
import numpy as np,torch
from velodepth.datasets.bonn import Bonn
from velodepth.datasets import collate_fn
cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']; old=np.array
class Legacy(np.ndarray):
 def tostring(self,*a,**k): return self.tobytes(*a,**k)
def compat(*a,**k): return old(*a,**k).view(Legacy)
try:
 np.array=compat
 ds=Bonn(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,crop=cfg['crop'],normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1,add_of=True)
finally: np.array=old
i=next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/person_tracking2_val')); ds.get_test_idxs=lambda n,k:(list(range(78,94)),None)
a=collate_fn([ds.get_single_sequence(i)],is_batched=False); d=a['data']; rows=[]
for j in range(16):
 z=d['depth'][0,j]; m=d['depth_mask'][0,j].bool(); v=z[m]
 rows.append({'hdf_index':78+j,'image_shape':str(tuple(d['image'][0,j].shape)),'depth_shape':str(tuple(z.shape)),'official_valid_pixels':int(m.sum()),'official_valid_min_m':float(v.min()) if v.numel() else '', 'official_valid_max_m':float(v.max()) if v.numel() else '', 'dataset_min_depth':ds.min_depth,'dataset_max_depth':ds.max_depth,'dataset_scale':ds.depth_scale,'crop':str(cfg['crop'])})
p=Path('outputs/saoof_00_closeout_20261112/official_gt_mask_audit.csv')
with p.open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('count',len(rows),'first',rows[0]); print('metadata',a['img_metas'][0]); print('camera',d['camera'].params.shape)
