import os,sys,json,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from velodepth.datasets.bonn import Bonn
cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']
old=np.array
class LegacyArray(np.ndarray):
 def tostring(self,*a,**kw): return self.tobytes(*a,**kw)
def compat(*a,**kw): return old(*a,**kw).view(LegacyArray)
try:
 np.array=compat
 ds=Bonn(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1)
finally: np.array=old
print('count',len(ds.dataset))
for i,x in enumerate(ds.dataset):
 name=x['sequence_name']
 if any('/'+s+'_val' in name for s in ['moving_obstructing_box','person_tracking2','removing_nonobstructing_box','moving_nonobstructing_box','placing_nonobstructing_box']):
  print('sequence',i,name,'samples',x['num_samples']); idxs,_=ds.get_test_idxs(len(ds.sequences[i]['image']),ds.sequences[i].get('keyframe_idx')); print('loader_indices',idxs)
  try:
   a=ds.get_single_sequence(i)
   print('output',[(k,str(getattr(v,'shape',type(v).__name__))) for k,v in a.items()])
  except Exception: traceback.print_exc()
  break
