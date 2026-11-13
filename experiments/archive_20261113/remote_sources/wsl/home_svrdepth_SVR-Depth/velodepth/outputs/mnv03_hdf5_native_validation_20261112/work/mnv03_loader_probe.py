import os,sys,json,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path('/home/svrdepth/SVR-Depth/velodepth');sys.path.insert(0,str(root))
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
import torch
import numpy as np
from velodepth.datasets.tum import TUM
from velodepth.datasets.scannet import ScanNetVid
cfg=json.loads((root/'configs/velodepth-eval.json').read_text())['data']
for cls in [TUM,ScanNetVid]:
 try:
  original_array=np.array
  class LegacyArray(np.ndarray):
   def tostring(self, *a, **k): return self.tobytes(*a, **k)
  def compatible_array(*a, **k): return original_array(*a, **k).view(LegacyArray)
  np.array=compatible_array
  ds=cls(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.0,num_frames=-1)
  np.array=original_array
  print('DATASET',cls.__name__,'FIRST',ds.dataset[0], 'KEYS',ds.sequences[0].keys(),flush=True)
  idxs,_=ds.get_test_idxs(len(ds.sequences[0]['image']),ds.sequences[0].get('keyframe_idx'))
  print('INDICES',idxs,flush=True)
  x=ds.get_single_sequence(0)
  print('RESULT_KEYS',list(x),flush=True)
  for k,v in x.items():
   print('OUT',str(k),type(v).__name__,getattr(v,'shape',None),str(v)[:150] if not hasattr(v,'shape') else '',flush=True)
 except Exception:
  np.array=original_array
  traceback.print_exc()
print('CUDA_INITIALIZED',torch.cuda.is_initialized())
