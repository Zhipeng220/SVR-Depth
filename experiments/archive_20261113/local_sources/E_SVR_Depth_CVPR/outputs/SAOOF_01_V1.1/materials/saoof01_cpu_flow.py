"""Measure official Loader's backward DIS generation separately, not e2e."""
import os,sys,json,time,statistics
from pathlib import Path
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'; os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from velodepth.datasets.bonn import Bonn
cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']; old=np.array
class Legacy(np.ndarray):
 def tostring(self,*a,**kw): return self.tobytes(*a,**kw)
def compat(*a,**kw): return old(*a,**kw).view(Legacy)
try:
 np.array=compat
 ds=Bonn(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,crop=cfg['crop'],normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1,add_of=True)
finally: np.array=old
ds.resizer.sample=False; ds.resizer.image_shape=[434,574]
names=['moving_obstructing_box','person_tracking2','removing_nonobstructing_box','moving_nonobstructing_box','placing_nonobstructing_box']
original=ds.generate_of; samples=[]
def measured(x):
 t=time.perf_counter(); output=original(x); samples.append((time.perf_counter()-t)*1000); return output
ds.generate_of=measured
results=[]
for name in names:
 ix=next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/'+name+'_val'))
 for j in range(10):
  v=1+(j%15); ds.get_test_idxs=lambda n,k,v=v:([v-1,v],None); ds.get_single_sequence(ix)
 samples.clear()
 for rep in range(3):
  for v in range(1,16):
   ds.get_test_idxs=lambda n,k,v=v:([v-1,v],None); ds.get_single_sequence(ix)
 assert len(samples)==45
 results.append({'sequence':name,'pair_count':len(samples),'warmup_pairs':10,'repeats':3,'mean_ms':statistics.mean(samples),'p50_ms':float(np.percentile(samples,50)),'p95_ms':float(np.percentile(samples,95)),'raw_ms':samples[:]})
 print(name,results[-1]['mean_ms'],flush=True)
p=Path('outputs/saoof_01_ref_probe_20261112/latency'); (p/'official_backward_cpu_flow.json').write_text(json.dumps({'results':results,'boundary':'CPU wall time inside official Bonn Loader generate_of for two-frame inputs; includes grayscale conversion and native backward DIS, excludes image decode/preprocessing; separately run from end-to-end benchmarks.'},indent=2))
