import os,json,h5py,numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
for domain in ['TUM','ScanNet']:
 p=Path('/home/svrdepth/datasets/velodepth_evaluation')/(domain+'.hdf5')
 with h5py.File(p,'r') as h:
  print('\nDOMAIN',domain,'TOP',list(h.keys())[:25], 'ATTR',dict(h.attrs))
  key='sequences_.json' if domain=='ScanNet' else 'sequences.json'
  seq=json.loads(np.asarray(h[key]).tobytes().decode('ascii'))
  for split in ['val.txt','test.txt']:
   if split in h: print('SPLIT',split,np.asarray(h[split]).tobytes().decode('ascii')[:600])
  sid=sorted(seq)[0]; x=seq[sid]; print('FIRST',sid,'FIELDS',list(x), 'FRAMECOUNT',len(x['image']))
  for k,v in x.items(): print('FIELD',k,'LEN',len(v) if isinstance(v,list) else '-', 'FIRST',repr(v[0] if isinstance(v,list) and v else v)[:400])
