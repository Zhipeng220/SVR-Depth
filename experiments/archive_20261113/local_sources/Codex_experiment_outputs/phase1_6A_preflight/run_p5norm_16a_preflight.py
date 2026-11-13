#!/usr/bin/env python3
"""Read-only source identity and memory gate; no model or CUDA execution."""
import os,hashlib,pathlib
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
root=pathlib.Path('/home/svrdepth/SVR-Depth/velodepth')
files={'experiments/p1/cvpr_m05_formal.py':'3e872b9cdd37f6368b73308e5fd8212cdc3964221da44537cc322d24c49e9e05','experiments/p1/gt4a2_exp1_train_eval.py':'c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f','experiments/p1/gt2_fixed_clip_repair_train.py':'bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192','velodepth/models/decoder_velo.py':'35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08','velodepth/models/persistent_feature_repair.py':'6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c','outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251'}
for name,expected in files.items():
 h=hashlib.sha256()
 with (root/name).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 print(name,h.hexdigest(),flush=True)
 if h.hexdigest()!=expected: raise SystemExit('STOP: source identity mismatch')
m={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))}
limit=pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip(); current=int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text())
effective=m['MemAvailable'] if limit=='max' else min(m['MemAvailable'],max(0,int(limit)-current))
print('effective_available_bytes',effective,'cgroup_max',limit,'cgroup_current',current,'swap_free',m['SwapFree'])
if effective<16*1024**3: raise SystemExit('BLOCKED: below 16 GiB; no model created')
raise SystemExit('STOP: memory gate passed; formal M05 CPU compatibility remains unresolved; no model created')
