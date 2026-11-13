#!/usr/bin/env python3
"""Read-only source identity check ONLY; does not run CUDA or inference."""
import argparse, hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
files={'experiments/p1/cvpr_m05_formal.py':'3e872b9cdd37f6368b73308e5fd8212cdc3964221da44537cc322d24c49e9e05','experiments/p1/gt4a2_exp1_train_eval.py':'c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f','experiments/p1/gt2_fixed_clip_repair_train.py':'bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192','velodepth/models/decoder_velo.py':'35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08','velodepth/models/persistent_feature_repair.py':'6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c','outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251'}
for name,want in files.items():
 h=hashlib.sha256()
 with (a.root/name).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 print(name,h.hexdigest(),flush=True)
 if h.hexdigest()!=want:raise SystemExit('STOP: source identity mismatch')
raise SystemExit('STOP: resource checks and two-frame diagnostic not implemented')
