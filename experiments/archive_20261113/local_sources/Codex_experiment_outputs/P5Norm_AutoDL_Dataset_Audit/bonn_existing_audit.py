#!/usr/bin/env python3
import json,glob,bisect,time
from pathlib import Path
from PIL import Image
base=Path('/root/autodl-tmp/datasets');out=base/'p5norm_v5_missing_20261110'/'existing_sequences_audit.json'
paths=sorted([*base.glob('rgbd_bonn_*'),*(base/'bonn').glob('rgbd_bonn_*')]);res=[]
for p in paths:
 if not p.is_dir():continue
 row={'path':str(p),'errors':[]}
 for mode in ['rgb','depth']:
  files=sorted((p/mode).glob('*.png'));row[mode+'_count']=len(files);idx=p/(mode+'.txt');refs=[]
  if idx.exists():
   for line in idx.read_text().splitlines():
    if line.strip() and not line.startswith('#'):
     v=line.split()
     if len(v)>=2:refs.append(v[1])
  row[mode+'_index_entries']=len(refs);row[mode+'_missing_index_references']=[x for x in refs if not (p/x).is_file()]
  for f in files:
   try:
    with Image.open(f) as im:im.load()
   except Exception as e:row['errors'].append(str(f)+': '+str(e))
 rgb=sorted(float(f.stem) for f in (p/'rgb').glob('*.png'));depth=sorted(float(f.stem) for f in (p/'depth').glob('*.png'));deltas=[]
 for t in rgb:
  i=bisect.bisect_left(depth,t);deltas.append(min([abs(depth[j]-t) for j in (i-1,i) if 0<=j<len(depth)],default=float('inf')))
 row['nearest_depth_within_20ms']=sum(x<=0.020 for x in deltas);row['unmatched_rgb_nearest_20ms']=sum(x>0.020 for x in deltas);row['matching_method']='independent nearest timestamp; not frozen one-to-one manifest'
 res.append(row);out.write_text(json.dumps({'status':'RUNNING','sequences':res},indent=2));print('DONE',p.name,row['rgb_count'],row['depth_count'],'errors',len(row['errors']),flush=True)
out.write_text(json.dumps({'status':'COMPLETE','sequences':res},indent=2))
