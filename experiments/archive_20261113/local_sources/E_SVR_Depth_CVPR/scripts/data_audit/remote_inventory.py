"""Read-only dataset metadata and sparse RGB payload fingerprint inventory."""
import csv,hashlib,io,json,os,re
from collections import Counter
from pathlib import Path
import h5py
from PIL import Image
root=Path.home()/'SVR-Depth/velodepth'; result={'server_root':str(root),'datasets':{},'bonn':[],'errors':[]}
for ds in ('TUM','ScanNet'):
 p=Path.home()/'datasets/velodepth_evaluation'/(ds+'.hdf5')
 if not p.is_file():result['errors'].append(str(p)+' missing');continue
 try:
  with h5py.File(p,'r') as h:
   key='sequences.json' if ds=='TUM' else 'sequences_.json';raw=bytes(h[key][:]);records=json.loads(raw.decode());items=[]
   for name,r in sorted(records.items()):
    images=r.get('image',[]);depth=r.get('depth',[]);indices=sorted({0,len(images)//2,len(images)-1}) if images else [];samples=[];dims=[];missing=[]
    for i in indices:
     k=images[i]
     if k not in h:missing.append(k);continue
     blob=h[k][:].tobytes();img=Image.open(io.BytesIO(blob));dims.append(list(img.size));samples.append({'index':i,'key':k,'encoded_sha256':hashlib.sha256(blob).hexdigest(),'decoded_rgb_sha256':hashlib.sha256(img.convert('RGB').tobytes()).hexdigest()})
    exists=all(k in h for k in images+depth);same_stem=len(images)==len(depth) and all(Path(a).stem.removesuffix('_c')==Path(b).stem.removesuffix('_d') for a,b in zip(images,depth))
    items.append({'sequence':name,'image_count':len(images),'depth_count':len(depth),'all_keys_exist':exists,'pair_stems_equal':same_stem,'sampled_rgb':samples,'sampled_dimensions':dims,'missing_sample_keys':missing,'timestamp_status':'UNRESOLVED_METADATA_KEYS_NOT_TRUSTED_AS_TIMESTAMPS'})
   result['datasets'][ds]={'path':str(p),'bytes':p.stat().st_size,'sha256':'NOT_COMPUTED_LARGE_CONTAINER','sequence_metadata_sha256':hashlib.sha256(raw).hexdigest(),'items':items}
 except Exception as e:result['errors'].append(ds+': '+repr(e))
manifest=root/'outputs/y04_v4_manifest_candidate_20261108_190940'
for p in sorted(manifest.glob('*_pairs_v4_candidate.csv')):
 try:
  with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  samples=[]
  for i in sorted({0,len(rows)//2,len(rows)-1}) if rows else []:
   image=Path(rows[i].get('rgb_path',''))
   if image.is_file():
    b=image.read_bytes();im=Image.open(io.BytesIO(b));samples.append({'index':i,'path':str(image),'encoded_sha256':hashlib.sha256(b).hexdigest(),'decoded_rgb_sha256':hashlib.sha256(im.convert('RGB').tobytes()).hexdigest(),'dimensions':list(im.size)})
  try:
   rgb_times=[float(r['rgb_timestamp']) for r in rows];depth_times=[float(r['depth_timestamp']) for r in rows if r.get('depth_timestamp')];matched=[r for r in rows if r.get('gt_match_status')=='MATCHED'];deltas=[float(r['abs_delta_ms']) for r in matched if r.get('abs_delta_ms')];time_info={'rgb_monotonic_strict':all(a<b for a,b in zip(rgb_times,rgb_times[1:])),'rgb_duplicate_adjacent':sum(a==b for a,b in zip(rgb_times,rgb_times[1:])),'depth_monotonic_nondecreasing':all(a<=b for a,b in zip(depth_times,depth_times[1:])),'matched_count':len(matched),'max_recorded_delta_ms':max(deltas) if deltas else None,'rgb_time_min':min(rgb_times) if rgb_times else None,'rgb_time_max':max(rgb_times) if rgb_times else None}
  except (ValueError,KeyError) as e:time_info={'status':'UNKNOWN','reason':str(e)}
  result['bonn'].append({'sequence':p.name.removesuffix('_pairs_v4_candidate.csv'),'manifest':str(p),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'frames':len(rows),'samples':samples,'columns':list(rows[0]) if rows else [],'timestamp_status':'PAIR_CSV_TIMESTAMPS_ONLY','timestamp_audit':time_info,'original_video':'UNRESOLVED'})
 except Exception as e:result['errors'].append(str(p)+': '+repr(e))
print(json.dumps(result,ensure_ascii=False))
