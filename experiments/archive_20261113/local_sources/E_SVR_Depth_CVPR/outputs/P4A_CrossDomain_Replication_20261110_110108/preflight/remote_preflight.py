import hashlib,json,sys,io
from pathlib import Path
import h5py,numpy as np
from PIL import Image
ROOT=Path.home()/'SVR-Depth/velodepth';run=ROOT/'outputs/m05x_p3c1_formal_20261110_084314';snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
expected={'checkpoint':('2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251',ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt'),'backbone':('615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e',snap/'model.safetensors'),'decoder':('35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08',ROOT/'velodepth/models/decoder_velo.py'),'repair_head':('6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c',ROOT/'velodepth/models/persistent_feature_repair.py'),'m05':('3e872b9cdd37f6368b73308e5fd8212cdc3964221da44537cc322d24c49e9e05',ROOT/'experiments/p1/cvpr_m05_formal.py'),'exp1':('c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f',ROOT/'experiments/p1/gt4a2_exp1_train_eval.py'),'gt2':('bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192',ROOT/'experiments/p1/gt2_fixed_clip_repair_train.py'),'model_config':('09b87d5ac7180318803d31f421c96bd565a4a356860419fc8778f4ba3a72cf8f',snap/'config.json')}
result={'identity':{},'domains':{},'errors':[]}
for name,(h,p) in expected.items():
 try:
  found=sha(p);result['identity'][name]={'path':str(p),'expected':h,'observed':found,'match':found==h}
  if found!=h:result['errors'].append('identity mismatch '+name)
 except Exception as e:result['errors'].append('identity unavailable '+name+' '+repr(e))
for ds,aid,metakey in [('TUM','TUM_00_age4','sequences.json'),('ScanNet','ScanNet_00_age4','sequences_.json')]:
 row={'anchor_id':aid,'status':'BLOCKED','checks':{},'error':''};result['domains'][ds]=row
 try:
  p=run/aid;meta=json.loads((p/'capture_metadata.json').read_text());expected_hashes=json.loads(sys.argv[1])[ds];row['checks']['historical_metadata_sha']=sha(p/'capture_metadata.json')==expected_hashes['metadata'];row['checks']['historical_npz_sha']=sha(p/'evidence_arrays.npz')==expected_hashes['npz']==meta['evidence_npz_sha256'];row['checks']['metadata_identity']=all(meta['identity'][k]==v[0] for k,v in expected.items());row['checks']['anchor_fixed']=(meta['anchor_id']==aid and meta['age']==4 and meta['dataset']==ds)
  hpath=Path.home()/'datasets/velodepth_evaluation'/(ds+'.hdf5')
  with h5py.File(hpath,'r') as h:
   raw=bytes(h[metakey][:]);row['checks']['sequence_metadata_sha']=hashlib.sha256(raw).hexdigest()==meta['identity']['sequence_metadata'];records=json.loads(raw.decode())[meta['sequence']];row['sequence']=meta['sequence'];row['dataset_path']=str(hpath);row['frames']=[]
   for i in (0,3,4):
    k=records['image'][i];blob=h[k][:].tobytes();rgbsha=hashlib.sha256(blob).hexdigest();im=Image.open(io.BytesIO(blob));match=k==meta['inputs'][i]['rgb_key'] and rgbsha==meta['inputs'][i]['rgb_sha256'];row['frames'].append({'index':i,'key':k,'sha256':rgbsha,'size':list(im.size),'matches_history':match})
   row['checks']['rgb_input_hashes']=all(x['matches_history'] for x in row['frames']);row['checks']['rgb_gt_metadata_lengths']=len(records['image'])==len(records['depth']);row['checks']['rgb_gt_metadata_stems']=all(Path(records['image'][i]).stem.removesuffix('_c')==Path(records['depth'][i]).stem.removesuffix('_d') for i in (3,4))
  with np.load(p/'evidence_arrays.npz',allow_pickle=False) as z:
   aliases=meta.get('array_aliases',{});get=lambda k:z[aliases.get(k,k)];keys=['parent_O__radius','parent_O__confidence','parent_O__rays_map']+['parent_O__feature'+str(i) for i in range(4)]+['replay_rng_O_cpu','replay_rng_O_cuda_0','input_O_anchor','current_D1_O']+['committed_O__feature'+str(i) for i in range(4)]+['candidate_V5__feature'+str(i) for i in range(4)];row['checks']['saved_arrays_present']=all(aliases.get(k,k) in z.files for k in keys);row['checks']['parent_and_predictions_finite']=all(np.isfinite(get(k)).all() for k in keys if 'rng_' not in k);row['shapes']={k:list(get(k).shape) for k in keys}
  row['identity']={k:meta['identity'][k] for k in ['checkpoint','backbone','decoder','repair_head','sequence_metadata']};row['checks']['historical_runtime']=bool(meta.get('device') and meta.get('torch') and meta.get('cuda'));row['status']='PASS' if all(row['checks'].values()) else 'BLOCKED'
  if row['status']!='PASS':result['errors'].append(ds+' failed checks '+str([k for k,v in row['checks'].items() if not v]))
 except Exception as e:row['error']=repr(e);result['errors'].append(ds+' '+repr(e))
result['status']='PASS' if not result['errors'] and all(x['status']=='PASS' for x in result['domains'].values()) else 'BLOCKED'
print('PREFLIGHT_JSON:'+json.dumps(result,allow_nan=False),flush=True)
