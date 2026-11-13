"""Verify P/O matched training via actual cache bytes; no claim on historical raw files."""
import hashlib,json,random
from pathlib import Path
from contract import digest,read_json,require
SEQS=('crowd2','crowd3','balloon2')
def order(seed):
 r=random.Random(int(seed));a=[]
 while len(a)<1000:
  block=list(range(24));r.shuffle(block);a.extend(block)
 return a[:1000]
def order_sha(seq):return hashlib.sha256(','.join(map(str,seq)).encode()).hexdigest()
def stable_digest(p):
 p=Path(p)
 with p.open('rb') as f:
  a=__import__('os').fstat(f.fileno());h=hashlib.sha256()
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  z=__import__('os').fstat(f.fileno())
  require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns),'Cache changed during hashing: '+str(p))
  return h.hexdigest(),a.st_size
def verify_cache_pairs(c,pinned_sources=None,expected_base=None):
 ps={str(x['seed']):x for x in c['checkpoints']['P']};os_={str(x['seed']):x for x in c['checkpoints']['O_multi']};require(set(ps)==set(os_) and len(ps)==3,'P/O seeds mismatch')
 seen={};evidence=[]
 for seed in sorted(ps):
  p,o=ps[seed],os_[seed];a,b=read_json(p['training_record']),read_json(o['training_record']);schedule=order(seed);osh=order_sha(schedule)
  require(a.get('checkpoint_sha256')==p['sha256'] and b.get('checkpoint_sha256')==o['sha256'],'Training summary checkpoint hash mismatch')
  require(a.get('steps')==1000 and b.get('steps')==1000 and b.get('optimizer_updates')==1000,'Training updates mismatch')
  require(a.get('initial_head_state_sha256') and a.get('initial_head_state_sha256')==b.get('initial_adapter_state_sha256'),'P/O initial heads differ')
  require(a.get('sample_order_sha256')==b.get('sample_order_sha256')==osh,'P/O sample order differs')
  require(a.get('training_pair_csv_sha256_by_sequence')==b.get('training_pair_csv_sha256_by_sequence'),'P/O pairing manifests differ')
  require(a.get('training_rgb_indices_by_sequence')==b.get('training_rgb_indices_by_sequence')=={s:[0,255] for s in SEQS},'P/O training prefix mismatch')
  pbase=Path(p['path']).resolve().parents[2];obase=Path(o['path']).resolve().parents[2];counts=[];cycles=[];manifests=[]
  for seq in SEQS:
   pdir=pbase/'training/scene_caches'/seq/'cycle_cache';odir=obase/'training/scene_caches'/seq/'cycle_cache';pm=read_json(pdir/'cache_manifest.json');om=read_json(odir/'cache_manifest.json')
   manifests.append({'sequence':seq,'P_manifest':str(pdir/'cache_manifest.json'),'P_manifest_sha256':digest(pdir/'cache_manifest.json'),'O_manifest':str(odir/'cache_manifest.json'),'O_manifest_sha256':digest(odir/'cache_manifest.json')})
   require(all(pm.get(k)==om.get(k) and pm.get(k) for k in ('format','pretrained_weights_sha256','pair_sha256','original_source_hashes')),'Cache identity differs')
   if expected_base is not None:require(pm.get('pretrained_weights_sha256')==expected_base,'Cache model differs from frozen snapshot')
   if pinned_sources is not None:require(all(pinned_sources.get(k)==v for k,v in pm['original_source_hashes'].items()) and set(pm['original_source_hashes'])=={'velodepth/models/decoder_velo.py','velodepth/models/persistent_feature_repair.py','experiments/p1/gt2_fixed_clip_repair_train.py','experiments/p1/gt4a2_exp1_train_eval.py'},'Cache source identity differs')
   require(pm['pair_sha256']==a['training_pair_csv_sha256_by_sequence'][seq],'Cache/pair manifest identity differs')
   for i in range(8):
    pe=pm.get('cycles',{}).get(str(i));oe=om.get('cycles',{}).get(str(i));require(isinstance(pe,dict) and isinstance(oe,dict),'Cache cycle manifest missing');require(pe.get('sha256')==oe.get('sha256') and pe.get('bytes')==oe.get('bytes') and pe.get('supervised')==oe.get('supervised'),'Cache cycle metadata differs')
    files=[]
    for directory,entry in ((pdir,pe),(odir,oe)):
     f=(directory/f'cycle_{i:02d}.pt').resolve();st=f.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
     if key not in seen:seen[key]=stable_digest(f)
     h,size=seen[key];require(h==entry['sha256'] and size==entry['bytes'],'Cache bytes differ from recorded manifest: '+str(f));files.append(str(f))
    count=pe.get('supervised');require(type(count) is int and 0<count<=31,'Invalid supervised frames')
    counts.append(count);cycles.append({'sequence':seq,'cycle':i,'cache_sha256':pe['sha256'],'supervised':count,'p_file':files[0],'o_file':files[1],'shared_path':files[0]==files[1]})
    print('[M01 CACHE]',seed,seq,i+1,'/8',flush=True)
  exposure=sum(counts[i] for i in schedule);require(exposure==b.get('supervised_propagation_frame_exposures'),'GT exposure differs from cache/order')
  evidence.append({'seed':seed,'P_checkpoint_sha256':p['sha256'],'O_multi_checkpoint_sha256':o['sha256'],'initial_head_sha256':a['initial_head_state_sha256'],'sample_order_sha256':osh,'steps':1000,'derived_gt_exposures':exposure,'cache_manifests':manifests,'cycles':cycles,'historical_original_gt_bytes':'unknown; not inferred from cache equality','historical_shared_path':pbase==obase})
 return {'status':'CACHE_TENSOR_EQUIVALENCE_VERIFIED','unique_cache_files_hashed':len(seen),'pairs':evidence,'limitations':['Cached training data matched; original historical raw GT image byte identity is not established.','20261025 P/O share the same historical cache path.']}
