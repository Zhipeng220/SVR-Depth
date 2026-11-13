import hashlib,json
from pathlib import Path
root=Path.cwd();w=root/'m05x_work/p4ab_parallel_20261110';p=root/'outputs/P4A_CrossDomain_Replication_20261110_110108/preflight';original=(p/'cross_domain_capture.py').read_text(encoding='utf8');s=original;lock=json.loads((w/'protocol_lock.json').read_text(encoding='utf8'));manifest=json.loads((root/'docs/m05x/p3c1_formal/anchor_manifest.json').read_text(encoding='utf8'))['formal_anchors'];allowed={(a['dataset'],a['anchor_id']):a['age'] for a in manifest}
def repl(old,new):
 global s
 if s.count(old)!=1:raise ValueError('fragment count '+str(s.count(old))+' '+old[:100])
 s=s.replace(old,new)
repl("if (domain,aid) not in [('TUM','TUM_00_age4'),('ScanNet','ScanNet_00_age4')]:raise RuntimeError('anchor not frozen')",f"allowed={allowed!r}\nif (domain,aid) not in allowed:raise RuntimeError('anchor not frozen')\nage=allowed[(domain,aid)]")
repl("meta['age']!=4","meta['age']!=age")
start=s.index("hpath=Path.home()/'datasets/velodepth_evaluation'");end=s.index("if not np.array_equal(array(frames[4]['img'])",start)
data="""if domain=='Bonn':
 pair=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940'/(meta['sequence']+'_pairs_v4_candidate.csv')
 if digest(pair)!=meta['identity']['pair_csv']:raise RuntimeError('Bonn pair manifest mismatch')
 records=exp1.read_manifest(pair,training=False);infos=meta['inputs']
 for i in sorted({0,age-1,age}):
  if str(records[i]['rgb_path'])!=infos[i]['rgb_path'] or digest(records[i]['rgb_path'])!=infos[i]['rgb_sha256']:raise RuntimeError('RGB identity mismatch '+str(i))
 shape=exp1.shape_info_for(model,Path(records[0]['rgb_path']),device);model._m05_shape=shape;frames={i:exp1.load_frame(records[i],shape,device,need_gt=False) for i in (age-1,age)}
else:
 hpath=Path.home()/'datasets/velodepth_evaluation'/(domain+'.hdf5');metakey='sequences.json' if domain=='TUM' else 'sequences_.json'
 with h5py.File(hpath,'r') as h:
  rawmeta=bytes(h[metakey][:])
  if hashlib.sha256(rawmeta).hexdigest()!=meta['identity']['sequence_metadata']:raise RuntimeError('sequence metadata mismatch')
  records=json.loads(rawmeta.decode())[meta['sequence']];infos=meta['inputs'];rgb={}
  for i in sorted({0,age-1,age}):
   key=records['image'][i];blob=h[key][:].tobytes()
   if key!=infos[i]['rgb_key'] or hashlib.sha256(blob).hexdigest()!=infos[i]['rgb_sha256']:raise RuntimeError('RGB identity mismatch '+str(i))
   rgb[i]=np.asarray(Image.open(io.BytesIO(blob)).convert('RGB'))
 with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
  Image.fromarray(rgb[0]).save(tmp.name);shape=exp1.shape_info_for(model,Path(tmp.name),device)
 model._m05_shape=shape;frames={}
 for i in (age-1,age):
  if rgb[i].shape[:2]!=tuple(shape['original_shape']):raise RuntimeError('RGB shape mismatch')
  img,u8=gt2.preprocess_rgb(rgb[i],device,shape['paddings'],shape['network_shape'],shape['mean'],shape['std']);frames[i]={'index':i,'img':img,'frame_u8':u8}
model.pixel_decoder.set_shapes({'image':frames[age]['img'],'features':[parent['features'][0].permute(0,2,3,1)]},[{'T':1,'B':1}])
"""
s=s[:start]+data+s[end:]
s=s.replace('frames[4]','frames[age]').replace('frames[3]','frames[age-1]')
# Retain C3 source byte-for-byte relative to recovered original smoke.
o=(p/'original_smoke_capture.py').read_text(encoding='utf8');start=" g=capture['grids'][i];actual=grid_samples[i];w=capture['fusion'][i];delta=";stop=" row={'level':i,"
a=o[o.index(start):o.index(stop,o.index(start))];b=s[s.index(start):s.index(stop,s.index(start))]
if a!=b:raise RuntimeError('C3 source block changed')
(w/'capture.py').write_text(s,encoding='utf8')
print('CAPTURE_PREPARED','anchors',len(allowed),'C3_SHA256',hashlib.sha256(a.encode()).hexdigest(),'script_sha256',hashlib.sha256(s.encode()).hexdigest())

