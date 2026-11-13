"""Adapt recovered smoke input/output to frozen HDF5 anchors; retain C3 literally."""
import hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;original=(p/'original_smoke_capture.py').read_text(encoding='utf8');s=original
def repl(old,new):
 global s
 if s.count(old)!=1:raise ValueError('source fragment count != 1: '+old[:80])
 s=s.replace(old,new)
repl('import base64,cv2,hashlib,io,json,math,sys,zlib','import base64,cv2,hashlib,io,json,math,sys,zlib,tempfile,h5py\nfrom PIL import Image')
start=s.index("cp=ROOT/'outputs/");end=s.index("if not torch.cuda.is_available():",start)
header="""domain,aid,meta_hash,npz_hash,outname=sys.argv[1:];outdir=Path(outname);outdir.mkdir(parents=True,exist_ok=False)
if (domain,aid) not in [('TUM','TUM_00_age4'),('ScanNet','ScanNet_00_age4')]:raise RuntimeError('anchor not frozen')
cp=ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt';snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';run=ROOT/'outputs/m05x_p3c1_formal_20261110_084314'/aid;meta_path=run/'capture_metadata.json';npz_path=run/'evidence_arrays.npz'
expected={'checkpoint':('2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251',cp),'backbone':('615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e',snap/'model.safetensors'),'decoder':('35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08',ROOT/'velodepth/models/decoder_velo.py'),'repair_head':('6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c',ROOT/'velodepth/models/persistent_feature_repair.py'),'exp1':('c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f',ROOT/'experiments/p1/gt4a2_exp1_train_eval.py'),'gt2':('bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192',ROOT/'experiments/p1/gt2_fixed_clip_repair_train.py'),'m05':('3e872b9cdd37f6368b73308e5fd8212cdc3964221da44537cc322d24c49e9e05',ROOT/'experiments/p1/cvpr_m05_formal.py'),'model_config':('09b87d5ac7180318803d31f421c96bd565a4a356860419fc8778f4ba3a72cf8f',snap/'config.json')}
identity={k:digest(p) for k,(_,p) in expected.items()}
if any(identity[k]!=v[0] for k,v in expected.items()):raise RuntimeError('frozen identity mismatch')
meta=json.loads(meta_path.read_text())
if digest(meta_path)!=meta_hash or digest(npz_path)!=npz_hash or digest(npz_path)!=meta['evidence_npz_sha256']:raise RuntimeError('historical evidence digest mismatch')
if meta['anchor_id']!=aid or meta['dataset']!=domain or meta['age']!=4:raise RuntimeError('wrong anchor')
with np.load(npz_path,allow_pickle=False) as z:
 def fetch(k):return z[meta.get('array_aliases',{}).get(k,k)].copy()
 parent={k:torch.from_numpy(fetch('parent_O__'+k)).cuda() for k in ('radius','confidence','rays_map')};parent['features']=[torch.from_numpy(fetch('parent_O__feature'+str(i))).cuda() for i in range(4)]
 rng={'cpu':torch.from_numpy(fetch('replay_rng_O_cpu')),'cuda':[torch.from_numpy(fetch('replay_rng_O_cuda_0'))]}
 previous_d1=fetch('current_D1_O');previous_input=fetch('input_O_anchor');previous_s0=[fetch('committed_O__feature'+str(i)) for i in range(4)];previous_s1=[fetch('candidate_V5__feature'+str(i)) for i in range(4)]
"""
s=s[:start]+header+s[end:]
start=s.index('rows=exp1.read_manifest(');end=s.index("if not np.array_equal(array(frames[4]['img'])",start)
data="""hpath=Path.home()/'datasets/velodepth_evaluation'/(domain+'.hdf5');metakey='sequences.json' if domain=='TUM' else 'sequences_.json'
with h5py.File(hpath,'r') as h:
 rawmeta=bytes(h[metakey][:])
 if hashlib.sha256(rawmeta).hexdigest()!=meta['identity']['sequence_metadata']:raise RuntimeError('sequence metadata mismatch')
 records=json.loads(rawmeta.decode())[meta['sequence']];infos=meta['inputs'];rgb={}
 for i in (0,3,4):
  key=records['image'][i];blob=h[key][:].tobytes()
  if key!=infos[i]['rgb_key'] or hashlib.sha256(blob).hexdigest()!=infos[i]['rgb_sha256']:raise RuntimeError('RGB identity mismatch '+str(i))
  rgb[i]=np.asarray(Image.open(io.BytesIO(blob)).convert('RGB'))
with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
 Image.fromarray(rgb[0]).save(tmp.name);shape=exp1.shape_info_for(model,Path(tmp.name),device)
model._m05_shape=shape;frames={}
for i in (3,4):
 if rgb[i].shape[:2]!=tuple(shape['original_shape']):raise RuntimeError('RGB shape mismatch')
 img,u8=gt2.preprocess_rgb(rgb[i],device,shape['paddings'],shape['network_shape'],shape['mean'],shape['std']);frames[i]={'index':i,'img':img,'frame_u8':u8}
model.pixel_decoder.set_shapes({'image':frames[4]['img'],'features':[parent['features'][0].permute(0,2,3,1)]},[{'T':1,'B':1}])
"""
s=s[:start]+data+s[end:]
repl("def enc_hook(module,inputs,out):capture['encoder'].append(out[1].detach().clone())", "def enc_hook(module,inputs,out):\n capture['encoder'].append(out[1].detach().clone());capture['encoder_outputs']={'features_residual':[x.detach().clone() for x in out[0]],'refined_flow':out[1].detach().clone(),'flow_features':out[2].detach().clone()}")
repl("'anchor':'Bonn_00_age4'", "'anchor':aid,'dataset':domain,'sequence':meta['sequence']")
repl(" and r['c3'] is not None for r in layers)"," and r['c3'] is not None and r['s0_equals_historical'] and r['candidate_equals_historical'] and r['raw_grid_nonfinite']==0 and r['actual_grid_nonfinite']==0 and r['delta_nonfinite']==0 and r['warp_nonfinite']==0 for r in layers) and result['historical_d1_equal'] and all(v['exact'] for v in equal.values())")
tail="""
# Save all demanded raw capture tensors; no future RGB/GT is loaded.
evidence={}
def save(k,t):evidence[k]=t.detach().cpu().contiguous().numpy().copy() if torch.is_tensor(t) else np.array(t,copy=True)
for i in range(4):
 g=capture['grids'][i];sample=grid_samples[i];m=torch.isfinite(g['raw']).all(dim=-1)&(g['raw'][...,0]>=-1)&(g['raw'][...,0]<=1)&(g['raw'][...,1]>=-1)&(g['raw'][...,1]<=1)
 for k,v in {'raw_grid':g['raw'],'actual_grid':g['actual'],'mask':m,'level_flow':g['flow'],'warp':capture['fusion'][i],'sample_input':sample['input'],'sample_grid':sample['grid'],'sample_output':sample['output'],'S0':source[i],'candidate':candidate[i],'delta':candidate[i].float()-source[i].float(),'base_feature':base['features'][i],'diag_feature':diagnostic['features'][i]}.items():save(k+'_'+str(i),v)
for k,v in capture['encoder_outputs'].items():
 if isinstance(v,list):
  for i,x in enumerate(v):save('small_encoder_'+k+'_'+str(i),x)
 else:save('small_encoder_'+k,v)
for k in ('radius','confidence','rays_map'):save('base_'+k,base[k]);save('diag_'+k,diagnostic[k])
save('base_D1',base_d1);save('diag_D1',diag_d1);save('historical_D1',previous_d1)
np.savez_compressed(outdir/'capture.npz',**evidence)
result.pop('encoded_grids');result['capture_npz_sha256']=digest(outdir/'capture.npz');result['arrays']={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k,v in evidence.items()}
(outdir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\\n',encoding='utf8')
print('CAPTURE_STATUS',result['pass'],outdir,flush=True)
raise SystemExit(0 if result['pass'] else 2)
"""
repl("print('RESULT_JSON:'+json.dumps(result,allow_nan=False),flush=True)",tail)
start=" g=capture['grids'][i];actual=grid_samples[i];w=capture['fusion'][i];delta=";stop=" row={'level':i,"
orig_block=original[original.index(start):original.index(stop,original.index(start))];new_block=s[s.index(start):s.index(stop,s.index(start))]
if orig_block!=new_block:raise RuntimeError('smoke C3 block changed')
(p/'cross_domain_capture.py').write_text(s,encoding='utf8');print('CAPTURE_PREPARED','original_c3_block_sha256',hashlib.sha256(orig_block.encode()).hexdigest(),'cross_script_sha256',hashlib.sha256(s.encode()).hexdigest())

