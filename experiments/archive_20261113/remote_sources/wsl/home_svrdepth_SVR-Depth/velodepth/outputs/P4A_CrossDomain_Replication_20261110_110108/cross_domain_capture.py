import base64,cv2,hashlib,io,json,math,sys,zlib,tempfile,h5py
from PIL import Image
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1,gt2_fixed_clip_repair_train as gt2,cvpr_m05_formal as m05
import velodepth.models.decoder_velo as decoder
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def array(x):return x.detach().float().cpu().numpy().copy()
def thash(x):return hashlib.sha256(np.ascontiguousarray(x.detach().cpu().numpy()).tobytes()).hexdigest()
def packed(x):
 b=io.BytesIO();np.save(b,np.asarray(x),allow_pickle=False);raw=b.getvalue();return {'format':'base64(zlib(npy))','sha256_npy':hashlib.sha256(raw).hexdigest(),'data':base64.b64encode(zlib.compress(raw,9)).decode()}
def metrics(a,b):
 a,b=a.detach().float(),b.detach().float();finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all());maxerr=float(torch.max(torch.abs(a-b)).item()) if finite and a.shape==b.shape else None
 return {'shape_equal':a.shape==b.shape,'finite':finite,'exact':bool(torch.equal(a,b)),'max_abs':maxerr,'pass_atol_1e_6_rtol_0':finite and a.shape==b.shape and bool(torch.allclose(a,b,atol=1e-6,rtol=0))}
domain,aid,meta_hash,npz_hash,outname=sys.argv[1:];outdir=Path(outname);outdir.mkdir(parents=True,exist_ok=False)
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
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
device=torch.device('cuda:0');model=gt2.load_model(snap,'pre_readout',device);model.pixel_decoder.persistent_feature_repair.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['repair_head_state_dict'],strict=True);model.eval();m05.set_mode(model,'pre_readout')
hpath=Path.home()/'datasets/velodepth_evaluation'/(domain+'.hdf5');metakey='sequences.json' if domain=='TUM' else 'sequences_.json'
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
if not np.array_equal(array(frames[4]['img']),previous_input):raise RuntimeError('current RGB tensor mismatch with saved anchor')
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
with torch.no_grad():static=exp1.transition_static(model,frames[3],frames[4],shape,dis)
def cstate():return m05.clone_state(parent)
def cstatic():return {k:v.detach().clone() if torch.is_tensor(v) else v for k,v in static.items()}
m05.restore_rng(torch,rng);base=m05.run_step(model,exp1,frames[3],frames[4],cstatic(),cstate(),torch);base_d1=m05.make_prediction(model,gt2,base,shape)
capture={'grids':[],'grid_sample':[],'fusion':{},'repair':[],'encoder':[]};pending={};hooks=[];original_flow=decoder.flow2grid_sample;original_sample=F.grid_sample
def wrapped_flow(flow,mask=True):
 flow_before=flow.detach().clone();raw=original_flow(flow.clone(),mask=False);actual=original_flow(flow,mask=mask)
 row={'flow':flow_before,'raw':raw.detach().clone(),'actual':actual.detach().clone(),'mask_argument':mask};capture['grids'].append(row);pending[id(actual)]=len(capture['grids'])-1
 return actual
def wrapped_sample(inp,grid,*args,**kwargs):
 out=original_sample(inp,grid,*args,**kwargs)
 if id(grid) in pending:
  i=pending.pop(id(grid));capture['grid_sample'].append({'index':i,'input':inp.detach().clone(),'grid':grid.detach().clone(),'output':out.detach().clone(),'args':repr(args),'kwargs':repr(kwargs)})
 return out
def enc_hook(module,inputs,out):
 capture['encoder'].append(out[1].detach().clone());capture['encoder_outputs']={'features_residual':[x.detach().clone() for x in out[0]],'refined_flow':out[1].detach().clone(),'flow_features':out[2].detach().clone()}
def head_hook(module,inputs,out):capture['repair'].append(([x.detach().clone() for x in inputs[0]],[x.detach().clone() for x in out]))
def fusion_hook(i):
 def pre(module,inputs):capture['fusion'][i]=inputs[0].detach().clone()
 return pre
try:
 decoder.flow2grid_sample=wrapped_flow;F.grid_sample=wrapped_sample
 hooks.append(model.small_encoder.register_forward_hook(enc_hook));hooks.append(model.pixel_decoder.persistent_feature_repair.register_forward_hook(head_hook))
 for i,module in enumerate(model.pixel_decoder.radial_module.fusion):hooks.append(module.register_forward_pre_hook(fusion_hook(i)))
 m05.restore_rng(torch,rng);diagnostic=m05.run_step(model,exp1,frames[3],frames[4],cstatic(),cstate(),torch)
finally:
 decoder.flow2grid_sample=original_flow;F.grid_sample=original_sample
 for h in hooks:h.remove()
restored=decoder.flow2grid_sample is original_flow and F.grid_sample is original_sample
diag_d1=m05.make_prediction(model,gt2,diagnostic,shape);base_t=torch.as_tensor(base_d1);diag_t=torch.as_tensor(diag_d1)
equal={'D1':metrics(base_t,diag_t)}
for k in ('radius','confidence','rays_map'):equal[k]=metrics(base[k],diagnostic[k])
for i in range(4):equal['feature'+str(i)]=metrics(base['features'][i],diagnostic['features'][i])
if len(capture['grids'])!=4 or len(capture['grid_sample'])!=4 or len(capture['fusion'])!=4 or len(capture['repair'])!=1 or len(capture['encoder'])!=1 or pending:raise RuntimeError('unexpected hook counts')
source,candidate=capture['repair'][0];layers=[];encoded={};grid_samples=sorted(capture['grid_sample'],key=lambda x:x['index']);flow_sequence=capture['encoder'][0]
for i in range(4):
 g=capture['grids'][i];actual=grid_samples[i];w=capture['fusion'][i];delta=candidate[i].float()-source[i].float();raw=g['raw'];mask=torch.isfinite(raw).all(dim=-1)&(raw[...,0]>=-1)&(raw[...,0]<=1)&(raw[...,1]>=-1)&(raw[...,1]<=1);expanded=mask.unsqueeze(1).expand_as(delta);finite=torch.isfinite(delta)&torch.isfinite(w.float());selected=expanded&finite
 flow_sequence=F.interpolate(flow_sequence,size=parent['features'][i].shape[-2:],mode='bilinear',align_corners=True);flow_match=bool(torch.equal(flow_sequence,g['flow']))
 reason=None;value=None
 if not bool(mask.any()):reason='EMPTY_GEOMETRIC_MASK'
 elif bool((expanded&~finite).any()):reason='NONFINITE_IN_GEOMETRIC_MASK'
 else:
  x=delta[selected].double();y=w.float()[selected].double();nx=torch.linalg.vector_norm(x);ny=torch.linalg.vector_norm(y)
  if nx.item()==0 or ny.item()==0:reason='ZERO_NORM'
  else:value=float(torch.dot(x,y).item()/(nx.item()*ny.item()))
 row={'level':i,'delta_shape':list(delta.shape),'warp_shape':list(w.shape),'flow_shape':list(g['flow'].shape),'grid_shape':list(raw.shape),'grid_dtype':str(raw.dtype),'s0_sha256':thash(source[i]),'delta_sha256':thash(delta),'warp_sha256':thash(w),'level_flow_sha256':thash(g['flow']),'flow_interpolation_sequence_exact':flow_match,'mask_valid':int(mask.sum().item()),'mask_total':mask.numel(),'mask_fraction':float(mask.float().mean().item()),'raw_grid_nonfinite':int((~torch.isfinite(raw)).sum().item()),'actual_grid_nonfinite':int((~torch.isfinite(g['actual'])).sum().item()),'delta_nonfinite':int((~torch.isfinite(delta)).sum().item()),'warp_nonfinite':int((~torch.isfinite(w)).sum().item()),'c3':value,'undefined_reason':reason,'grid_actual_equals_sample_input':bool(torch.equal(g['actual'],actual['grid'])),'warp_equals_fusion_input':bool(torch.equal(actual['output'],w)),'grid_sample_input_equals_parent':bool(torch.equal(actual['input'],parent['features'][i])),'grid_sample_args':actual['args'],'grid_sample_kwargs':actual['kwargs'],'flow2grid_mask_argument':g['mask_argument'],'s0_equals_historical':bool(np.array_equal(array(source[i]),previous_s0[i])),'candidate_equals_historical':bool(np.array_equal(array(candidate[i]),previous_s1[i]))}
 layers.append(row);encoded[str(i)]={'raw_grid':packed(g['raw'].detach().cpu().numpy()),'actual_grid':packed(g['actual'].detach().cpu().numpy())}
result={'anchor':aid,'dataset':domain,'sequence':meta['sequence'],'identity':identity,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'hook_counts':{'small_encoder':len(capture['encoder']),'repair':len(capture['repair']),'grid':len(capture['grids']),'grid_sample':len(capture['grid_sample']),'fusion':len(capture['fusion'])},'refined_flow_shape':list(capture['encoder'][0].shape),'refined_flow_sha256':thash(capture['encoder'][0]),'refined_flow_nonfinite':int((~torch.isfinite(capture['encoder'][0])).sum().item()),'hooks_restored':restored,'baseline_vs_diagnostic':equal,'historical_d1_equal':bool(np.array_equal(base_d1,previous_d1)),'layers':layers,'encoded_grids':encoded}
result['pass']=restored and all(v['pass_atol_1e_6_rtol_0'] for v in equal.values()) and result['refined_flow_nonfinite']==0 and all(r['flow_interpolation_sequence_exact'] and r['grid_actual_equals_sample_input'] and r['warp_equals_fusion_input'] and r['grid_sample_input_equals_parent'] and r['flow2grid_mask_argument'] is True and r['c3'] is not None and r['s0_equals_historical'] and r['candidate_equals_historical'] and r['raw_grid_nonfinite']==0 and r['actual_grid_nonfinite']==0 and r['delta_nonfinite']==0 and r['warp_nonfinite']==0 for r in layers) and result['historical_d1_equal'] and all(v['exact'] for v in equal.values())

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
(outdir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
print('CAPTURE_STATUS',result['pass'],outdir,flush=True)
raise SystemExit(0 if result['pass'] else 2)


