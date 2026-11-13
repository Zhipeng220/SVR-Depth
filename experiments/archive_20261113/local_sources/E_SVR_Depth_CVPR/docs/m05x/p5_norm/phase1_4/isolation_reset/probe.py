"""Short, isolated runtime probe; does not alter project code or checkpoint."""
import sys,os,json,hashlib,math,gc,traceback
from pathlib import Path
import numpy as np, torch, cv2
root=Path.cwd();out=Path(sys.argv[1]);sys.path.insert(0,str(root/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1,gt2_fixed_clip_repair_train as gt2,cvpr_m05_formal as m05
cp=root/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt';snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';manifest=root/'outputs/y04_v4_manifest_candidate_20261108_190940/moving_nonobstructing_box_pairs_v4_candidate.csv'
expected={'checkpoint':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251','backbone':'615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e','m05':'3e872b9cdd37f6368b73308e5fd8212cdc3964221da44537cc322d24c49e9e05','decoder':'35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08','head':'6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def tensorhash(t):
 a=t.detach().cpu().contiguous().numpy();h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def tensors(s):return {'radius':s['radius'],'confidence':s['confidence'],'rays_map':s['rays_map'],**{'feature'+str(i):x for i,x in enumerate(s['features'])}}
def hashes(s):return {k:tensorhash(v) for k,v in tensors(s).items()}
def save():
 for name,value in [('manifest',meta),('trace',trace),('branch_audit',audit)]: (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False))
meta={'stage':'STARTED','selected_dataset':'Bonn','selected_sequence':'moving_nonobstructing_box','selection_rule':'first Bonn sequence in frozen P2B manifest; fixed before outcomes','frames':8,'repetitions':2,'gpu':None,'checkpoint_path':str(cp),'checkpoint_bytes':None,'identities':{},'errors':[],'scope':'fresh model and per-run cache; no exhaustive hidden-state proof'};trace=[];audit=[]
try:
 paths={'checkpoint':cp,'backbone':snap/'model.safetensors','m05':root/'experiments/p1/cvpr_m05_formal.py','decoder':root/'velodepth/models/decoder_velo.py','head':root/'velodepth/models/persistent_feature_repair.py'}
 for k,p in paths.items():meta['identities'][k]=sha(p);assert meta['identities'][k]==expected[k],k+' mismatch'
 meta['checkpoint_bytes']=cp.stat().st_size;meta['model_config_sha256']=sha(snap/'config.json');meta['pair_csv_sha256']=sha(manifest);meta['torch']=torch.__version__;meta['cuda']=torch.version.cuda
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 meta['gpu']=torch.cuda.get_device_name(0);meta['memory_free_total']=list(torch.cuda.mem_get_info());meta['stage']='IDENTITY_PASS';save()
 fold=json.loads((out/'fold_00.json').read_text());assert (fold['heldout_dataset'],fold['heldout_sequence_id'])==('Bonn','moving_nonobstructing_box') and fold['alpha']==1
 records=exp1.read_manifest(manifest,training=False);assert len(records)>=8
 initial_ref=None;repeat_reference={};used=0
 for rep in range(1):
  for group in ['G1']:
   torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);device=torch.device('cuda:0');model=gt2.load_model(snap,'pre_readout',device)
   ck=torch.load(cp,map_location='cpu',weights_only=False);assert 'repair_head_state_dict' in ck;model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
   meta['checkpoint_load']='PASS_STRICT';meta['loader']='gt2.load_model + persistent_feature_repair.load_state_dict(repair_head_state_dict, strict=True)';meta['decoder_class']=type(model.pixel_decoder).__module__+'.'+type(model.pixel_decoder).__name__;meta['head_class']=type(model.pixel_decoder.persistent_feature_repair).__module__+'.'+type(model.pixel_decoder.persistent_feature_repair).__name__;meta['stage']='MODEL_LOADED';save()
   shape=exp1.shape_info_for(model,Path(records[0]['rgb_path']),device);model._m05_shape=shape;frames=[exp1.load_frame(records[i],shape,device,need_gt=False) for i in range(8)];dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
   with torch.no_grad():rays,r,c,features=exp1.keyframe_state(model,frames[0],shape)
   state={'radius':r,'confidence':c,'features':features,'rays_map':rays};initial=hashes(state)
   if initial_ref is None:initial_ref=initial
   if initial!=initial_ref:raise RuntimeError('initial parent mismatch: '+group+' rep '+str(rep))
   meta['initial_parent_hashes']=initial;prev_after=None
   for i in range(1,2):
    static=exp1.transition_static(model,frames[i-1],frames[i],shape,dis);parent=m05.clone_state(state);before=hashes(parent)
    if prev_after is not None and before!=prev_after:raise RuntimeError('previous after != next parent')
    tri=[m05.clone_state(parent) for _ in range(3)]
    for src in range(3):
     for k,x in tensors(tri[src]).items():
      ptrs=[tensors(z)[k].untyped_storage().data_ptr() for z in tri]
      if len(set(ptrs))!=3:raise RuntimeError('three-way shared storage '+k)
      before_others=[tensorhash(tensors(tri[j])[k]) for j in range(3) if j!=src]
      old=x.reshape(-1)[0].item();x.reshape(-1)[0]=old+1
      assert before_others==[tensorhash(tensors(tri[j])[k]) for j in range(3) if j!=src]
      x.reshape(-1)[0]=old
      audit.append({'group':['G1','G2','G3'][src],'state_field':k,'pointers':ptrs,'mutation_isolated':True})
    a=m05.clone_state(parent);b=m05.clone_state(parent)
    for k,x in tensors(a).items():
     y=tensors(b)[k];pa=x.untyped_storage().data_ptr();pb=y.untyped_storage().data_ptr()
     if pa==pb:raise RuntimeError('shared mutable tensor '+k)
     if x.numel():
      by=tensorhash(y);old=x.reshape(-1)[0].item();x.reshape(-1)[0]=old+1
      if tensorhash(y)!=by:raise RuntimeError('mutation propagated '+k)
      x.reshape(-1)[0]=old
     if rep==0 and i==1:audit.append({'repetition':rep,'group':group,'state_field':k,'parent_clones_storage_distinct':True,'mutation_isolated':True,'pointer_a':pa,'pointer_b':pb})
    rng=m05.capture_rng(torch)
    m05.set_mode(model,'off');m05.restore_rng(torch,rng);s0=m05.run_step(model,exp1,frames[i-1],frames[i],static,a,torch)
    m05.set_mode(model,'pre_readout');m05.restore_rng(torch,rng);s1=m05.run_step(model,exp1,frames[i-1],frames[i],static,b,torch)
    layer=[];norms=[]
    for l,(x,y) in enumerate(zip(s0['features'],s1['features'])):
     if x.shape!=y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():raise RuntimeError('invalid S0/S1')
     delta=y.float()-x.float();norms.append(float(torch.linalg.vector_norm(delta.double()).item()));layer.append({'layer':l,'shape':list(x.shape),'S0_dtype':str(x.dtype),'S1_dtype':str(y.dtype),'device':str(x.device),'finite':True,'S0_sha256':tensorhash(x),'S1_sha256':tensorhash(y),'update_l2':norms[-1]})
    if not any(v>0 for v in norms):raise RuntimeError('S0/S1 candidate identical')
    norm=[math.sqrt(sum(z*z for z in norms))]+norms;pred=fold['ridge_intercept']+math.fsum((v-m)/s*c for v,m,s,c in zip(norm,fold['scaler_mean'],fold['scaler_scale'],fold['ridge_coef']));assert math.isfinite(pred)
    commit=group in ('G0','G1') or (group=='G3' and pred>0);selected=s1 if group=='G0' else {'radius':s0['radius'],'confidence':s0['confidence'],'rays_map':parent['rays_map'],'features':s1['features'] if commit else s0['features']};state=m05.clone_state(selected);after=hashes(state)
    for l in range(4):
     want=tensorhash(s1['features'][l] if commit else s0['features'][l]);assert after['feature'+str(l)]==want
    if group!='G0':assert after['radius']==tensorhash(s0['radius']) and after['confidence']==tensorhash(s0['confidence']) and after['rays_map']==tensorhash(parent['rays_map'])
    event={'repetition':rep,'group':group,'frame':i,'parent_hashes':before,'feature_hash_after':{k:v for k,v in after.items() if k.startswith('feature')},'state_hash_after':after,'commit':commit,'predicted_utility':pred,'norm_global':norm[0],'layer':layer,'next_parent_source':'own previous committed state','nonfeature_source':'S1 native' if group=='G0' else 'S0 radius/confidence and parent rays'};trace.append(event);prev_after=after;save();print('STEP',rep,group,i,flush=True)
   key=group;signature=[(e['frame'],e['commit'],e['state_hash_after']) for e in trace if e['group']==group and e['repetition']==rep]
   if rep==0:repeat_reference[key]=signature
   elif signature!=repeat_reference[key]:raise RuntimeError('reset repeat mismatch '+group)
   del model,frames,static,state,parent,a,b,s0,s1,features,rays,r,c,shape,dis;gc.collect();torch.cuda.empty_cache();used+=1;print('RESET',rep,group,flush=True)
 meta['stage']='COMPLETED';meta['runs']=used;meta['reset_repeat_exact']=True;meta['caveats']=['fresh model and DIS for each run; general hidden/global state inventory not exhaustive','same initial state established by hashes; groups run serially and independently','current D1 equal across same parent not explicitly measured','no GT loaded, no depth evaluation, no formal branch metrics'];save();print('PROBE_COMPLETE',flush=True)
except BaseException as e:
 meta['stage']='FAILED';meta['errors'].append(type(e).__name__+': '+str(e));save();traceback.print_exc();raise
