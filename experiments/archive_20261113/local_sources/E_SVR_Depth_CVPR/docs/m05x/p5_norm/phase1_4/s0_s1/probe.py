import sys,json,hashlib,math
from pathlib import Path
import torch,cv2
root=Path.cwd();sys.path.insert(0,str(root/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1,gt2_fixed_clip_repair_train as gt2,cvpr_m05_formal as m05
out=Path(sys.argv[1]);cp=root/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt';snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';manifest=root/'outputs/y04_v4_manifest_candidate_20261108_190940/moving_nonobstructing_box_pairs_v4_candidate.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def th(t):
 a=t.detach().cpu().contiguous().numpy();return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
assert sha(cp)=='2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251'
assert sha(root/'experiments/p1/cvpr_m05_formal.py')=='3e872b9cdd37f6368b73308e5fd8212cdc3964221da44537cc322d24c49e9e05'
fold=json.loads((out/'fold.json').read_text());assert fold['heldout_sequence_id']=='moving_nonobstructing_box'
torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);dev=torch.device('cuda:0');model=gt2.load_model(snap,'pre_readout',dev);model.pixel_decoder.persistent_feature_repair.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['repair_head_state_dict'],strict=True);model.eval();rows=exp1.read_manifest(manifest,training=False);shape=exp1.shape_info_for(model,Path(rows[0]['rgb_path']),dev);model._m05_shape=shape;frames=[exp1.load_frame(rows[i],shape,dev,need_gt=False) for i in range(3)];dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
with torch.no_grad():rays,r,c,f=exp1.keyframe_state(model,frames[0],shape)
state={'radius':r,'confidence':c,'features':f,'rays_map':rays};events=[]
for i in [1,2]:
 static=exp1.transition_static(model,frames[i-1],frames[i],shape,dis);parent=m05.clone_state(state);capture={}
 def hook(mod,args,result):
  capture['s0']=[x.detach().clone() for x in args[0]];capture['s1']=[x.detach().clone() for x in result]
 handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook)
 try:
  m05.set_mode(model,'pre_readout');s1=m05.run_step(model,exp1,frames[i-1],frames[i],static,m05.clone_state(parent),torch)
 finally:handle.remove()
 assert len(capture)==2 and all(torch.equal(a,b) for a,b in zip(capture['s1'],s1['features']))
 s0=capture['s0'];s1f=capture['s1']; norms=[float(torch.linalg.vector_norm((b.float()-a.float()).double())) for a,b in zip(s0,s1f)];x=[math.sqrt(sum(n*n for n in norms))]+norms;prediction=fold['ridge_intercept']+math.fsum((v-m)/s*w for v,m,s,w in zip(x,fold['scaler_mean'],fold['scaler_scale'],fold['ridge_coef']))
 for layer,(a,b,par) in enumerate(zip(s0,s1f,parent['features'])):
  assert torch.isfinite(a).all() and torch.isfinite(b).all();events.append({'frame':i,'layer':layer,'s0_shape':list(a.shape),'s0_dtype':str(a.dtype),'device':str(a.device),'s0_hash':th(a),'s1_hash':th(b),'parent_hash':th(par),'s0_diff_parent':not torch.equal(a,par),'s0_diff_s1':not torch.equal(a,b),'g1_selected':th(b),'g2_selected':th(a),'g3_selected':th(b if prediction>0 else a),'predicted_utility':prediction,'commit':prediction>0})
 state=m05.clone_state({'radius':s1['radius'],'confidence':s1['confidence'],'features':s1f,'rays_map':parent['rays_map']})
(out/'events.json').write_text(json.dumps(events,indent=2));print('PASS_EVENTS',len(events),flush=True)
