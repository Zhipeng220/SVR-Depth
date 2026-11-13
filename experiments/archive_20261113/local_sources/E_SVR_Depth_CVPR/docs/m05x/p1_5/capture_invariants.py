"""P1.5 single-anchor developer evidence capture. No training or test set."""
import csv, hashlib, json, os, sys
from pathlib import Path
import cv2, numpy as np, torch

ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1, gt2_fixed_clip_repair_train as gt2, cvpr_m05_formal as m05

def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def arr(t): return t.detach().cpu().contiguous().numpy().copy() if torch.is_tensor(t) else np.array(t,copy=True)
def fields(state):
    return {'radius':state['radius'],'confidence':state['confidence'],'rays_map':state['rays_map'],**{f'feature{i}':v for i,v in enumerate(state['features'])}}
def put_state(target,prefix,state):
    for key,value in fields(state).items(): target[prefix+'__'+key]=arr(value)

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
cp=ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt'
snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c'
manifest=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940/moving_obstructing_box_pairs_v4_candidate.csv'
expected={'checkpoint':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251','backbone':'615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e','m05':'3e872b9cdd37f6368b73308e5fd8212cdc3964221da44537cc322d24c49e9e05'}
actual={'checkpoint':filehash(cp),'backbone':filehash(snap/'model.safetensors'),'m05':filehash(ROOT/'experiments/p1/cvpr_m05_formal.py'),'pair_csv':filehash(manifest),'capture_script':filehash(__file__)}
for k,v in expected.items():
    if actual[k]!=v:raise RuntimeError('frozen identity mismatch: '+k)
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025)
device=torch.device('cuda:0');model=gt2.load_model(snap,'pre_readout',device)
model.pixel_decoder.persistent_feature_repair.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['repair_head_state_dict'],strict=True);model.eval()
records=exp1.read_manifest(manifest,training=False); anchor=int(sys.argv[2]) if len(sys.argv)>2 else 1
if anchor not in (1,4):raise ValueError('allowed predetermined ages: 1,4')
if any(str(records[i].get('gt_match_status','')).upper()!='MATCHED' for i in range(anchor+1,anchor+5)):raise RuntimeError('GT missing in predetermined window')
shape=exp1.shape_info_for(model,Path(records[0]['rgb_path']),device); model._m05_shape=shape
# GT is not loaded into inference frames; load it only after both replays finish.
frames=[exp1.load_frame(records[i],shape,device,need_gt=False) for i in range(anchor+5)]
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
with torch.no_grad(): static={i:exp1.transition_static(model,frames[i-1],frames[i],shape,dis) for i in range(1,anchor+5)}
rng=m05.capture_rng(torch);parents=m05.build_parent_states(model,exp1,frames,static,0,[anchor],'off',rng,torch);parent=parents[anchor]
o_parent=m05.clone_state(parent);t_parent=m05.clone_state(parent); arrays={};put_state(arrays,'parent_O',o_parent);put_state(arrays,'parent_T',t_parent)
arrays['rng_cpu']=arr(rng['cpu'])
for i,x in enumerate(rng['cuda']): arrays[f'rng_cuda_{i}']=arr(x)
alias={}
for name,x in fields(o_parent).items():
    y=fields(t_parent)[name]
    before=arr(y);old=x.reshape(-1)[0].item(); x.reshape(-1)[0]=old+1
    arrays['alias_O_before__'+name]=np.array(old);arrays['alias_O_mutated__'+name]=np.array(x.reshape(-1)[0].item())
    after=arr(y); x.reshape(-1)[0]=old
    arrays['alias_before_T__'+name]=before;arrays['alias_after_T__'+name]=after
    alias[name]={'o_ptr':int(x.untyped_storage().data_ptr()),'t_ptr':int(y.untyped_storage().data_ptr()),'mutation_old':old,'mutation_value':old+1}
parent_rng=m05.capture_rng(torch)
with torch.no_grad():
    m05.set_mode(model,'off');m05.restore_rng(torch,parent_rng);s0=m05.run_step(model,exp1,frames[anchor-1],frames[anchor],static[anchor],o_parent,torch)
    m05.set_mode(model,'pre_readout');m05.restore_rng(torch,parent_rng);s1=m05.run_step(model,exp1,frames[anchor-1],frames[anchor],static[anchor],t_parent,torch)
    d1=m05.make_prediction(model,gt2,s1,shape)
    arrays['current_D1_O']=np.array(d1,copy=True);arrays['current_D1_T']=np.array(d1,copy=True)
    branch_O=m05.clone_state({'radius':s0['radius'],'confidence':s0['confidence'],'features':s0['features'],'rays_map':parent['rays_map']})
    branch_T=m05.clone_state({'radius':s0['radius'],'confidence':s0['confidence'],'features':s1['features'],'rays_map':parent['rays_map']})
    put_state(arrays,'committed_O',branch_O);put_state(arrays,'committed_T',branch_T)
    committed_alias={}
    for name,x in fields(branch_O).items():
        y=fields(branch_T)[name];before=arr(y);old=x.reshape(-1)[0].item();x.reshape(-1)[0]=old+1
        arrays['commit_alias_O_before__'+name]=np.array(old);arrays['commit_alias_O_mutated__'+name]=np.array(x.reshape(-1)[0].item())
        arrays['commit_alias_before_T__'+name]=before;arrays['commit_alias_after_T__'+name]=arr(y);x.reshape(-1)[0]=old
        committed_alias[name]={'o_ptr':int(x.untyped_storage().data_ptr()),'t_ptr':int(y.untyped_storage().data_ptr())}
    for branch,init in [('O',branch_O),('T',branch_T)]:
        m05.set_mode(model,'off');m05.restore_rng(torch,parent_rng);state=init
        branch_rng=m05.capture_rng(torch);arrays[f'replay_rng_{branch}_cpu']=arr(branch_rng['cpu'])
        for i,x in enumerate(branch_rng['cuda']): arrays[f'replay_rng_{branch}_cuda_{i}']=arr(x)
        arrays[f'input_{branch}_anchor']=arr(frames[anchor]['img'])
        for h in range(1,5):
            idx=anchor+h
            arrays[f'input_{branch}_{h}']=arr(frames[idx]['img'])
            for k,v in static[idx].items():arrays[f'static_{branch}_{h}__{k}']=arr(v)
            state=m05.run_step(model,exp1,frames[idx-1],frames[idx],static[idx],state,torch)
            arrays[f'pred_{branch}_{h}']=m05.make_prediction(model,gt2,state,shape)
            for k,v in static[idx].items():arrays[f'static_after_{branch}_{h}__{k}']=arr(v)
# The GT is first loaded after inference is complete.
metrics={}; inputs=[]
for i in range(anchor+5):
    p=Path(records[i]['rgb_path']);inputs.append({'frame_index':i,'rgb_path':str(p),'rgb_sha256':filehash(p)})
for h in range(1,5):
    idx=anchor+h;gt=exp1.load_frame(records[idx],shape,device,need_gt=True)['gt_np'];arrays[f'gt_{h}']=gt.copy()
    for branch in ('O','T'):
        pred=arrays[f'pred_{branch}_{h}'];arrays[f'gt_valid_{branch}_{h}']=np.isfinite(gt)&(gt>0)
        arrays[f'metric_valid_{branch}_{h}']=arrays[f'gt_valid_{branch}_{h}']&np.isfinite(pred)&(pred>0)
        metrics[f'{branch}_H{h}']=gt2.calculate_metrics(gt,pred)
    inputs[idx]['gt_path']=records[idx]['depth_path'];inputs[idx]['gt_sha256']=filehash(records[idx]['depth_path'])
np.savez_compressed(out/'evidence_arrays.npz',**arrays)
meta={'experiment':'P1.5 NEW RUN; not historical evidence','checkpoint':'V5','identity':actual,'seed':20261025,'dataset':'Bonn development moving_obstructing_box','cycle':0,'age':anchor,'anchor':anchor,'horizons':[1,2,3,4],'repair_replay_mode':'off','gt_loaded_after_replays':True,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'alias_runtime':alias,'committed_alias_runtime':committed_alias,'inputs':inputs,'metrics':metrics}
(out/'capture_metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
print('CAPTURE_COMPLETE',out,flush=True)
