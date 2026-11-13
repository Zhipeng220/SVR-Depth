"""SAOOF-00 camera-geometry and output integration smoke, not an accuracy test."""
import glob, sys, json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, 'experiments/p1')
import gt2_fixed_clip_repair_train as gt2
import gt4a2_exp1_train_eval as exp1
from saoof_v1_core import keyframe_anchor, decompose, fuse

device=torch.device('cuda')
root=Path('outputs/saoof_00_contract_20261112')
snapshot=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
files=sorted(glob.glob('/home/svrdepth/datasets/bonn/rgbd_bonn_person_tracking2/rgb/*.png'))[:2]
native=gt2.load_model(snapshot,'off',device); ref=gt2.load_model(snapshot,'off',device)
shape=exp1.shape_info_for(native,Path(files[0]),device)
assert shape['original_shape']==(480,640) and shape['network_shape']==(434,574)
K=np.array([[542.822841,0,315.593520],[0,542.576870,237.756098],[0,0,1]],np.float64)
dist=np.array([.039903,-.099343,-.000730,-.000144,0.],np.float64)
padl,padr,padt,padb=shape['paddings']; ph,pw=shape['padded_shape']

def f(i,sh):
 return exp1.load_frame({'frame_index_zero_based':i,'rgb_path':files[i],'rgb_timestamp':0},sh,device,need_gt=False)

def rays_for(size):
 h,w=size; sx=w/pw; sy=h/ph
 yy,xx=np.mgrid[:h,:w].astype(np.float64)
 # Inverse of pad followed by torch interpolate(align_corners=False).
 rawx=(xx+.5)/sx-.5-padl; rawy=(yy+.5)/sy-.5-padt
 coords=np.stack([rawx,rawy],axis=-1).reshape(-1,1,2)
 xy=cv2.undistortPoints(coords,K,dist).reshape(h,w,2)
 xyz=np.concatenate([xy,np.ones((h,w,1))],axis=2)
 normalized=xyz/np.linalg.norm(xyz,axis=2,keepdims=True)
 valid=(rawx>=0)&(rawx<=639)&(rawy>=0)&(rawy<=479)
 samples=xyz[valid][::max(1,int(valid.sum())//1000)].reshape(-1,1,3)
 projected,_=cv2.projectPoints(samples,np.zeros(3),np.zeros(3),K,dist)
 original=coords.reshape(h,w,2)[valid][::max(1,int(valid.sum())//1000)]
 err=np.linalg.norm(projected.reshape(-1,2)-original,axis=1)
 return torch.from_numpy(normalized).permute(2,0,1).unsqueeze(0).to(device=device,dtype=torch.float32),valid,{'sample_count':len(err),'reprojection_max_pixels':float(err.max()),'reprojection_p95_pixels':float(np.percentile(err,95)),'resize_factors':[sx,sy],'padding_lrtb':list(shape['paddings'])}

def reference(i,size):
 h,w=size; s=shape.copy(); s['network_shape']=size; s['validity_mask']=torch.ones((1,1,h,w),device=device,dtype=torch.bool)
 x=f(i,s)['img']; rays,valid,geom=rays_for(size)
 inp={'image':x,'image_original':x,'validity_mask':s['validity_mask'],'rays':rays}
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
  feats,tokens=ref.pixel_encoder(x)
  inp['features']=[ref.stacking_fn(feats[j:k]).contiguous() for j,k in ref.slices_encoder_range]
  inp['tokens']=[ref.stacking_fn(tokens[j:k]).contiguous() for j,k in ref.slices_encoder_range]
  ref.pixel_decoder.set_shapes(inp,[{'B':1,'T':1}]); ray_out,radius,_,_=ref.pixel_decoder.forward_keyframe(inp,[{'B':1,'T':1}])
 ray_out=ray_out.reshape(1,h,w,3).permute(0,3,1,2)
 z=(radius.float()*ray_out[:,2:3].float()).clamp_min(1e-12)
 z=F.interpolate(z.log(),size=shape['network_shape'],mode='bilinear',align_corners=False).exp()[0,0]
 mask=F.interpolate(torch.from_numpy(valid.astype(np.float32))[None,None].to(device),size=shape['network_shape'],mode='nearest')[0,0]>.5
 z=torch.where(mask,z,torch.nan)
 return z,{'size':list(size),'ray_error':float((ray_out-rays).abs().max()),'finite_positive_fraction':float((torch.isfinite(z)&(z>0)).float().mean()),**geom}

f0,f1=f(0,shape),f(1,shape)
rays,r,c,feats=exp1.keyframe_state(native,f0,shape)
prop0=(r.float()*rays[:,2:3].float())[0,0]
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
static=exp1.transition_static(native,f0,f1,shape,dis)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
 r1,c1,feats1=exp1.one_fast_step(native,f0,f1,static,r,c,feats)
prop1=(r1.float()*rays[:,2:3].float())[0,0]
assert bool((torch.isfinite(prop0)&(prop0>0)&torch.isfinite(prop1)&(prop1>0)).all())
checks=[]
for sz in ((224,294),(294,392)):
 ref0,g0=reference(0,sz); ref1,g1=reference(1,sz)
 anchor=keyframe_anchor(prop0,ref0); assert anchor is not None
 # All-true correspondence is a deliberately artificial integration input, not F/B flow.
 m,local,valid,reason=decompose(prop1,ref1,anchor,torch.ones_like(prop1,dtype=torch.bool)); assert reason is None
 zero=torch.zeros_like(prop1); one=torch.ones_like(prop1)
 o0=fuse(prop1,m,local,valid,0.,zero); o1=fuse(prop1,m,local,valid,1.,one)
 endpoint=(ref1[valid]*anchor.exp()); err=(o1[valid]-endpoint).abs().max()
 fallback=fuse(prop1,*decompose(prop1,ref1,None,zero)[:2])
 checks.append({'size':list(sz),'keyframe_geometry':g0,'frame1_geometry':g1,'anchor':float(anchor),'m':float(m),'gate0_max_abs':float((o0-prop1).abs().max()),'gate1_ref_max_abs':float(err),'invalid_anchor_exact_fallback':bool(torch.equal(fallback,prop1)),'correspondence':'synthetic_all_true'})
result={'status':'ENGINEERING_CHECK_ONLY','native_baseline':'historical predicted rays','rgb_paths':files,'official_calibration_source':'https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/','checks':checks,'limitations':'No real flow correspondence, official eval parity, full K16 trajectory, or end-to-end latency.'}
(root/'geometry_integration_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
