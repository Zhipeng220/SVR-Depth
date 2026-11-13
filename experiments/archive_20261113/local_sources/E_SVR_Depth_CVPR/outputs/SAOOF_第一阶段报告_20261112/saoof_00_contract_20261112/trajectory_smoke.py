"""SAOOF-00 limited native/reference isolation and geometry smoke check."""
import sys, glob, json, copy
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
sys.path.insert(0, 'experiments/p1')
import gt2_fixed_clip_repair_train as gt2
import gt4a2_exp1_train_eval as exp1
from velodepth.utils.camera import Pinhole

device = torch.device('cuda')
snapshot = Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
paths = sorted(glob.glob('/home/svrdepth/datasets/bonn/rgbd_bonn_person_tracking2/rgb/*.png'))[:5]
assert len(paths) == 5
native = gt2.load_model(snapshot, 'off', device)
reference = gt2.load_model(snapshot, 'off', device)
assert native.pixel_decoder is not reference.pixel_decoder
shape = exp1.shape_info_for(native, Path(paths[0]), device)
assert tuple(shape['network_shape']) == (434, 574)
def frame(i, shp):
    return exp1.load_frame({'frame_index_zero_based':i, 'rgb_path':paths[i], 'rgb_timestamp':0}, shp, device, need_gt=False)
frames = [frame(i, shape) for i in range(len(paths))]
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
statics = [exp1.transition_static(native, frames[i-1], frames[i], shape, dis) for i in range(1,len(paths))]

def reference_key(i, size):
    h,w=size
    s=shape.copy(); s['network_shape']=size
    s['validity_mask']=torch.ones((1,1,h,w),device=device,dtype=torch.bool)
    f=frame(i,s); x=f['img']
    # Synthetic pinhole K tests use of supplied rays only, not Bonn calibration.
    params=torch.tensor([[500*w/640,500*h/480,320*w/640,240*h/480]],device=device,dtype=torch.float32)
    rays=Pinhole(params=params).get_rays((1,h,w))
    inputs={'image':x,'image_original':x,'validity_mask':s['validity_mask'],'rays':rays}
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        features,tokens=reference.pixel_encoder(x)
        inputs['features']=[reference.stacking_fn(features[j:k]).contiguous() for j,k in reference.slices_encoder_range]
        inputs['tokens']=[reference.stacking_fn(tokens[j:k]).contiguous() for j,k in reference.slices_encoder_range]
        reference.pixel_decoder.set_shapes(inputs,[{'B':1,'T':1}])
        out=reference.pixel_decoder.forward_keyframe(inputs,[{'B':1,'T':1}])
    ray_out=out[0].reshape(1,h,w,3).permute(0,3,1,2)
    radius=out[1].float(); z=(radius*ray_out[:,2:3].float())
    valid=torch.isfinite(z)&(z>0)
    # Interpolation applies only to positive finite depth; all pixels required in this smoke case.
    assert valid.all().item()
    aligned=F.interpolate(z.log(),size=shape['network_shape'],mode='bilinear',align_corners=False).exp()
    return {'size':list(size),'ray_max_abs':float((ray_out-rays).abs().max()),'valid_depth_fraction':float(valid.float().mean()),'aligned_shape':list(aligned.shape),'aligned_finite_positive':bool((torch.isfinite(aligned)&(aligned>0)).all())}

def trajectory(with_ref):
    rays,r,c,feat=exp1.keyframe_state(native,frames[0],shape)
    saved=[(r.float().cpu(),c.float().cpu(),[x.float().cpu() for x in feat])]
    checks=[]
    for i in range(1,len(frames)):
        before=(tuple(native.pixel_decoder.common_shape),tuple(native.pixel_decoder.radial_module.original_shapes),tuple(native.pixel_decoder.angular_module.shapes))
        if with_ref:
            checks.append(reference_key(i,(224,294) if i%2 else (294,392)))
        after=(tuple(native.pixel_decoder.common_shape),tuple(native.pixel_decoder.radial_module.original_shapes),tuple(native.pixel_decoder.angular_module.shapes))
        assert before==after
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
            r,c,feat=exp1.one_fast_step(native,frames[i-1],frames[i],statics[i-1],r,c,feat)
        saved.append((r.float().cpu(),c.float().cpu(),[x.float().cpu() for x in feat]))
    return saved,checks

control,_=trajectory(False)
test,checks=trajectory(True)
diff=[]
for idx,(a,b) in enumerate(zip(control,test)):
    diff.append({'frame':idx,'radius_max_abs':float((a[0]-b[0]).abs().max()),'confidence_max_abs':float((a[1]-b[1]).abs().max()),'feature_max_abs':[float((x-y).abs().max()) for x,y in zip(a[2],b[2])]})
result={'status':'ENGINEERING_CHECK_ONLY','snapshot':str(snapshot),'frames':paths,'native_shape':list(shape['network_shape']),'reference_checks':checks,'trajectory_differences':diff,'synthetic_intrinsics':True,'limits':'No production fusion, fallback, actual calibration, or official evaluation baseline validated.'}
out=Path('outputs/saoof_00_contract_20261112'); out.mkdir(parents=True,exist_ok=True)
(out/'trajectory_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
