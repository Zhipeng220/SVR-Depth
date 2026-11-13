"""One K16 raw-Bonn regression of the per-frame native streaming API."""
import sys,glob,json
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,'experiments/p1')
import gt2_fixed_clip_repair_train as gt2
import gt4a2_exp1_train_eval as exp1
from saoof_native_stream import SAOOFNativeStream

device=torch.device('cuda'); out=Path('outputs/saoof_00_closeout_20261112')
snap=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
paths=sorted(glob.glob('/home/svrdepth/datasets/bonn/rgbd_bonn_person_tracking2/rgb/*.png'))[:16]
native=gt2.load_model(snap,'off',device); reference=gt2.load_model(snap,'off',device)
native.pixel_decoder.camera_gt=False; reference.pixel_decoder.camera_gt=False
shape=exp1.shape_info_for(native,Path(paths[0]),device); assert shape['paddings']==(0,0,0,0)
def get(i): return exp1.load_frame({'frame_index_zero_based':i,'rgb_path':paths[i],'rgb_timestamp':0},shape,device,need_gt=False)
frames=[get(i) for i in range(16)]; mask=shape['validity_mask'][0,0]; h,w=shape['network_shape']
badflow=(np.zeros((h,w,2),np.float32),np.full((h,w,2),np.nan,np.float32))
def run(mode):
 size=(294,392) if mode=='valid_mid' else (224,294)
 stream=SAOOFNativeStream(native,reference,shape,ref_size=size); first,_=stream.start(frames[0],mask,reference_enabled=mode not in ('control','missing'))
 if mode=='invalid_anchor': stream.output.begin_cycle(first,None,mask)
 states=[]; depths=[]; chosen=[]; statuses=[]
 for frame in frames[1:]:
  output,info=stream.step(frame,mask,.5,torch.full((h,w),.5,device=device),reference_enabled=mode not in ('control','missing'),flow_override=badflow if mode=='insufficient_mask' else None)
  states.append([stream.radius.float().cpu(),stream.confidence.float().cpu(),*[x.float().cpu() for x in stream.features]])
  depths.append(info['native_depth'].cpu()); chosen.append(output.cpu()); statuses.append(info['status'])
 return states,depths,chosen,statuses
control,control_d,_,_=run('control'); results={}
for mode in ('missing','invalid_anchor','insufficient_mask','valid','valid_mid'):
 states,depths,chosen,status=run(mode)
 sd=max(float((a-b).abs().max()) for u,v in zip(control,states) for a,b in zip(u,v))
 dd=max(float((a-b).abs().max()) for a,b in zip(control_d,depths))
 delta=max(float((a-b).abs().max()) for a,b in zip(chosen,depths))
 equal=all(torch.equal(a,b) for a,b in zip(chosen,depths))
 assert sd==0 and dd==0
 if mode in ('valid','valid_mid'): assert delta>1e-6 and all(s=='ok' for s in status)
 else: assert equal
 results[mode]={'native_state_max_abs':sd,'native_depth_max_abs':dd,'chosen_vs_native_max_abs':delta,'fallback_bitwise':equal,'statuses':status}
summary={'status':'ENGINEERING_CHECK_ONLY','source':'Bonn person_tracking2 raw first 16 frames','shapes':list(shape['network_shape']),'reference_sizes':[[224,294],[294,392]],'camera_convention':'both predicted','results':results,'limitations':'Single raw K16 clip; independent of HDF official sequence. Test gamma=eta=.5; no GT or accuracy assessment. Caller supplies geometry validity and gates.'}
(out/'native_stream_k16.json').write_text(json.dumps(summary,indent=2),encoding='utf8'); print(json.dumps(summary,indent=2))
