"""SAOOF-00 native/reference streaming adapter for the Bonn raw-frame path.

Call start on the native keyframe then step once per subsequent frame. Returned
depth is a separate output; propagation state is private and stays native.
This adapter is independent of the batch-only official forward_test hook.
"""
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
import gt4a2_exp1_train_eval as exp1
from saoof_stream_output import SAOOFStreamOutput

class SAOOFNativeStream:
 def __init__(self,native,reference,shape,ref_size=(224,294)):
  if native is reference or native.pixel_decoder is reference.pixel_decoder: raise ValueError('independent models and decoders required')
  if native.pixel_decoder.camera_gt or reference.pixel_decoder.camera_gt: raise ValueError('predicted-ray convention required')
  self.native=native; self.reference=reference; self.shape=shape; self.ref_size=tuple(ref_size)
  self.output=SAOOFStreamOutput(); self.dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
  self.started=False

 def _ref(self,frame):
  s=self.shape.copy(); s['network_shape']=self.ref_size
  s['validity_mask']=torch.ones((1,1,*self.ref_size),device=frame['img'].device,dtype=torch.bool)
  row={'frame_index_zero_based':frame['index'],'rgb_path':str(frame['rgb_path']),'rgb_timestamp':frame['rgb_timestamp']}
  rf=exp1.load_frame(row,s,frame['img'].device,need_gt=False)
  rays,r,_,_=exp1.keyframe_state(self.reference,rf,s)
  z=(r.float()*rays[:,2:3].float())
  if not bool((torch.isfinite(z)&(z>0)).all()): return None
  return F.interpolate(z.log(),size=self.shape['network_shape'],mode='bilinear',align_corners=False).exp()[0,0]

 def start(self,frame,geometry_mask,reference_enabled=True):
  self.rays,self.radius,self.confidence,self.features=exp1.keyframe_state(self.native,frame,self.shape)
  self.previous=frame; self.started=True
  depth=(self.radius.float()*self.rays[:,2:3].float())[0,0]
  reference=self._ref(frame) if reference_enabled else None
  self.output.begin_cycle(depth,reference,geometry_mask)
  return depth.clone(),{'status':'keyframe_native'}

 def step(self,frame,geometry_mask,gamma,eta,reference_enabled=True,flow_override=None):
  if not self.started: raise RuntimeError('start must be called first')
  static=exp1.transition_static(self.native,self.previous,frame,self.shape,self.dis)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
   r,c,features=exp1.one_fast_step(self.native,self.previous,frame,static,self.radius,self.confidence,self.features)
  self.radius,self.confidence,self.features=r,c,features
  depth=(r.float()*self.rays[:,2:3].float())[0,0]
  reference=self._ref(frame) if reference_enabled else None
  if reference is not None:
   a=self.previous['frame_u8'].permute(1,2,0).numpy(); b=frame['frame_u8'].permute(1,2,0).numpy()
   a=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY); b=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)
   ff=self.dis.calc(a,b,None); fb=self.dis.calc(b,a,None)
   if flow_override is not None: ff,fb=flow_override
  else: ff=fb=None
  chosen,status=self.output.process(depth,reference,ff,fb,geometry_mask,gamma,eta)
  self.previous=frame
  return chosen,{'status':status,'native_depth':depth.clone()}
