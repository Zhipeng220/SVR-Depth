"""CUDA adapters and frame-at-a-time fixed-K / original Auto state machine."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from y08y09_common import require, repair_due, FLOOR


class ChannelAffine(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.ParameterList([nn.Parameter(torch.ones(1,512,1,1)) for _ in range(4)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1,512,1,1)) for _ in range(4)])

    def forward(self, features):
        require(len(features)==4 and all(x.shape[1]==512 for x in features), 'B2 feature channels differ')
        return [x*self.scale[i].to(dtype=x.dtype)+self.bias[i].to(dtype=x.dtype) for i,x in enumerate(features)]


class FastDepthAdapter(nn.Module):
    """10,465 parameters; final depth + already available RGB, quarter resolution.

    Only emitted depth changes. No recurrent state, confidence, or decoder pass.
    The bounded log residual is declared up front, not tuned on validation.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4,32,3,padding=1), nn.SiLU(),
            nn.Conv2d(32,32,3,padding=1), nn.SiLU(), nn.Conv2d(32,1,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, depth, rgb_small, gain=1.):
        shape = depth.shape[-2:]
        small = F.interpolate(depth[None,None].float().clamp_min(FLOOR).log(),
                              size=rgb_small.shape[-2:], mode='bilinear', align_corners=False)
        delta = .5*torch.tanh(self.net(torch.cat([small,rgb_small],1)))
        delta = F.interpolate(delta.float(), size=shape, mode='bilinear', align_corners=False)[0,0]
        return depth.float()*torch.exp(float(gain)*delta)


class GainHead(nn.Module):
    def __init__(self, head, gain):
        super().__init__(); self.head=head; self.gain=gain; self.active=True; self.calls=0

    def forward(self, features):
        if not self.active: return features
        self.calls += 1
        if self.gain == 1.: return self.head(features)
        if hasattr(self.head,'blocks'):
            return [x+self.gain*block(x) for x,block in zip(features,self.head.blocks)]
        corrected=self.head(features)
        return [x+self.gain*(y-x) for x,y in zip(features,corrected)]


def readout_radius(model, features):
    r=model.pixel_decoder.radial_module
    lateral=[features[0]]+[r.process_features[i](features[i+1]) for i in range(len(r.ups))]
    latent=lateral[0]; outputs=[]
    for i,up in enumerate(r.ups):
        latent=up(latent+lateral[i+1]); outputs.append(latent)
    return torch.exp(r.depth_proj(outputs).clip(min=-8.,max=8.)+2.)


def rgb_for_output(frame, shape):
    rgb=F.interpolate(frame['img'].float(),size=shape['padded_shape'],mode='bilinear',align_corners=False)
    l,r,t,b=shape['paddings']; h,w=rgb.shape[-2:]
    rgb=rgb[:,:,t:h-b if b else h,l:w-r if r else w]
    return F.interpolate(rgb, size=(max(1,rgb.shape[-2]//4),max(1,rgb.shape[-1]//4)),
                         mode='bilinear',align_corners=False)


def versions(state):
    r,c,f,_=state
    return [(id(x),x._version) for x in (r,c,*f)]


def output_depth(rt, model, state, shape):
    r,c,f,rays=state
    return rt.gt2.make_z_depth(r,rays,shape['padded_shape'],shape['paddings'],model.interpolation_mode)


def transition(rt, model, previous, current, shape, dis):
    # Same audited DIS/preprocessing as Experiment1, retain raw pixels for Auto audit.
    np,cv2=rt.np,rt.cv2
    a=current['frame_u8'].permute(1,2,0).numpy(); b=previous['frame_u8'].permute(1,2,0).numpy()
    raw=dis.calc(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),None)
    px=torch.from_numpy(raw).to(current['img'].device).permute(2,0,1)[None]
    flow,grid=model._normalize_and_grid_from_flow(px.clone(),current['img'],shape['image_paddings'],shape['depth_paddings'])
    residual=rt.gt2.rgb2luma(current['img']).unsqueeze(1)-F.grid_sample(
        rt.gt2.rgb2luma(previous['img']).unsqueeze(1),grid,mode='bilinear')
    return dict(flow_norm=flow,grid_sample=grid,image_residual=residual,
                validity_mask=shape['validity_mask'],raw_flow_px=px)


class Stream:
    def __init__(self,rt,model,cfg,shape,adapter=None,debug=False):
        self.rt,self.model,self.cfg,self.shape,self.adapter=rt,model,cfg,shape,adapter
        self.debug=debug
        self.dis=rt.cv2.DISOpticalFlow_create(rt.cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.reset()

    def reset(self):
        self.previous=None; self.state=None; self.first=True; self.age=0
        self.flow_history=self.luma_history=0.; self.last_key=-1
        self.counts=dict(keyframe_calls=0,flow_calls=0,native_fast_calls=0,
            repair_head_calls=0,output_adapter_calls=0,additional_readout_passes=0,rejected_candidates=0)

    def keyframe(self,frame,index,auto_restart=False):
        rays,r,c,f=self.rt.exp1.keyframe_state(self.model,frame,self.shape)
        if auto_restart:
            # Original overlap=1, schedule[0]=0: new rays/features, candidate r/c.
            old_r,old_c,old_f,old_rays=self.state
            r=old_r; c=old_c
        self.state=(r,c,f,rays); self.last_key=index; self.age=1
        self.flow_history=self.luma_history=0.
        self.counts['keyframe_calls']+=1

    def fast(self,frame,static,apply):
        rt,m=self.rt,self.model; r,c,features,rays=self.state
        head=m.pixel_decoder.persistent_feature_repair
        if isinstance(head,GainHead): head.active=apply
        flow=static['flow_norm'].clone(); residual=static['image_residual'].clone()
        res,refined,flow_features=m.small_encoder(rgb_t=frame['img'],depth_t=r,
            flow_t=flow,raft_t=None,residuals_t=residual)
        if len(res)>4:
            res=[x.permute(0,3,1,2) for i,x in enumerate(res) if i in [2,5,8,11]]
        data=dict(image=frame['img'],image_original=frame['img'],validity_mask=static['validity_mask'],
            grid_sample=static['grid_sample'],flow_bwd=static['flow_norm'],image_residual=static['image_residual'],
            features_teacher=features,features_residual=res,flow_features=flow_features,flow_bwd_refined=refined)
        r_new,c_new,f_new=m.pixel_decoder.forward_fast(data,[dict(T=1,B=1)])
        self.counts['native_fast_calls']+=1
        if self.cfg['schedule']=='Auto' and not self.first and self.age==1:
            # Original overlap forward uses new KF features again; keep both calls.
            data['features_teacher']=features
            r_overlap,c_overlap,_=m.pixel_decoder.forward_fast(data,[dict(T=1,B=1)])
            self.counts['native_fast_calls']+=1
            c_new=c  # schedule[1]=1, confidence_teacher in upstream implementation.
        self.state=(r_new,c_new,f_new,rays)
        return refined,flow

    def advance(self,frame,index,static=None):
        cfg=self.cfg; auto=cfg['schedule']=='Auto'; initial=self.state is None
        scheduled=not auto and index%int(cfg['schedule'][1:])==0
        key=initial or scheduled; requested_age=0 if key else self.age
        repair=not key and cfg['family']=='P' and repair_due(self.age,cfg['frequency'])
        before_head=getattr(self.model.pixel_decoder.persistent_feature_repair,'calls',0)
        if key:
            self.keyframe(frame,index)
        else:
            require(self.previous is not None,'Missing previous RGB')
            # Preserve the historical A/O vs native P autocast flow boundary.
            if static is None:
                if cfg['family']=='P' or auto:
                    with torch.autocast('cuda',dtype=torch.float16):
                        static=transition(self.rt,self.model,self.previous,frame,self.shape,self.dis)
                else:
                    static=transition(self.rt,self.model,self.previous,frame,self.shape,self.dis)
            self.counts['flow_calls']+=1
            refined,flow=self.fast(frame,static,repair)
            if auto:
                from velodepth.models.decoder_velo import flow2grid_sample
                self.flow_history=self.flow_history*.5+flow.norm(dim=1).quantile(.95).item()
                mask=F.grid_sample(torch.ones_like(static['validity_mask']).float(),
                    flow2grid_sample(refined.clone(),mask=False),mode='nearest')<.5
                self.luma_history=self.luma_history*.5+(mask & static['validity_mask'].bool()).float().mean().item()
                restart=(self.flow_history>self.model.alpha_flow+self.model.beta_flow*(.9**(self.age-1)) or
                         self.luma_history>self.model.alpha_luma+self.model.beta_luma*(.9**(self.age-1)))
                if restart:
                    self.counts['rejected_candidates']+=1; self.first=False
                    self.keyframe(frame,index,auto_restart=True); key=True; requested_age=0
                else: self.age+=1
            else: self.age+=1
        self.counts['repair_head_calls']+=getattr(self.model.pixel_decoder.persistent_feature_repair,'calls',0)-before_head
        with torch.autocast('cuda',dtype=torch.float16):
            pred=output_depth(self.rt,self.model,self.state,self.shape)
            if self.debug: self.native_prediction=pred.detach()
            apply=not key and cfg['family'] in ('B2','O_multi','O_fast') and repair_due(requested_age,cfg['frequency'])
            if apply:
                before=versions(self.state) if self.debug else None
                if cfg['family']=='O_fast':
                    pred=self.adapter(pred,rgb_for_output(frame,self.shape),cfg['gain'])
                else:
                    r,c,f,rays=self.state
                    if cfg['family']=='O_multi' and cfg['gain']!=1.:
                        adapted=[x+cfg['gain']*block(x) for x,block in zip(f,self.adapter.blocks)]
                    else:
                        adapted=self.adapter(f)
                        if cfg['gain']!=1.: adapted=[x+cfg['gain']*(y-x) for x,y in zip(f,adapted)]
                    r_out=readout_radius(self.model,adapted)
                    pred=self.rt.gt2.make_z_depth(r_out,rays,self.shape['padded_shape'],self.shape['paddings'],self.model.interpolation_mode)
                    self.counts['additional_readout_passes']+=1
                self.counts['output_adapter_calls']+=1
                if self.debug: require(before==versions(self.state),'Output adapter modified recursive tensors')
        self.previous=frame
        return pred,dict(keyframe=key,age=requested_age,last_keyframe=self.last_key,
            output_adapter_applied=apply,repair_requested=repair)

    def step(self,index,record,calibration=None):
        frame=self.rt.exp1.load_frame(record,self.shape,torch.device('cuda'),need_gt=False)
        static=None
        if self.state is not None and self.cfg['schedule']!='Auto' and self.cfg['family']!='P' and index%int(self.cfg['schedule'][1:])!=0:
            static=transition(self.rt,self.model,self.previous,frame,self.shape,self.dis)
        with torch.autocast('cuda',dtype=torch.float16):
            pred,meta=self.advance(frame,index,static)
        if calibration is not None:
            pred=(pred.float()*calibration['scale']+calibration['bias_m']).clamp_min(FLOOR)
        return pred.detach().float().cpu().numpy(),meta
