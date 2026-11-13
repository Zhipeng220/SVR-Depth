"""V5 same-parent teacher prediction preservation, fixed K32/TBPTT4."""
import numpy as np
import torch
import torch.nn.functional as F
import gt4a2_c3_closed_loop as c3
import gt4a2_exp1_train_eval as exp1
import gt2_fixed_clip_repair_train as gt2

LAMBDA=82.71035398147102
def train_cycle(model,head,optimizer,scaler,cycle,device):
    if len(cycle['frames'])!=32 or len(cycle['statics'])!=31:raise RuntimeError('Cycle length mismatch')
    n=cycle['n_supervised'];assert n>0
    radius,confidence,features=c3.move_state(cycle['keyframe_state'],device);rays=cycle['keyframe_state']['rays_map'].to(device)
    pads=cycle['paddings'];pad_shape=cycle['padded_shape'];shape={'validity_mask':torch.ones((1,1,*cycle['network_shape']),device=device,dtype=torch.bool)}
    prev=c3.to_gpu_frame(cycle['frames'][0],device);optimizer.zero_grad(set_to_none=True);depth_values=[];pres_values=[];segments=0;lengths=[];consumed=0
    for begin in range(1,32,4):
        end=min(31,begin+3);terms=[];hold=[prev]
        for offset in range(begin,end+1):
            curr=c3.to_gpu_frame(cycle['frames'][offset],device);static=c3.to_gpu_static(cycle['statics'][offset-1],device,shape)
            model.pixel_decoder.feature_repair_mode='off'
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                r0,_,_=exp1.one_fast_step(model,prev,curr,static,radius.detach().clone(),confidence.detach().clone(),[x.detach().clone() for x in features]);d0=gt2.make_z_depth(r0,rays,pad_shape,pads,model.interpolation_mode).float()
            model.pixel_decoder.feature_repair_mode='pre_readout'
            with torch.autocast('cuda',dtype=torch.float16):
                radius,confidence,features=exp1.one_fast_step(model,prev,curr,static,radius,confidence,features);d1=gt2.make_z_depth(radius,rays,pad_shape,pads,model.interpolation_mode).float()
                if not bool((torch.isfinite(d0)&torch.isfinite(d1)&(d0>0)&(d1>0)).all()):raise RuntimeError('Invalid preserve depth')
                preserve=F.smooth_l1_loss(d1.log(),d0.detach().log(),reduction='mean',beta=1.0);terms.append(LAMBDA*preserve/31.0);pres_values.append(float(preserve.detach()))
                if curr['gt'] is not None:
                    loss,_=gt2.masked_log_depth_loss(d1,curr['gt']);terms.append(loss/n);depth_values.append(float(loss.detach()));consumed+=1
            hold.append((curr,static));prev=curr
        scaler.scale(torch.stack(terms).sum()).backward();segments+=1
        radius,confidence,features=c3.detach_state(radius,confidence,features);prev={k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in prev.items()};lengths.append(end-begin+1);del terms,hold
    if consumed!=n:raise RuntimeError('GT coverage mismatch')
    scaler.unscale_(optimizer);grads=[p.grad for p in head.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):raise RuntimeError('Invalid gradients')
    nonzero=sum(bool(torch.count_nonzero(g).item()) for g in grads)
    if not nonzero:raise RuntimeError('Zero gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),max_norm=1.0));old=float(scaler.get_scale());scaler.step(optimizer);scaler.update()
    if float(scaler.get_scale())<old:raise RuntimeError('Gradient scaler skipped update')
    return {'mean_log_depth_loss_before_update':float(np.mean(depth_values)),'mean_preserve_loss':float(np.mean(pres_values)),'preserve_lambda':LAMBDA,'supervised_propagation_frames':consumed,'preserve_frames':len(pres_values),'gradient_segments':segments,'gradient_nonzero_tensors':nonzero,'gradient_norm_before_clip':norm,'grad_scaler_scale':float(scaler.get_scale()),'segment_lengths':','.join(map(str,lengths))}
