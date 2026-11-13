"""Separate differentiable fast path. Never call diagnostic no_grad wrappers in training."""
from y05_common import *
import gc,time

def detached_state(state,device=None):
    def copy(x):
        x=x.detach().clone()
        return x.to(device) if device is not None else x
    r,c,f,rays=state;return copy(r),copy(c),[copy(x) for x in f],copy(rays)

def tsave(torch,path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    torch.save(obj,t);t.replace(p)

def valid_state(torch,state):return all(bool(torch.isfinite(x).all().item()) for x in (state[0],state[1],*state[2],state[3]))

def depth_loss(torch,pred,gt):
    require(gt is not None,'Loss requested without GT')
    mask=(gt>0)&torch.isfinite(gt);count=int(mask.sum().item());require(count>0,'Empty GT-only mask')
    p=pred.float()[mask];g=gt.float()[mask]
    require(bool((torch.isfinite(p)&(p>0)).all().item()),'Invalid prediction on fixed GT mask; never shrink mask')
    return ((p-g).abs()/g).mean(),count

def numpy_metrics(np,pred,frame):
    gt=frame['gt_np']
    if gt is None:
        require(bool(np.isfinite(pred).all() and (pred>0).all()),'Invalid prediction on unscored RGB')
        return dict(AbsRel='',RMSE='',delta1='',valid_pixels=0)
    mask=(gt>0)&np.isfinite(gt);require(mask.any(),'Empty GT mask')
    require(bool((np.isfinite(pred[mask])&(pred[mask]>0)).all()),'Invalid prediction on fixed GT mask')
    p=pred[mask].astype('float64');g=gt[mask].astype('float64')
    return dict(AbsRel=float(np.mean(abs(p-g)/g)),RMSE=float(np.sqrt(np.mean((p-g)**2))),delta1=float(np.mean(np.maximum(p/g,g/p)<1.25)),valid_pixels=int(mask.sum()))

class Engine:
    def __init__(self,ctx,seq):
        self.ctx=ctx;self.rt=ctx.rt;torch=self.rt.torch
        self.model=self.rt.gt2.load_model(Path(ctx.args.snapshot),'pre_readout',torch.device('cuda'))
        head=self.model.pixel_decoder.persistent_feature_repair;head.load_state_dict(ctx.head_state,strict=True)
        require(sum(p.numel() for p in head.parameters())==170496,'Wrong P architecture')
        class Gate(torch.nn.Module):
            def __init__(self,h):
                super().__init__();self.head=h;self.active=False;self.gain=1.;self.capture=False;self.delta=None;self.calls=0
            def forward(self,features):
                self.delta=None
                if not self.active:return features
                self.calls+=1
                if self.gain==1.:out=self.head(features)
                else:out=[x+self.gain*block(x) for x,block in zip(features,self.head.blocks)]
                if self.capture:
                    numerator=sum((y.float()-x.float()).square().sum() for x,y in zip(features,out))
                    denominator=sum(x.detach().float().square().sum() for x in features).clamp_min(1e-12)
                    self.delta=numerator/denominator
                return out
        self.gate=Gate(head);self.head=head;self.model.pixel_decoder.persistent_feature_repair=self.gate
        self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        for p in head.parameters():p.requires_grad_(True)
        self.dis=self.rt.cv2.DISOpticalFlow_create(self.rt.cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.seq=None;self.configure(seq)

    def configure(self,seq):
        if self.seq==seq:return
        self.seq=seq;self.shape=self.rt.exp1.shape_info_for(self.model,Path(self.ctx.manifests[seq][0]['rgb_path']),self.rt.torch.device('cuda'))
        # Sets native geometry caches; no graph is allowed to survive a video switch.
        frame=self.load(0,False);self.rt.exp1.keyframe_state(self.model,frame,self.shape)

    def load(self,index,gt=False):
        r=self.ctx.manifests[self.seq][index]
        return self.rt.exp1.load_frame(r,self.shape,self.rt.torch.device('cuda'),need_gt=gt and matched(r))

    def static(self,previous,current,origin):
        torch=self.rt.torch
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=origin!='A'):
            return self.rt.exp1.transition_static(self.model,previous,current,self.shape,self.dis)

    def read_depth(self,state):
        with self.rt.torch.autocast('cuda',dtype=self.rt.torch.float16):
            return self.rt.gt2.make_z_depth(state[0],state[3],self.shape['padded_shape'],self.shape['paddings'],self.model.interpolation_mode)

    def tensor_step(self,state,previous,current,k,origin,apply=False,static=None,gain=1.,checkpoint=False,capture=False):
        torch=self.rt.torch;i=current['index']
        if i%k==0:
            rays,r,c,f=self.rt.exp1.keyframe_state(self.model,current,self.shape)
            return (r,c,f,rays),r.new_zeros((),dtype=torch.float32)
        require(state is not None,'Missing parent state')
        if static is None:static=self.static(previous,current,origin)
        # All closure settings are immutable per step. Recompute must restore its own gate,
        # not the last forward step's disabled/enabled value.
        active=bool(apply);scale=float(gain);cap=bool(capture)
        def transition(r,c,*features):
            prior=(self.gate.active,self.gate.gain,self.gate.capture)
            self.gate.active,self.gate.gain,self.gate.capture=active,scale,cap
            try:
                with torch.autocast('cuda',dtype=torch.float16):
                    rr,cc,ff=self.rt.gt2.fast_step(self.model,{'frames':[previous,current],'fast_static':[static]},1,r,list(features))
                    delta=self.gate.delta if cap and active else rr.new_zeros((),dtype=torch.float32)
                require(delta is not None,'Missing anchor residual graph')
                return (rr,cc,*ff,delta)
            finally:self.gate.active,self.gate.gain,self.gate.capture=prior
        tensors=(state[0],state[1],*state[2])
        if checkpoint:
            from torch.utils.checkpoint import checkpoint as cp
            result=cp(transition,*tensors,use_reentrant=False,preserve_rng_state=True)
        else:result=transition(*tensors)
        return (result[0],result[1],list(result[2:6]),state[3]),result[6]

    def eval_step(self,state,previous,current,k,origin,apply=False,gain=1.,static=None):
        with self.rt.torch.no_grad():
            result,_=self.tensor_step(state,previous,current,k,origin,apply,static,gain)
            require(valid_state(self.rt.torch,result),'Nonfinite recurrent state')
            depth=self.read_depth(result).detach().float().cpu().numpy()
        return result,depth

    def context(self,case,horizon=None):
        self.configure(case['sequence']);h=case['horizon'] if horizon is None else horizon
        frames={j:self.load(case['anchor']+j,j>=0) for j in range(-1,h+1)}
        statics={j:self.static(frames[j-1],frames[j],case['origin']) for j in range(h+1) if frames[j]['index']%case['K']!=0}
        return frames,statics

    def close(self):
        del self.model,self.head,self.gate;gc.collect();self.rt.torch.cuda.empty_cache()

def ensure_bank(ctx,engine,cases,label):
    """Replay from RGB0, cache parents only. Initial P is frozen, never a trained arm."""
    ctx.bank_objects=getattr(ctx,'bank_objects',{});paths={}
    for seq in SEQS:
        for origin in ('A','S8'):
            selected=[c for c in cases if c['sequence']==seq and c['origin']==origin]
            if not selected:continue
            directory=ctx.out/'cache'/label/f'{seq}_{origin}';path=directory/'parents.pt';meta_path=directory/'complete.json'
            wanted=sorted({c['snapshot_index'] for c in selected});paths[seq,origin]=path
            if meta_path.exists():
                meta=jread(meta_path);require(meta['signature']==ctx.signature and meta['required']==wanted and sha(path)==meta['sha256'],'Parent cache identity changed')
                continue
            engine.configure(seq);engine.head.load_state_dict(ctx.head_state,strict=True)
            snapshots={};state=prev=None;start=time.perf_counter();audits=[]
            for i in range(max(wanted)+1):
                score=i<256 or i>=TAIL[seq];frame=engine.load(i,score)
                state,pred=engine.eval_step(state,prev,frame,32,origin,apply=origin=='S8' and pulse_due(i,32))
                if origin=='A' and score and frame['gt_np'] is not None:
                    mm=numpy_metrics(ctx.rt.np,pred,frame);ref=ctx.refs['A_K32_g0_f1/raw',seq,i]
                    err=max(abs(mm[k]-float(ref[k])) for k in ('AbsRel','RMSE','delta1'))
                    require(err<=2e-5 and mm['valid_pixels']==int(ref['valid_pixels']),'Native parent replay differs from Y09')
                    audits.append(err)
                if i in wanted:snapshots[i]=detached_state(state,'cpu')
                prev=frame
                if i==0 or (i+1)%200==0 or i==max(wanted):print(f'[CACHE {label}] {seq}/{origin} {i+1}/{max(wanted)+1}',flush=True)
            require(sorted(snapshots)==wanted,'Parent bank incomplete')
            tsave(ctx.rt.torch,path,dict(signature=ctx.signature,snapshots=snapshots))
            jwrite(meta_path,dict(signature=ctx.signature,required=wanted,sha256=sha(path),processed_RGB=max(wanted)+1,wall_sec=time.perf_counter()-start,n_native_reference_checks=len(audits),max_native_reference_error=max(audits) if audits else None))
    summaries=[]
    for path in paths.values():summaries.append(dict(cache=str(path),**jread(path.with_name('complete.json'))))
    jwrite(ctx.out/f'cache_{label}_summary.json',dict(signature=ctx.signature,caches=summaries,total_completed_build_wall_sec=sum(x['wall_sec'] for x in summaries),shared_cost='charged once; all training arms share the frozen parent bank'))
    return paths

def parent_for(ctx,paths,case):
    path=paths[case['sequence'],case['origin']];key=str(path)
    if key not in ctx.bank_objects:
        require(sha(path)==jread(path.with_name('complete.json'))['sha256'],'Parent cache changed')
        obj=ctx.rt.torch.load(path,map_location='cpu',weights_only=False)
        require(obj['signature']==ctx.signature,'Foreign parent cache');ctx.bank_objects[key]=obj['snapshots']
    return detached_state(ctx.bank_objects[key][case['snapshot_index']],'cuda')

def pulse_forward(ctx,engine,case,parent,frames,statics,kind,checkpoint=True,detach_after_anchor=False):
    """Reference is never repaired. Returned future losses retain the state-input graph."""
    torch=ctx.rt.torch;state=detached_state(parent,parent[0].device);ref=detached_state(parent,parent[0].device)
    current=None;future=[];reference=[];n_gt=0;delta=None
    for h in range(case['horizon']+1):
        frame=frames[h];apply=h==0 or kind=='P_closed_loop'
        state,reg=engine.tensor_step(state,frames[h-1],frame,case['K'],case['origin'],apply,statics.get(h),checkpoint=checkpoint,capture=h==0)
        require(valid_state(torch,state),'Nonfinite training state, including on missing-GT RGB')
        if h==0:
            delta=reg
            if detach_after_anchor:state=detached_state(state)
        if frame['gt'] is not None:
            pred=engine.read_depth(state);loss,_=depth_loss(torch,pred,frame['gt'])
            if h==0:current=loss
            else:future.append(loss);n_gt+=1
        with torch.no_grad():
            ref,_=engine.tensor_step(ref,frames[h-1],frame,case['K'],case['origin'],False,statics.get(h))
            if h>0 and frame['gt'] is not None:
                value,_=depth_loss(torch,engine.read_depth(ref),frame['gt']);reference.append(value.detach())
    require(current is not None and future,'No current/future supervision')
    a=torch.stack(future);b=torch.stack(reference)
    lf=a.mean();lh=torch.relu(a-b-ctx.plan['settings']['harm_margin_AbsRel']).mean()
    cfg=ctx.plan['settings'];base=current if kind=='C_current' else lf
    objective=base+(cfg['lambda_harm']*lh if kind=='FH_future_harm' else 0.)+cfg['mu_anchor_delta']*delta
    return dict(objective=objective,current=current,future=lf,harm=lh,delta=delta,future_losses=future,reference=b.mean(),future_GT=n_gt,current_GT=1)
