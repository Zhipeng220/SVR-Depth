"""Native-warp direction probe. Torch backend at runtime, NumPy backend in tests.

No learned alpha/beta controller is claimed. Probes select (alpha,beta)=(1,0)
or (0,1), plus scalar and scrambled-direction controls on a frozen C head.
"""
import math

class NumpyOps:
    def __init__(self,np):self.np=np
    def float(self,x):return x.astype('float32')
    def cast(self,x,like):return x.astype(like.dtype)
    def sum_channels(self,x):return self.np.sum(x,axis=1,keepdims=True)
    def sqrt(self,x):return self.np.sqrt(x)
    def clamp_min(self,x,value):return self.np.maximum(x,value)
    def where(self,c,a,b):return self.np.where(c,a,b)
    def scramble(self,x):
        signs=(1-2*(self.np.arange(x.shape[1])%2)).reshape(1,-1,1,1)
        return self.np.flip(x,axis=1)*signs.astype('float32')
    def total(self,x):return float(self.np.sum(x))
    def mean(self,x):return float(self.np.mean(x))
    def finite(self,x):return bool(self.np.isfinite(x).all())

class TorchOps:
    def __init__(self,torch):self.torch=torch
    def float(self,x):return x.float()
    def cast(self,x,like):return x.to(dtype=like.dtype)
    def sum_channels(self,x):return x.sum(dim=1,keepdim=True)
    def sqrt(self,x):return x.sqrt()
    def clamp_min(self,x,value):return x.clamp_min(value)
    def where(self,c,a,b):return self.torch.where(c,a,b)
    def scramble(self,x):
        signs=(1-2*(self.torch.arange(x.shape[1],device=x.device)%2)).reshape(1,-1,1,1)
        return self.torch.flip(x,dims=[1])*signs.to(dtype=x.dtype)
    def total(self,x):return float(x.float().sum().item())
    def mean(self,x):return float(x.float().mean().item())
    def finite(self,x):return bool(self.torch.isfinite(x).all().item())

def components(delta,innovation,features,ops):
    # Actual channel-space projection at every spatial position and feature level.
    u2=ops.sum_channels(innovation*innovation)
    f2=ops.sum_channels(features*features)
    valid=u2>1e-12*(1+f2)
    dot=ops.sum_channels(delta*innovation)
    parallel=ops.where(valid,dot/ops.clamp_min(u2,1e-30)*innovation,delta*0)
    orthogonal=delta-parallel
    return parallel,orthogonal,valid

def match_norm(vector,target,delta,ops):
    v2=ops.sum_channels(vector*vector);d2=ops.sum_channels(delta*delta)
    direction=ops.where(v2>1e-20,vector/ops.sqrt(ops.clamp_min(v2,1e-30)),delta/ops.sqrt(ops.clamp_min(d2,1e-30)))
    return direction*ops.sqrt(ops.clamp_min(target,0))

def edit(features,native_output,warped_previous,mode,ops):
    f=ops.float(features);d=ops.float(native_output)-f;u=f-ops.float(warped_previous)
    p,q,valid=components(d,u,f,ops)
    p2=ops.sum_channels(p*p);q2=ops.sum_channels(q*q);d2=ops.sum_channels(d*d)
    if mode=='full':result=native_output  # exact native identity, no round trip
    else:
        if mode=='parallel':edited=p
        elif mode=='orthogonal':edited=q
        elif mode=='norm_parallel':edited=match_norm(d,p2,d,ops)
        elif mode=='norm_orthogonal':edited=match_norm(d,q2,d,ops)
        elif mode in ('scrambled_parallel','scrambled_orthogonal'):
            random_p,random_q,_=components(d,ops.scramble(u),f,ops)
            edited=match_norm(random_p if mode=='scrambled_parallel' else random_q,p2 if mode=='scrambled_parallel' else q2,d,ops)
        else:raise RuntimeError('Unknown direction mode '+mode)
        result=features+ops.cast(edited,features)
    realized=ops.float(result)-f
    u2=ops.sum_channels(u*u);dot=ops.sum_channels(d*u)
    cosine=dot/ops.sqrt(ops.clamp_min(d2*u2,1e-30))
    stats=dict(original_energy=ops.total(d2),parallel_energy=ops.total(p2),orthogonal_energy=ops.total(q2),realized_energy=ops.total(ops.sum_channels(realized*realized)),innovation_energy=ops.total(u2),valid_innovation_fraction=ops.mean(valid),signed_cosine_mean=ops.mean(cosine),parallel_energy_fraction=ops.total(p2)/max(ops.total(d2),1e-30),target_energy=ops.total(p2 if 'parallel' in mode and 'orthogonal' not in mode else q2 if 'orthogonal' in mode else d2))
    if not ops.finite(result) or not all(math.isfinite(v) for v in stats.values()):raise RuntimeError('Nonfinite direction edit/statistic')
    return result,stats

class DirectionalEditor:
    """Capture exact native init_warped using temporary fusion pre-hooks."""
    def __init__(self,engine,ops=None):
        self.engine=engine;self.ops=ops or TorchOps(engine.rt.torch)
        self.original_forward=engine.head.forward;self.handles=[];self.warped={};self.stats=[]
        self.enabled=False;self.mode='full'
        fusion=engine.model.pixel_decoder.radial_module.fusion
        if len(fusion)!=4:raise RuntimeError('Expected exactly four reviewed native fusion levels')
        for i,layer in enumerate(fusion):
            def hook(module,args,index=i):
                if self.enabled:
                    x=args[0]
                    self.warped[index]=x.detach().clone() if hasattr(x,'detach') else x.copy()
            self.handles.append(layer.register_forward_pre_hook(hook))
        def forward(features):
            output=self.original_forward(features)
            if not self.enabled:return output
            if len(features)!=4 or set(self.warped)!=set(range(4)):raise RuntimeError('Native warped features not captured before pre_readout repair')
            result=[];self.stats=[]
            for i,(f,y) in enumerate(zip(features,output)):
                if f.shape!=self.warped[i].shape:raise RuntimeError('Native/warped feature coordinates or channels differ')
                edited,stat=edit(f,y,self.warped[i],self.mode,self.ops);result.append(edited)
                self.stats.append(dict(level=i,mode=self.mode,**stat))
            return result
        engine.head.forward=forward
    def configure(self,mode,enabled):
        self.mode=mode;self.enabled=enabled;self.warped={}
        if enabled:self.stats=[]
    def close(self):
        self.engine.head.forward=self.original_forward
        for handle in self.handles:handle.remove()
        self.handles=[];self.warped={}
