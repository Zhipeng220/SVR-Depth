"""Frozen C* carrier and transient readout. No trainable state commit or future rollout."""
from .common import *
from types import SimpleNamespace
import gc,sys

def prepare(args,out,audit):
    policy=read(HERE/'policy.json');identity=read(HERE/'identity.json');y05r,prior,cases,coverage,source=audit
    sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1')]
    import y05_common as old
    ctx=old.preflight(SimpleNamespace(**prior['paths'],smoke_only=False));oldplan=read(Path(prior['paths']['y05_dir'])/'protocol.json')
    require(ctx.plan==oldplan,'Original Y05 protocol no longer reproduces; do not modify old hashes/manifests')
    t=ctx.rt.torch;require(t.cuda.is_available(),'Actual VeloDepth requires CUDA');t.set_num_threads(4)
    cp=y05r/'training/C_star/final.pt';obj=t.load(cp,map_location='cpu',weights_only=False)
    require(obj['signature']==sha(y05r/'protocol.json') and obj['experiment']=='Y05-R' and obj['kind']=='C_star','C* source signature differs')
    require(ctx.rt.exp1.state_sha256(obj['state'])==obj['head_sha256']==identity['Cstar_tensor_sha256'],'C* tensor identity differs')
    ctx.head_state=obj['state'];ctx.policy=policy;ctx.args=SimpleNamespace(**prior['paths']);ctx.out=out;ctx.y05r=y05r;ctx.cstar_file=cp
    ctx.old_sequences=policy['old_sequences'];ctx.manifests=dict(ctx.manifests)
    runtime_cases,runtime_coverage=coverage_plan(ctx.manifests,policy);require(runtime_cases==cases and runtime_coverage==coverage,'CPU coverage and audited runtime manifests differ')
    inputs=dict(oldplan['input_sha256']);inputs.update(source['input_sha256'])
    inputs.update({str(p.resolve()):sha(p) for p in [y05r/'summary.json',HERE/'policy.json',HERE/'identity.json',HERE/'installed_manifest.json',*HERE.glob('*.py')]})
    inputs.update({str((HERE/k).resolve()):v for k,v in read(HERE/'installed_manifest.json').items()})
    if not args.smoke_only:
        dp=Path(args.dataset_protocol).resolve();require(dp.is_file(),'Existing Y11 dataset_protocol.json required; set N14A_DATASET_PROTOCOL. No data will be downloaded.')
        data=read(dp);require(list(data['manifests'])==policy['development_sequences'],'Do not substitute Y11 development sequences')
        verify_hashes(data['input_sha256']);inputs.update(data['input_sha256']);inputs[str(dp)]=sha(dp)
        for p in data.get('expected_missing_depth',[]):require(not Path(p).exists(),'GT availability changed after Y11 data freeze')
        for s,m in data['manifests'].items():
            rr=ctx.rt.exp1.read_manifest(Path(m['path']),training=False);require(len(rr)==m['RGB'],'Y11 manifest coverage changed');ctx.manifests[s]=rr
    op=Path(ctx.args.y04_dir)/'training/O_multi/Y04_O_multi_final_step1000.pt';ctx.omulti_state=t.load(op,map_location='cpu',weights_only=False)['adapter_state_dict'];inputs[str(op.resolve())]=sha(op)
    updates=2 if args.smoke_only else policy['train_updates'];order=order_for(cases,updates,policy['seed'],policy['old_sequences'])
    ctx.plan=dict(experiment='N14-A',version=policy['version'],partial_smoke=args.smoke_only,policy=policy,y05r_dir=str(y05r),
        dataset_protocol=str(Path(args.dataset_protocol).resolve()),Cstar_tensor_sha256=obj['head_sha256'],Cstar_file_sha256=sha(cp),
        train_cases=cases,coverage=coverage,sample_order=order,sample_order_sha256=canonical(order),updates=updates,
        input_sha256=inputs,environment=ctx.plan['environment'],
        data_split='old three RGB0..255 current-frame training; old tails and already-viewed Y11 five recordings DEVELOPMENT ONLY',
        state_contract='exact frozen C* K32/S8 trajectory; micro-readout never writes radius, confidence, features or rays',
        supervision='current frame GT AbsRel + 1e-4 normalized feature residual energy; no future GT, teacher, function distillation or gate',
        cache_dtype='preserve original native tensor dtype; never silently quantize FP32 state',
        prediction_retention='per-frame metrics/SHA256 and five fixed float32 probe depths; all scored depths optional',save_depths=args.save_depths,
        method_pass=False,final_blind_test=False,auto_N14B=False)
    return ctx

def save_torch(t,p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');t.save(obj,q);q.replace(p)
def loss(t,p,g):
    mask=(g>0)&t.isfinite(g);require(bool(mask.any().item()),'Empty GT-only mask')
    require(bool((t.isfinite(p[mask])&(p[mask]>0)).all().item()),'Invalid prediction on fixed GT mask')
    return ((p.float()[mask]-g.float()[mask]).abs()/g.float()[mask]).mean()
def metric(np,p,g):
    if g is None:
        require(bool(np.isfinite(p).all() and (p>0).all()),'Invalid unscored RGB prediction');return dict(AbsRel='',RMSE='',delta1='',valid_pixels=0,mask_sha256='')
    mask=(g>0)&np.isfinite(g);require(bool(mask.any()),'Empty GT mask')
    require(bool((np.isfinite(p[mask])&(p[mask]>0)).all()),'Invalid prediction; GT mask cannot shrink')
    a=p[mask].astype('float64');b=g[mask].astype('float64')
    return dict(AbsRel=float(np.mean(abs(a-b)/b)),RMSE=float(np.sqrt(np.mean((a-b)**2))),delta1=float(np.mean(np.maximum(a/b,b/a)<1.25)),valid_pixels=int(mask.sum()),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
def finite_state(t,state):return all(bool(t.isfinite(x).all().item()) for x in state_fields(state).values())

class FrozenCarrier:
    def __init__(self,ctx,seq):
        from y05_engine import Engine
        self.ctx=ctx;self.t=ctx.rt.torch;self.base=Engine(ctx,seq);self.seq=seq;self.state=self.previous=None
        for p in self.base.model.parameters():p.requires_grad_(False);p.grad=None
        self.base.model.eval();self.base.head.eval()
    @property
    def shape(self):return self.base.shape
    def configure(self,seq):
        if seq!=self.seq:self.base.configure(seq);self.seq=seq;self.reset()
    def reset(self):self.state=self.previous=None
    def load(self,i,gt=False):return self.base.load(i,gt)
    def causal(self,static):
        require(static is not None,'Non-pulse micro-readout requires existing current/previous flow')
        t=self.t;flow=static['flow_norm'].detach();res=static['image_residual'].detach()
        require(flow.ndim==4 and flow.shape[1]==2 and res.ndim==4 and res.shape[1]==1,'Audited Z_fast must be 2 flow + 1 residual channels; no age channel or implicit reshape')
        res=t.nn.functional.interpolate(res.float(),size=flow.shape[-2:],mode='bilinear',align_corners=False)
        return t.cat([flow.float(),res],1).detach()
    def advance(self,frame,need_native_depth=True):
        i=frame['index'];static=None
        with self.t.no_grad():
            if i%32:static=self.base.static(self.previous,frame,'S8')
            self.state,_=self.base.tensor_step(self.state,self.previous,frame,32,'S8',apply=active(i),static=static,checkpoint=False)
            native=self.base.read_depth(self.state) if need_native_depth else None
        self.previous=frame;return native,static
    def read_features(self,features,rays):
        from y08y09_models import readout_radius
        with self.t.autocast('cuda',dtype=self.t.float16):
            radius=readout_radius(self.base.model,features)
            return self.ctx.rt.gt2.make_z_depth(radius,rays,self.shape['padded_shape'],self.shape['paddings'],self.base.model.interpolation_mode)
    def micro(self,module,native,static,index,compute_regularizer=False):
        if not micro_due(index):return native,native.new_zeros((),dtype=self.t.float32)
        # This function deliberately has no assignment to any persistent state field.
        features=[x.detach() for x in self.state[2]];causal=self.causal(static)
        with self.t.autocast('cuda',dtype=self.t.float16):read,reg=module(features,causal,phase(index),compute_regularizer);pred=self.read_features(read,self.state[3].detach())
        return pred,reg
    def close(self):
        self.state=self.previous=None;self.base.close();del self.base;gc.collect();self.t.cuda.empty_cache()

def micro_from_cache(ctx,carrier,module,obj):
    t=ctx.rt.torch;carrier.configure(obj['sequence'])
    features=[x.detach().to('cuda') for x in obj['features']];rays=obj['rays'].detach().to('cuda');causal=obj['causal'].detach().to('cuda');gt=obj['gt'].detach().to('cuda')
    require(all(not x.requires_grad for x in [*features,rays,causal,gt]),'Cached frozen inputs unexpectedly carry a graph')
    with t.autocast('cuda',dtype=t.float16):read,reg=module(features,causal,obj['phase']);pred=carrier.read_features(read,rays)
    current=loss(t,pred,gt);objective=current+ctx.policy['lambda_residual']*reg
    return dict(objective=objective,current=current,regularizer=reg)
