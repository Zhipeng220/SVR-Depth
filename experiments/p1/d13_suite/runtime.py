"""Adapter to reviewed Y05 Engine and VeloDepth, with genuinely differentiable H1 state transport."""
from .common import *
from types import SimpleNamespace
import gc,sys

def prepare(args,out):
    policy=read(HERE/'policy.json');identity=read(HERE/'identity.json')
    y05r=locate_y05r(args.y05r_dir,identity);prior=read(y05r/'protocol.json')
    verify_hashes({str(ROOT/p):d for p,d in identity['original_source_sha256'].items()})
    require(sha(ROOT/'velodepth/models/decoder_velo.py')==identity['decoder_sha256'],'Reviewed decoder source differs')
    sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1')]
    import y05_common as old
    ctx=old.preflight(SimpleNamespace(**prior['paths'],smoke_only=False))
    oldprotocol=read(Path(prior['paths']['y05_dir'])/'protocol.json')
    require(ctx.plan==oldprotocol,'Original Y05 protocol no longer reproduces; do not edit old manifests/hashes')
    t=ctx.rt.torch;require(t.cuda.is_available(),'Real training requires CUDA');t.set_num_threads(4)
    cp=y05r/'training/C_star/final.pt';obj=t.load(cp,map_location='cpu',weights_only=False)
    require(obj['signature']==sha(y05r/'protocol.json') and obj['experiment']=='Y05-R' and obj['kind']=='C_star','C* source identity differs')
    require(ctx.rt.exp1.state_sha256(obj['state'])==obj['head_sha256']==identity['Cstar_tensor_sha256'],'C* tensor SHA differs')
    ctx.head_state=obj['state'];ctx.policy=policy;ctx.args=SimpleNamespace(**prior['paths']);ctx.out=out
    ctx.cstar_file=cp;ctx.y05r=y05r
    inputs=dict(oldprotocol['input_sha256'])
    inputs.update({str(p.resolve()):sha(p) for p in [cp,y05r/'protocol.json',y05r/'summary.json',HERE/'policy.json',HERE/'identity.json',*HERE.glob('*.py')]})
    ctx.old_sequences=list(policy['old_sequences']);ctx.manifests=dict(ctx.manifests)
    if not args.smoke_only:
        dp=Path(args.dataset_protocol).resolve()
        require(dp.is_file(),'Existing Y11 dataset_protocol.json required; set --dataset-protocol. This package NEVER downloads data.')
        data=read(dp);require(list(data['manifests'])==policy['development_sequences'],'Expected existing five Y11 recordings; do not substitute sequences')
        verify_hashes(data['input_sha256']);inputs.update(data['input_sha256']);inputs[str(dp)]=sha(dp)
        for p in data.get('expected_missing_depth',[]):require(not Path(p).exists(),'GT availability changed after Y11 freeze: '+p)
        for s,m in data['manifests'].items():
            rr=ctx.rt.exp1.read_manifest(Path(m['path']),training=False)
            require(len(rr)==m['RGB'],'Y11 RGB manifest differs');ctx.manifests[s]=rr
    cases=cases_for(ctx.manifests,policy);updates=2 if args.smoke_only else policy['train_updates']
    order=order_for(cases,updates,policy['seed'],policy['old_sequences'])
    o_path=Path(ctx.args.y04_dir)/'training/O_multi/Y04_O_multi_final_step1000.pt'
    o=t.load(o_path,map_location='cpu',weights_only=False);ctx.omulti_state=o['adapter_state_dict']
    inputs[str(o_path.resolve())]=sha(o_path)
    plan=dict(experiment=policy['experiment'],version=policy['version'],policy=policy,partial_smoke=args.smoke_only,
        y05r_dir=str(y05r),dataset_protocol=str(Path(args.dataset_protocol).resolve()),
        source_y05r_signature=sha(y05r/'protocol.json'),Cstar_tensor_sha256=obj['head_sha256'],
        train_cases=cases,sample_order=order,sample_order_sha256=canonical(order),updates=updates,
        input_sha256=inputs,environment=ctx.plan['environment'],
        data_split='old prefixes RGB0..255 training only; old tails plus already-viewed Y11 five recordings DEVELOPMENT ONLY',
        state_contract='F-only extra commit; persistent radius/confidence/rays are native frozen C* fields; readout radius is transient and never reused',
        limitations=['one new seed','off-policy common frozen C* parent bank','no final blind test','not equal compute',
            'Envelope is an RMS soft shrinkage, not a spectral/gradient/manifold safety guarantee',
            'D gate uses frame-level counterfactual H1 targets, not per-pixel certified risk'])
    ctx.plan=plan
    return ctx

def detach(t,state,device=None):
    def cp(x):return x.detach().clone().to(device or x.device)
    r,c,f,rays=state;return cp(r),cp(c),[cp(x) for x in f],cp(rays)
def save_torch(t,path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);q=path.with_name(path.name+'.tmp');t.save(obj,q);q.replace(path)
def finite_state(t,state):return all(bool(t.isfinite(x).all().item()) for x in (state[0],state[1],*state[2],state[3]))
def loss(t,p,g):
    require(g is not None,'GT missing in selected H1 pair');m=(g>0)&t.isfinite(g);require(bool(m.any().item()),'Empty GT mask')
    require(bool((t.isfinite(p[m])&(p[m]>0)).all().item()),'Invalid prediction on fixed GT mask')
    return ((p.float()[m]-g.float()[m]).abs()/g.float()[m]).mean()
def metric(np,p,g):
    if g is None:
        require(bool(np.isfinite(p).all() and (p>0).all()),'Invalid unscored RGB prediction');return dict(AbsRel='',RMSE='',delta1='',valid_pixels=0,mask_sha256='')
    m=(g>0)&np.isfinite(g);require(bool(m.any()),'Empty GT mask')
    require(bool((np.isfinite(p[m])&(p[m]>0)).all()),'Invalid prediction; GT mask cannot shrink')
    a=p[m].astype('float64');b=g[m].astype('float64')
    return dict(AbsRel=float(np.mean(abs(a-b)/b)),RMSE=float(np.sqrt(np.mean((a-b)**2))),delta1=float(np.mean(np.maximum(a/b,b/a)<1.25)),
        valid_pixels=int(m.sum()),mask_sha256=hashlib.sha256(m.tobytes()).hexdigest())

class AdaptedEngine:
    def __init__(self,ctx,seq,arm=None):
        from y05_engine import Engine
        from .models import ResidualPaths
        self.ctx=ctx;self.t=ctx.rt.torch;self.base=Engine(ctx,seq);self.arm=arm;self.seq=seq
        self.paths=ResidualPaths(ctx.policy,arm).cuda() if arm else None
        t=self.t;paths=self.paths;cstar=self.base.head
        class Hook(t.nn.Module):
            def __init__(self):
                super().__init__();self.cstar=cstar;self.paths=paths;self.active=False;self.gain=1.;self.capture=False
                self.delta=None;self.calls=0;self.packet=None;self.enabled=False;self.causal=None
            def forward(self,features):
                self.packet=None;self.delta=None
                if not self.active:return features
                self.calls+=1;base=self.cstar(features)
                if self.enabled:
                    packet=self.paths(base,self.causal);packet['base']=base;self.packet=packet;self.delta=packet['regularizer']
                return base
        self.hook=Hook();self.base.model.pixel_decoder.persistent_feature_repair=self.hook;self.base.gate=self.hook
        for p in self.base.model.parameters():p.requires_grad_(False)
        if paths is not None:
            for p in paths.parameters():p.requires_grad_(True)
        self.base.model.eval()

    def configure(self,seq):self.base.configure(seq);self.seq=seq
    def load(self,i,gt=False):return self.base.load(i,gt)
    def causal(self,static,index):
        t=self.t;flow=static['flow_norm'].float();res=static['image_residual'].float()
        res=t.nn.functional.interpolate(res,size=flow.shape[-2:],mode='bilinear',align_corners=False)
        age=t.full_like(res,float(index%self.ctx.policy['train_K'])/self.ctx.policy['train_K'])
        return t.cat([flow,res,age],1).detach()
    def read_features(self,features,rays):
        from y08y09_models import readout_radius
        with self.t.autocast('cuda',dtype=self.t.float16):
            r=readout_radius(self.base.model,features)
            return self.ctx.rt.gt2.make_z_depth(r,rays,self.base.shape['padded_shape'],self.base.shape['paddings'],self.base.model.interpolation_mode)
    def step(self,state,prev,frame,extra=True,static=None):
        t=self.t;i=frame['index'];apply=active(i,self.ctx.policy['evaluation_K'],self.ctx.policy['evaluation_period'])
        self.hook.packet=None;self.hook.enabled=bool(extra and self.paths is not None and apply)
        if i%self.ctx.policy['evaluation_K'] and static is None:static=self.base.static(prev,frame,'S8')
        if self.hook.enabled:self.hook.causal=self.causal(static,i)
        state,_=self.base.tensor_step(state,prev,frame,self.ctx.policy['evaluation_K'],'S8',apply=apply,static=static,checkpoint=False)
        packet=self.hook.packet
        if packet is not None:
            # The current readout branch never enters ANY persistent field, including radius.
            pred=self.read_features(packet['read'],state[3]);state=(state[0],state[1],packet['commit'],state[3])
        else:pred=self.base.read_depth(state)
        require(finite_state(t,state),'Nonfinite persistent state')
        return state,pred,packet
    def close(self):
        self.hook.packet=None;self.base.close();del self.base,self.hook,self.paths;gc.collect();self.t.cuda.empty_cache()

def parent_bank(ctx,engine):
    result={};t=ctx.rt.torch
    for s in ctx.policy['old_sequences']:
        wanted=sorted(c['parent'] for c in ctx.plan['train_cases'] if c['sequence']==s)
        p=ctx.out/'cache'/s/'parents.pt';meta=p.with_suffix('.json')
        if meta.is_file():
            m=read(meta);require(m['signature']==ctx.signature and m['indices']==wanted and m['sha256']==sha(p),'Parent cache changed')
            obj=t.load(p,map_location='cpu',weights_only=False);require(obj['signature']==ctx.signature,'Foreign parent bank');result[s]=obj['states'];continue
        engine.configure(s);state=prev=None;states={}
        with t.no_grad():
            for i in range(max(wanted)+1):
                frame=engine.load(i);state,_,_=engine.step(state,prev,frame,extra=False)
                if i in wanted:states[i]=detach(t,state,'cpu')
                prev=frame
        save_torch(t,p,dict(signature=ctx.signature,states=states));write(meta,dict(signature=ctx.signature,indices=wanted,sha256=sha(p)))
        result[s]=states;print('[CACHE]',s,len(states),'C* prefix parents',flush=True)
    return result

def pair_forward(ctx,eng,case,parent,negative=False):
    t=ctx.rt.torch;eng.configure(case['sequence']);i=case['anchor']
    f0,f1,f2=eng.load(i-1),eng.load(i,True),eng.load(i+1,True)
    st0=eng.base.static(f0,f1,'S8');st1=eng.base.static(f1,f2,'S8')
    state,read_pred,packet=eng.step(detach(t,parent,'cuda'),f0,f1,extra=True,static=st0)
    require(packet is not None,'Training anchor did not execute residual paths')
    current=loss(t,read_pred,f1['gt'])
    base_state=(state[0],state[1],packet['base'],state[3])
    # No transient readout-radius or readout features are reused for the future loss.
    next_state,next_pred,_=eng.step(detach(t,state) if negative else state,f1,f2,extra=False,static=st1)
    future=loss(t,next_pred,f2['gt'])
    with t.no_grad():
        _,reference_pred,_=eng.step(detach(t,base_state),f1,f2,extra=False,static=st1)
        reference=loss(t,reference_pred,f2['gt']);reference0=eng.base.read_depth(base_state).detach()
    committed0=eng.read_features(packet['commit'],state[3])
    mask=(f1['gt']>0)&t.isfinite(f1['gt'])
    function=((committed0.float()[mask]-reference0.float()[mask]).abs()/reference0.float()[mask].clamp_min(1e-6)).mean()
    gate=current.new_zeros(());harm=current.new_zeros(())
    if ctx.policy['experiment'].startswith('D') and eng.arm.endswith('_B'):
        from .models import gate_loss
        # Labels evaluate the UNGATED transport residual; gate cannot create its own target by closing.
        with t.no_grad():
            ungated=(state[0],state[1],packet['ungated'],state[3])
            _,ungated_pred,_=eng.step(detach(t,ungated),f1,f2,extra=False,static=st1)
            delta=loss(t,ungated_pred,f2['gt'])-reference
        gate=gate_loss(packet['alpha'],delta,ctx.policy['gate_tau']);harm=t.relu(future-reference)
    p=ctx.policy
    objective=current+p['lambda_future']*future+p['lambda_function']*function+p['lambda_residual']*packet['regularizer']+p['lambda_gate']*gate+p['lambda_harm']*harm
    return dict(objective=objective,current=current,future=future,reference=reference,function=function,regularizer=packet['regularizer'],gate=gate,harm=harm)
