"""Single-pulse training, full-state recovery, and an actual measured-time P control."""
from y05_common import *
from y05_engine import Engine,ensure_bank,parent_for,pulse_forward,detached_state,tsave
import time

def seed(ctx):
    rt=ctx.rt;s=ctx.plan['settings']['seed'];random.seed(s);rt.np.random.seed(s);rt.torch.manual_seed(s);rt.torch.cuda.manual_seed_all(s)
def rng(ctx):
    t=ctx.rt.torch;return dict(py=random.getstate(),np=ctx.rt.np.random.get_state(),torch=t.get_rng_state(),cuda=t.cuda.get_rng_state_all())
def restore_rng(ctx,x):
    random.setstate(x['py']);ctx.rt.np.random.set_state(x['np']);ctx.rt.torch.set_rng_state(x['torch']);ctx.rt.torch.cuda.set_rng_state_all(x['cuda'])
def cpu_head(engine):return {k:v.detach().cpu().clone() for k,v in engine.head.state_dict().items()}

def gradient_vector(torch,loss,params,retain=False):
    require(loss.requires_grad,'Future objective has no gradient graph')
    # Fixed scaling avoids diagnosing FP16 underflow as a detached state graph.
    g=torch.autograd.grad(loss*1024.,params,allow_unused=True,retain_graph=retain)
    vector=torch.cat([(torch.zeros_like(p) if a is None else a).detach().float().reshape(-1) for p,a in zip(params,g)])/1024.
    require(bool(torch.isfinite(vector).all().item()) and float(vector.abs().sum().item())>0,'Future loss cannot reach head with finite nonzero gradients')
    return vector

def gradient_audit(ctx,engine,paths):
    path=ctx.out/'gradient_audit.json'
    if path.exists():
        obj=jread(path);require(obj['signature']==ctx.signature and obj['passed'],'Gradient audit differs');return
    torch=ctx.rt.torch;engine.head.load_state_dict(ctx.head_state);records=[];initial=ctx.rt.exp1.state_sha256(cpu_head(engine))
    for seq in SEQS:
        for origin in ('A','S8'):
            choices=[c for c in ctx.plan['sample_plan'] if c['sequence']==seq and c['origin']==origin and all(matched(ctx.manifests[seq][c['anchor']+h]) for h in range(5))]
            require(choices,'Gradient canary needs a full-GT prefix window');case=choices[0]
            frames,statics=engine.context(case);parent=parent_for(ctx,paths,case);params=list(engine.head.parameters())
            result=pulse_forward(ctx,engine,case,parent,frames,statics,'FH_future_harm',checkpoint=True)
            last=gradient_vector(torch,result['future_losses'][-1],params,retain=True)
            combined=gradient_vector(torch,result['future'],params)
            records.append(dict(sequence=seq,origin=origin,last_h=4,mean_future_grad_l1=float(combined.abs().sum().item()),h4_grad_l1=float(last.abs().sum().item())))
            del result,last,combined
            negative=pulse_forward(ctx,engine,case,parent,frames,statics,'FH_future_harm',checkpoint=True,detach_after_anchor=True)
            require(not negative['future'].requires_grad,'Detached-state negative control still reaches head')
            del negative
            # A one-future-step check fits the 8GB target; full H4 plain autograd can exceed it.
            short={**case,'horizon':1}
            a=pulse_forward(ctx,engine,short,parent,frames,statics,'F_future',checkpoint=True)
            ga=gradient_vector(torch,a['future'],params);la=float(a['future'].detach().item());del a
            b=pulse_forward(ctx,engine,short,parent,frames,statics,'F_future',checkpoint=False)
            gb=gradient_vector(torch,b['future'],params);lb=float(b['future'].detach().item());del b
            rel=float(torch.linalg.vector_norm(ga-gb).item()/max(torch.linalg.vector_norm(gb).item(),1e-12))
            require(abs(la-lb)<=1e-6 and rel<=5e-3,'Checkpoint recomputation changed future loss/gradient')
            records[-1].update(detach_negative_control=True,H1_checkpoint_loss_difference=abs(la-lb),H1_checkpoint_gradient_relative_difference=rel)
            del ga,gb,frames,statics,parent;torch.cuda.empty_cache()
    require(ctx.rt.exp1.state_sha256(cpu_head(engine))==initial,'Gradient audit changed weights')
    jwrite(path,dict(signature=ctx.signature,passed=True,records=records,full_H4_future_gradient_checked=True,H1_checkpoint_vs_plain_checked=True,scope='Real model/CUDA; no optimizer updates; tolerates nondeterministic CUDA backward differences only up to frozen 0.5%'))

def materialize(ctx,directory,obj,kind):
    """Every published head is derived from one atomically committed recovery object."""
    for name,snapshot in obj['selected'].items():
        path=directory/(name+'.pt')
        payload=dict(signature=ctx.signature,kind=kind,selection=name,step=snapshot['step'],state=snapshot['state'],head_sha256=snapshot['head_sha256'],cuda_sec=snapshot['cuda_sec'],warmstart_checkpoint_sha256=ctx.plan['checkpoint_sha256'])
        if path.exists():
            old=ctx.rt.torch.load(path,map_location='cpu',weights_only=False)
            require(old['signature']==ctx.signature and old['head_sha256']==payload['head_sha256'] and old['step']==payload['step'],'Published training snapshot changed')
        else:tsave(ctx.rt.torch,path,payload)
    cwrite(directory/'curve.csv',obj['curve'])

def train_arm(ctx,engine,paths,kind,target_cuda=None):
    torch=ctx.rt.torch;cfg=ctx.plan['settings'];directory=ctx.out/'training'/kind;directory.mkdir(parents=True,exist_ok=True)
    recovery=directory/'recovery.pt';seed(ctx);engine.head.load_state_dict(ctx.head_state,strict=True)
    optimizer=torch.optim.AdamW(engine.head.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=cfg['amp_initial_scale'],growth_interval=cfg['amp_growth_interval']);initial=ctx.rt.exp1.state_sha256(cpu_head(engine));curve=[];selected={};step0=0;prior_peak=0.
    if recovery.exists():
        require(ctx.args.resume,'Training recovery exists; use --resume')
        obj=torch.load(recovery,map_location='cpu',weights_only=False)
        require(obj['signature']==ctx.signature and obj['kind']==kind and obj['initial_sha']==initial and obj['target_cuda_sec']==target_cuda,'Recovery training identity/budget differs')
        curve=obj['curve'];checked_curve(curve,ctx.plan['sample_order']);step0=obj['step'];selected=obj['selected']
        prior_peak=float(obj.get('peak_cuda_allocated_gib',0.))
        require(step0==len(curve),'Recovery cursor differs')
        engine.head.load_state_dict(obj['state']);optimizer.load_state_dict(obj['optimizer']);scaler.load_state_dict(obj['scaler']);restore_rng(ctx,obj['rng'])
        materialize(ctx,directory,obj,kind);print(f'[RESUME {kind}] committed update {step0}',flush=True)
    cases={c['case_id']:c for c in ctx.plan['sample_plan']};limit=cfg['max_P_updates'] if kind=='P_closed_loop' else cfg['fixed_updates']
    journal=directory/'physical_updates.jsonl';torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    def done():return ('equal_updates' in selected and 'equal_compute' in selected) if kind=='P_closed_loop' else 'final' in selected
    def commit(step):
        obj=dict(signature=ctx.signature,kind=kind,step=step,initial_sha=initial,target_cuda_sec=target_cuda,state=cpu_head(engine),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng(ctx),curve=curve,selected=selected,peak_cuda_allocated_gib=max(prior_peak,torch.cuda.max_memory_allocated()/1024**3))
        tsave(torch,recovery,obj);materialize(ctx,directory,obj,kind)
    for step in range(step0+1,limit+1):
        if done():break
        case=cases[ctx.plan['sample_order'][step-1]]
        torch.cuda.synchronize();event0=torch.cuda.Event(enable_timing=True);event1=torch.cuda.Event(enable_timing=True)
        wall_start=time.perf_counter();event0.record();torch.cuda.synchronize()
        frames,statics=engine.context(case);parent=parent_for(ctx,paths,case);prep=time.perf_counter()-wall_start
        optimizer.zero_grad(set_to_none=True);calls0=engine.gate.calls
        result=pulse_forward(ctx,engine,case,parent,frames,statics,kind,checkpoint=True)
        require(bool(torch.isfinite(result['objective']).item()),'Nonfinite training objective')
        scaler.scale(result['objective']).backward();scaler.unscale_(optimizer)
        gradients=[p.grad for p in engine.head.parameters() if p.grad is not None]
        require(gradients and all(bool(torch.isfinite(g).all().item()) for g in gradients),'Nonfinite/missing head gradient; update not committed')
        head_ids={id(p) for p in engine.head.parameters()}
        require(all(not p.requires_grad and p.grad is None for p in engine.model.parameters() if id(p) not in head_ids),'Frozen backbone acquired gradient')
        norm=float(torch.nn.utils.clip_grad_norm_(engine.head.parameters(),cfg['clip_grad_norm']).item())
        require(math.isfinite(norm),'Nonfinite gradient norm')
        if step==1:require(norm>0,'First training update has zero gradient')
        old_scale=float(scaler.get_scale());scaler.step(optimizer);scaler.update()
        require(float(scaler.get_scale())>=old_scale,'AMP skipped an update; stopped without advancing recovery')
        require(all(bool(torch.isfinite(p).all().item()) for p in engine.head.parameters()),'Optimizer produced nonfinite head weights')
        event1.record();torch.cuda.synchronize();cuda_sec=float(event0.elapsed_time(event1))/1000.
        require(math.isfinite(cuda_sec) and cuda_sec>0,'Invalid CUDA step interval')
        row=dict(step=step,case_id=case['case_id'],sequence=case['sequence'],origin=case['origin'],anchor=case['anchor'],horizon=4,
            loss=float(result['objective'].detach().item()),current_AbsRel=float(result['current'].detach().item()),future_AbsRel=float(result['future'].detach().item()),reference_future_AbsRel=float(result['reference'].detach().item()),harm_penalty=float(result['harm'].detach().item()),normalized_anchor_delta=float(result['delta'].detach().item()),gradient_norm=norm,
            current_GT=result['current_GT'],future_GT=result['future_GT'],logical_head_calls=5 if kind=='P_closed_loop' else 1,physical_head_forwards=engine.gate.calls-calls0,
            cuda_sec=cuda_sec,preparation_wall_sec=prep,step_wall_sec=time.perf_counter()-wall_start)
        curve.append(row)
        with journal.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        elapsed=sum(r['cuda_sec'] for r in curve);snapshot_trigger=False
        names=[]
        if kind=='P_closed_loop':
            if step==cfg['fixed_updates']:names.append('equal_updates')
            if 'equal_compute' not in selected and compute_crossed(elapsed,target_cuda):names.append('equal_compute')
        elif step==cfg['fixed_updates']:names.append('final')
        for name in names:
            state=cpu_head(engine);selected[name]=dict(step=step,state=state,head_sha256=ctx.rt.exp1.state_sha256(state),cuda_sec=elapsed)
            snapshot_trigger=True
        if snapshot_trigger or step%cfg['checkpoint_interval']==0:commit(step)
        if step==1 or step%10==0 or snapshot_trigger:print(f'[TRAIN {kind}] {step} loss={row["loss"]:.6f} future={row["future_AbsRel"]:.6f} CUDA={elapsed:.1f}s snapshots={list(selected)}',flush=True)
        engine.gate.delta=None
        del result,frames,statics,parent
    require(done(),f'{kind}: max update limit reached before frozen CUDA budget; no matched-compute claim')
    checked_curve(curve,ctx.plan['sample_order']);require(all(x['head_sha256']!=initial for x in selected.values()),'Training never changed head')
    physical=[json.loads(x) for x in journal.read_text().splitlines() if x.strip()]
    require(len(physical)>=len(curve),'Physical journal lost updates')
    details=dict(signature=ctx.signature,kind=kind,steps=len(curve),initial_head_sha256=initial,inherited_checkpoint_sha256=ctx.plan['checkpoint_sha256'],
        samples_order_sha256=ctx.plan['sample_order_sha256'],committed_cuda_sec=sum(r['cuda_sec'] for r in curve),physical_cuda_sec=sum(r['cuda_sec'] for r in physical),physical_updates=len(physical),physical_GT_exposures=sum(r['current_GT']+r['future_GT'] for r in physical),
        committed_current_GT_exposures=sum(r['current_GT'] for r in curve),committed_future_GT_exposures=sum(r['future_GT'] for r in curve),GT_supervision='C: current only; other arms: future only; harm references use same future GT',
        physical_preparation_wall_sec=sum(r['preparation_wall_sec'] for r in physical),physical_step_wall_sec=sum(r['step_wall_sec'] for r in physical),
        current_invocation_sec=time.perf_counter()-started,peak_cuda_allocated_gib=max(prior_peak,torch.cuda.max_memory_allocated()/1024**3),
        selected={k:{x:v for x,v in s.items() if x!='state'} for k,s in selected.items()},target_cuda_sec=target_cuda,
        timing_scope='CUDA-event interval includes per-step preparation/DIS, forward/reference/backward/optimizer; excludes checkpoint I/O; common cache costs separate',
        compute_match_scope='committed optimization trajectory; interrupted physical repeats separately counted; not equal FLOPs/GT exposures/total wall',
        initial_P_exposure='inherits original reviewed Y04 P_multi; not training from scratch')
    if kind=='P_closed_loop':
        s=selected['equal_compute'];last=curve[s['step']-1]['cuda_sec'];over=s['cuda_sec']-target_cuda
        require(0<=over<last+1e-7,'Compute checkpoint was not first crossing')
        details.update(compute_match_relative_overshoot=over/target_cuda,compute_match_overshoot_sec=over,compute_match_last_step_sec=last)
    jwrite(directory/'summary.json',details)
    del optimizer,scaler;torch.cuda.empty_cache();return details

def train_all(ctx):
    engine=Engine(ctx,SEQS[0]);paths=ensure_bank(ctx,engine,ctx.plan['sample_plan'],'fit')
    try:
        gradient_audit(ctx,engine,paths);summaries={}
        for kind in TRAIN_ARMS:
            target=summaries['FH_future_harm']['selected']['final']['cuda_sec'] if kind=='P_closed_loop' else None
            summaries[kind]=train_arm(ctx,engine,paths,kind,target)
        jwrite(ctx.out/'training_summary.json',summaries);return summaries
    finally:engine.close()

def head_states(ctx):
    states={'P_init':ctx.head_state};paths={}
    for kind in TRAIN_ARMS:
        for selection in (('equal_updates','equal_compute') if kind=='P_closed_loop' else ('final',)):
            path=ctx.out/'training'/kind/(selection+'.pt');obj=ctx.rt.torch.load(path,map_location='cpu',weights_only=False)
            require(obj['signature']==ctx.signature and ctx.rt.exp1.state_sha256(obj['state'])==obj['head_sha256'],'Selected checkpoint identity differs')
            label='P_'+selection if kind=='P_closed_loop' else kind;states[label]=obj['state'];paths[label]=dict(path=str(path),sha256=sha(path),step=obj['step'])
    return states,paths
