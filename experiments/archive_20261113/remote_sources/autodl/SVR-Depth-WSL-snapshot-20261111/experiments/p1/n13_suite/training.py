from .common import *
from .runtime import *
import math,random,time

def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
def seed(ctx):
    p=ctx.policy['seed'];random.seed(p);ctx.rt.np.random.seed(p);ctx.rt.torch.manual_seed(p);ctx.rt.torch.cuda.manual_seed_all(p)
def rng(ctx):
    t=ctx.rt.torch;return dict(py=random.getstate(),np=ctx.rt.np.random.get_state(),cpu=t.get_rng_state(),cuda=t.cuda.get_rng_state_all())
def restore_rng(ctx,r):
    t=ctx.rt.torch;random.setstate(r['py']);ctx.rt.np.random.set_state(r['np']);t.set_rng_state(r['cpu']);t.cuda.set_rng_state_all(r['cuda'])

def real_audit(ctx,parents):
    """Real frozen VeloDepth canaries; finite positive transport gradients and detached negative control."""
    t=ctx.rt.torch;report=[];seed(ctx);eng=AdaptedEngine(ctx,ctx.old_sequences[0],ctx.policy['arms'][1])
    try:
        for s in ctx.old_sequences:
            case=next(c for c in ctx.plan['train_cases'] if c['sequence']==s)
            value=pair_forward(ctx,eng,case,parents[s][case['parent']]);params=list(eng.paths.parameters())
            grad=t.autograd.grad(value['future']*1024.,params,allow_unused=True)
            magnitude=sum(float(x.detach().float().abs().sum().item())/1024. for x in grad if x is not None)
            require(magnitude>0 and all(bool(t.isfinite(x).all().item()) for x in grad if x is not None),'Future gradient missing/nonfinite on real VeloDepth '+s)
            # Only shared trunk/transport/gate may receive H1 gradients in separated branch B.
            read_ids={id(p) for p in eng.paths.read_heads.parameters()}
            require(all(g is None for p,g in zip(params,grad) if id(p) in read_ids),'Transient readout leaked into persistent H1 state')
            del value,grad
            neg=pair_forward(ctx,eng,case,parents[s][case['parent']],negative=True)
            require(not neg['future'].requires_grad,'Detached negative control still has a transport graph')
            report.append(dict(sequence=s,case_id=case['id'],future_gradient_l1=magnitude,readout_to_future_gradient=False,detach_negative_control=True))
            del neg;t.cuda.empty_cache()
        # Zero-initialized new paths must reproduce actual C* on a complete native schedule.
        original=AdaptedEngine(ctx,ctx.old_sequences[0],None);eng.configure(ctx.old_sequences[0]);a=b=prev=None;maximum=0.
        with t.no_grad():
            for i in range(40):
                frame=eng.load(i);a,pa,_=eng.step(a,prev,frame);b,pb,_=original.step(b,prev,frame)
                maximum=max(maximum,float((pa.float()-pb.float()).abs().max().item()))
                require(t.equal(a[0],b[0]) and t.equal(a[1],b[1]) and t.equal(a[3],b[3]),'Zero identity changed persistent non-F fields')
                require(all(t.equal(x,y) for x,y in zip(a[2],b[2])),'Zero-initialized paths changed C* features')
                prev=frame
        require(maximum<=1e-4,'Zero-initialized readout does not reproduce C* depth')
        original.close();write(ctx.out/'gradient_audit.json',dict(signature=ctx.signature,passed=True,records=report,zero_identity_depth_max_error=maximum,
            actual_VeloDepth_CUDA=True,no_optimizer_updates=True))
        print('[REAL GRADIENT PASS] H1 reaches transport; readout isolated; zero initialization reproduces C*',flush=True)
    finally:eng.close()

def train_one(ctx,parents,arm,resume):
    t=ctx.rt.torch;seed(ctx);eng=AdaptedEngine(ctx,ctx.old_sequences[0],arm);p=ctx.policy
    d=ctx.out/'training'/arm;d.mkdir(parents=True,exist_ok=True);recovery=d/'recovery.pt';final=d/'final.pt';summary=d/'summary.json'
    params=list(eng.paths.parameters());initial_sha=ctx.rt.exp1.state_sha256(cpu_state(eng.paths))
    if final.exists() and summary.exists():
        done=read(summary);obj=t.load(final,map_location='cpu',weights_only=False)
        require(done['signature']==obj['signature']==ctx.signature and obj['arm']==arm and obj['step']==ctx.plan['updates'] and obj['tensor_sha256']==ctx.rt.exp1.state_sha256(obj['state']),'Completed arm identity differs')
        require(done['final_file_sha256']==sha(final),'Completed checkpoint bytes changed')
        print('[REUSE]',arm,'completed checkpoint',flush=True);eng.close();return done
    opt=t.optim.AdamW(params,lr=p['learning_rate'],weight_decay=p['weight_decay'])
    scaler=t.amp.GradScaler('cuda',init_scale=1024.,growth_interval=2000);curve=[];step0=0
    if recovery.exists():
        require(resume,'Recovery exists: use --resume-latest');r=t.load(recovery,map_location='cpu',weights_only=False)
        require(r['signature']==ctx.signature and r['arm']==arm and r['initial_sha256']==initial_sha,'Recovery identity differs')
        require(r['sample_order_sha256']==ctx.plan['sample_order_sha256'] and r['step']==len(r['curve']),'Recovery cursor differs')
        check_recovery(r['curve'],ctx.plan['sample_order']);curve=r['curve'];step0=r['step']
        eng.paths.load_state_dict(r['state'],strict=True);opt.load_state_dict(r['optimizer']);scaler.load_state_dict(r['scaler']);restore_rng(ctx,r['rng'])
        print('[RESUME]',arm,step0,'/',ctx.plan['updates'],flush=True)
    cases={c['id']:c for c in ctx.plan['train_cases']};journal=d/'physical_updates.jsonl';start=time.perf_counter()
    def commit(step):
        state=cpu_state(eng.paths)
        save_torch(t,recovery,dict(signature=ctx.signature,arm=arm,step=step,state=state,optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=rng(ctx),
            initial_sha256=initial_sha,sample_order_sha256=ctx.plan['sample_order_sha256'],curve=curve))
        csvwrite(d/'curve.csv',curve)
        if step==ctx.plan['updates']:save_torch(t,final,dict(signature=ctx.signature,arm=arm,step=step,state=state,tensor_sha256=ctx.rt.exp1.state_sha256(state),
            Cstar_tensor_sha256=ctx.plan['Cstar_tensor_sha256'],policy=ctx.policy))
    try:
        for step in range(step0+1,ctx.plan['updates']+1):
            case=cases[ctx.plan['sample_order'][step-1]];opt.zero_grad(set_to_none=True);t.cuda.synchronize();begin=time.perf_counter()
            ev0=t.cuda.Event(enable_timing=True);ev1=t.cuda.Event(enable_timing=True);ev0.record()
            result=pair_forward(ctx,eng,case,parents[case['sequence']][case['parent']])
            require(bool(t.isfinite(result['objective']).item()),'Nonfinite objective')
            scaler.scale(result['objective']).backward();scaler.unscale_(opt)
            grad=[x.grad for x in params if x.grad is not None]
            require(grad and all(bool(t.isfinite(g).all().item()) for g in grad),'Nonfinite/missing gradients')
            ids={id(x) for x in params};require(all(not x.requires_grad and x.grad is None for x in eng.base.model.parameters() if id(x) not in ids),'Frozen model acquired gradients')
            norm=float(t.nn.utils.clip_grad_norm_(params,p['clip_grad_norm']).item());require(norm>0 and norm<float('inf'),'Invalid gradient norm')
            scale=scaler.get_scale();scaler.step(opt);scaler.update();require(scaler.get_scale()>=scale,'AMP skipped update; recovery cursor not advanced')
            require(all(bool(t.isfinite(x).all().item()) for x in params),'Nonfinite learned weights');ev1.record();t.cuda.synchronize()
            row=dict(step=step,arm=arm,case_id=case['id'],sequence=case['sequence'],anchor=case['anchor'],gradient_norm=norm,
                cuda_sec=ev0.elapsed_time(ev1)/1000.,step_wall_sec=time.perf_counter()-begin,current_GT=1,future_GT=1,
                **{k:float(v.detach().item()) for k,v in result.items()})
            row['timing_valid']=math.isfinite(row['cuda_sec']) and 0<row['cuda_sec']<=row['step_wall_sec']*1.1+.05
            # If an interruption left a partial record, preserve it as its own line.
            if journal.exists() and journal.stat().st_size:
                with journal.open('rb') as f:f.seek(-1,2);needs_newline=f.read(1)!=b'\n'
                if needs_newline:
                    with journal.open('ab') as f:f.write(b'\n')
            with journal.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            require(row['timing_valid'],'CUDA interval inconsistent with wall time; physical attempt retained, recovery not advanced')
            curve.append(row)
            if step%p['checkpoint_interval']==0 or step==ctx.plan['updates']:commit(step)
            if step==1 or step%10==0 or step==ctx.plan['updates']:
                print(f'[TRAIN {arm}] {step}/{ctx.plan["updates"]} current={row["current"]:.6f} H1={row["future"]:.6f} wall={row["step_wall_sec"]:.2f}s',flush=True)
            eng.hook.packet=None;del result
        require(final.is_file(),'Final checkpoint missing')
        physical=[];malformed=0
        if journal.exists():
            for line in journal.read_text().splitlines():
                if not line.strip():continue
                try:physical.append(json.loads(line))
                except json.JSONDecodeError:malformed+=1
        done=dict(signature=ctx.signature,status='TRAINING_COMPLETE',arm=arm,updates=len(curve),trainable_parameters=sum(x.numel() for x in params),
            initial_sha256=initial_sha,final_file_sha256=sha(final),committed_cuda_sec=sum(r['cuda_sec'] for r in curve),
            committed_wall_sec=sum(r['step_wall_sec'] for r in curve),current_invocation_sec=time.perf_counter()-start,
            physical_updates=len(physical),physical_cuda_sec=sum(r['cuda_sec'] for r in physical),
            physical_wall_sec=sum(r['step_wall_sec'] for r in physical),malformed_physical_journal_lines=malformed,
            recovery_scope='atomic last committed optimizer/scaler/RNG cursor; later physical attempts may replay',
            physical_journal=str(journal),compute_claim=ctx.policy['compute_claim'])
        write(summary,done);return done
    finally:eng.close()

def train_all(ctx,resume=False):
    carrier=AdaptedEngine(ctx,ctx.old_sequences[0],None)
    try:parents=parent_bank(ctx,carrier)
    finally:carrier.close()
    # Audit is deliberately replayed before optimizer recovery, no inputs or thresholds changed.
    real_audit(ctx,parents);summaries={}
    for arm in ctx.policy['arms']:summaries[arm]=train_one(ctx,parents,arm,resume)
    write(ctx.out/'training_summary.json',summaries)
