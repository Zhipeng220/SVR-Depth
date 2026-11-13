"""Current-frame-only training from exact frozen C* trajectory caches."""
from .common import *
from .runtime import *
import math,random,time

def seed(ctx):
    s=ctx.policy['seed'];random.seed(s);ctx.rt.np.random.seed(s);ctx.rt.torch.manual_seed(s);ctx.rt.torch.cuda.manual_seed_all(s)
def rng(ctx):
    t=ctx.rt.torch;return dict(py=random.getstate(),np=ctx.rt.np.random.get_state(),cpu=t.get_rng_state(),cuda=t.cuda.get_rng_state_all())
def restore_rng(ctx,x):
    t=ctx.rt.torch;random.setstate(x['py']);ctx.rt.np.random.set_state(x['np']);t.set_rng_state(x['cpu']);t.cuda.set_rng_state_all(x['cuda'])
def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
def new_micro(ctx,arm):
    from .models import PhaseMicroReadout
    seed(ctx);m=PhaseMicroReadout(conditioned=arm=='N14_A1').cuda()
    require(sum(p.numel() for p in m.parameters())==9244,'Micro head parameter count changed');return m

def real_smoke(ctx):
    t=ctx.rt.torch;models={a:new_micro(ctx,a) for a in ctx.policy['arms']};records=[]
    for s in ctx.old_sequences:
        car=FrozenCarrier(ctx,s);maximum=0.;seen=set();calls={a:0 for a in models}
        try:
            with t.no_grad():
                for i in range(32):
                    frame=car.load(i);native,static=car.advance(frame);before=state_digest(car.state);versions=state_versions(car.state);original=car.state;seen.add(phase(i))
                    for arm,m in models.items():
                        pred,reg=car.micro(m,native,static,i)
                        require(car.state is original and state_versions(car.state)==versions,'Micro readout replaced/mutated persistent tensors')
                        compare_state(before,state_digest(car.state));require(float(reg.item())==0.,'Micro initialization is not an identity residual')
                        err=float((pred.float()-native.float()).abs().max().item());maximum=max(maximum,err)
                        if phase(i)==0:require(t.equal(pred,native),'h=0 failed exact bypass')
                        else:calls[arm]+=1
                    require(finite_state(t,car.state),'Nonfinite frozen state')
            require(seen==set(range(8)) and all(n==28 for n in calls.values()),'Smoke lacks a complete K32/S8 period')
            require(maximum<=1e-4,'Zero-init transient decoder does not reproduce C*')
            records.append(dict(sequence=s,RGB=32,phases=sorted(seen),micro_calls=calls,zero_init_max_depth_error=maximum,state_unchanged=True))
        finally:car.close()
    write(ctx.out/'smoke_checks.json',dict(signature=ctx.signature,passed=True,actual_model_CUDA=True,parameter_count=9244,records=records))
    print('[SMOKE PASS] three full K32 periods; h0 bypass; 28 micro calls/32; exact persistent state identity',flush=True)
    del models;t.cuda.empty_cache()

def cache_paths(ctx,case):
    p=ctx.out/'cache'/case['sequence']/(f'{case["index"]:06d}.pt');return p,p.with_suffix('.json')
def verify_cache(ctx,case):
    p,meta=cache_paths(ctx,case);m=read(meta)
    require(m['signature']==ctx.signature and m['case_id']==case['id'] and m['file_sha256']==sha(p),'Current-frame cache identity changed')
    return m
def build_cache(ctx):
    t=ctx.rt.torch;all_cases=ctx.plan['train_cases'];selected=all_cases
    if ctx.plan['partial_smoke']:selected=[next(c for c in all_cases if c['sequence']==s and c['phase']==h) for s in ctx.old_sequences for h in range(1,8)]
    manifest={};cost=[]
    for s in ctx.old_sequences:
        wanted={c['index']:c for c in selected if c['sequence']==s};complete=ctx.out/'cache'/s/'complete.json'
        if complete.exists():
            m=read(complete);require(m['signature']==ctx.signature and m['indices']==sorted(wanted),'Cache sequence plan changed')
            for c in wanted.values():manifest[c['id']]=verify_cache(ctx,c)
            cost.append(m);continue
        car=FrozenCarrier(ctx,s);start=time.perf_counter();estimated=False
        try:
            with t.no_grad():
                for i in range(max(wanted)+1):
                    frame=car.load(i,i in wanted);native,static=car.advance(frame)
                    require(finite_state(t,car.state),'Nonfinite C* prefix state, including missing-GT RGB')
                    if i in wanted:
                        case=wanted[i];p,mp=cache_paths(ctx,case)
                        if mp.exists():manifest[case['id']]=verify_cache(ctx,case)
                        else:
                            gt=frame['gt'];require(gt is not None,'Current training GT absent');mask=(gt>0)&t.isfinite(gt);require(bool(mask.any().item()),'Empty current-frame GT mask')
                            features=[x.detach().cpu().clone() for x in car.state[2]];rays=car.state[3].detach().cpu().clone();causal=car.causal(static).cpu().clone()
                            obj=dict(signature=ctx.signature,case_id=case['id'],sequence=s,index=i,phase=phase(i),features=features,rays=rays,causal=causal,gt=gt.detach().cpu().clone(),mask=mask.cpu().clone())
                            size=sum(x.numel()*x.element_size() for x in [*features,rays,causal,obj['gt'],obj['mask']])
                            if not estimated:
                                remain=sum(not cache_paths(ctx,c)[1].exists() for c in wanted.values())
                                require(shutil.disk_usage(ctx.out).free>size*remain*1.15+(1<<30),'Insufficient disk for this frozen prefix cache; free space before --resume-latest')
                                print('[CACHE SPACE]',s,f'roughly {size*remain/1024**3:.2f} GiB remaining for this sequence',flush=True);estimated=True
                            tensor_sha=dict(features=[tensor_digest(x) for x in features],rays=tensor_digest(rays),causal=tensor_digest(causal),gt=tensor_digest(obj['gt']),mask=tensor_digest(obj['mask']))
                            save_torch(t,p,obj);record=ctx.manifests[s][i]
                            m=dict(signature=ctx.signature,case_id=case['id'],sequence=s,index=i,phase=phase(i),path=str(p),file_sha256=sha(p),Cstar_checkpoint_sha256=ctx.plan['Cstar_file_sha256'],code_sha256=canonical({k:v for k,v in ctx.plan['input_sha256'].items() if k.endswith('.py')}),
                                rgb_path=record['rgb_path'],rgb_sha256=ctx.plan['input_sha256'][str(Path(record['rgb_path']).resolve())],GT_path=record['depth_path'],GT_sha256=ctx.plan['input_sha256'][str(Path(record['depth_path']).resolve())],
                                tensor_sha256=tensor_sha,flow_norm_sha256=tensor_digest(static['flow_norm']),image_residual_sha256=tensor_digest(static['image_residual']),feature_dtypes=[str(x.dtype) for x in features],feature_shapes=[list(x.shape) for x in features],valid_pixels=int(mask.sum().item()))
                            write(mp,m);manifest[case['id']]=m
                    if i==0 or (i+1)%64==0 or i==max(wanted):print('[CACHE]',s,f'{i+1}/{max(wanted)+1}',flush=True)
            m=dict(signature=ctx.signature,sequence=s,indices=sorted(wanted),RGB_replayed=max(wanted)+1,wall_sec=time.perf_counter()-start,cache_files=len(wanted));write(complete,m);cost.append(m)
        finally:car.close()
    write(ctx.out/'cache_manifest.json',dict(signature=ctx.signature,cases=manifest,sequences=cost,total_completed_cache_wall_sec=sum(r['wall_sec'] for r in cost),scope='shared once by A0/A1; every prefix RGB replayed from zero; current-frame tensors only',physical_retry_cost='partial interrupted cache attempts may add cost; see invocation logs'))
    return manifest

def load_cache(ctx,case):
    meta=verify_cache(ctx,case);obj=ctx.rt.torch.load(meta['path'],map_location='cpu',weights_only=False)
    require(set(obj)=={'signature','case_id','sequence','index','phase','features','rays','causal','gt','mask'},'Cache contains undeclared inputs; only one current frame is allowed')
    require(obj['signature']==ctx.signature and obj['case_id']==case['id'] and obj['index']==case['index'] and obj['phase']==case['phase'],'Cache sample changed')
    require(tensor_digest(obj['mask'])==meta['tensor_sha256']['mask'],'Fixed GT cache mask changed');return obj

def gradient_audit(ctx):
    t=ctx.rt.torch;records=[];car=FrozenCarrier(ctx,ctx.old_sequences[0]);module=new_micro(ctx,'N14_A1')
    try:
        for s in ctx.old_sequences:
            for h in range(1,8):
                c=next(c for c in ctx.plan['train_cases'] if c['sequence']==s and c['phase']==h);obj=load_cache(ctx,c);module.zero_grad(set_to_none=True)
                result=micro_from_cache(ctx,car,module,obj);(result['current']*1024.).backward();grads=[p.grad for p in module.parameters() if p.grad is not None]
                require(grads and all(bool(t.isfinite(g).all().item()) for g in grads),'Current GT gradient missing/nonfinite')
                magnitude=sum(float(g.float().abs().sum().item()) for g in grads)/1024.;require(magnitude>0,'Current GT cannot train micro-readout')
                require(all(not p.requires_grad and p.grad is None for p in car.base.model.parameters()),'Frozen C*/backbone/decoder received parameter gradients')
                require(all(not x.requires_grad for x in [*obj['features'],obj['rays'],obj['causal'],obj['gt']]),'Cache inputs carry a state graph')
                records.append(dict(sequence=s,phase=h,case_id=c['id'],current_GT_gradient_l1=magnitude,frozen_parameter_gradients=False,current_frame_only=True));del result,obj
        write(ctx.out/'gradient_audit.json',dict(signature=ctx.signature,passed=True,actual_model_CUDA=True,records=records,future_GT_exposures=0))
        print('[GRADIENT PASS] current GT reaches micro head only; 21 sequence/phase canaries; future GT=0',flush=True)
    finally:car.close();del module;t.cuda.empty_cache()

def train_one(ctx,arm,resume):
    t=ctx.rt.torch;p=ctx.policy;module=new_micro(ctx,arm);car=FrozenCarrier(ctx,ctx.old_sequences[0]);params=list(module.parameters())
    d=ctx.out/'training'/arm;d.mkdir(parents=True,exist_ok=True);recovery=d/'recovery.pt';final=d/'final.pt';summary=d/'summary.json'
    initial=ctx.rt.exp1.state_sha256(cpu_state(module));curve=[];step0=0
    if final.exists() and summary.exists():
        m=read(summary);obj=t.load(final,map_location='cpu',weights_only=False)
        require(m['signature']==obj['signature']==ctx.signature and obj['arm']==arm and obj['step']==ctx.plan['updates'] and obj['tensor_sha256']==ctx.rt.exp1.state_sha256(obj['state']) and m['final_file_sha256']==sha(final),'Completed checkpoint changed')
        print('[REUSE]',arm,'training complete',flush=True);car.close();return m
    opt=t.optim.AdamW(params,lr=p['learning_rate'],weight_decay=p['weight_decay']);scaler=t.amp.GradScaler('cuda',init_scale=1024.,growth_interval=2000)
    if recovery.exists():
        require(resume,'Recovery exists; use --resume-latest');x=t.load(recovery,map_location='cpu',weights_only=False)
        require(x['signature']==ctx.signature and x['arm']==arm and x['initial_sha256']==initial and x['sample_order_sha256']==ctx.plan['sample_order_sha256'],'Recovery source/order differs')
        curve=x['curve'];check_recovery(curve,ctx.plan['sample_order']);require(x['step']==len(curve),'Recovery cursor invalid');step0=x['step']
        module.load_state_dict(x['state'],strict=True);opt.load_state_dict(x['optimizer']);scaler.load_state_dict(x['scaler']);restore_rng(ctx,x['rng']);print('[RESUME]',arm,step0,flush=True)
    cases={c['id']:c for c in ctx.plan['train_cases']};journal=d/'physical_updates.jsonl';started=time.perf_counter()
    def commit(step):
        state=cpu_state(module);save_torch(t,recovery,dict(signature=ctx.signature,arm=arm,step=step,state=state,optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=rng(ctx),curve=curve,initial_sha256=initial,sample_order_sha256=ctx.plan['sample_order_sha256']))
        csvwrite(d/'curve.csv',curve)
        if step==ctx.plan['updates']:save_torch(t,final,dict(signature=ctx.signature,arm=arm,step=step,state=state,tensor_sha256=ctx.rt.exp1.state_sha256(state),Cstar_tensor_sha256=ctx.plan['Cstar_tensor_sha256'],parameter_count=9244))
    try:
        for step in range(step0+1,ctx.plan['updates']+1):
            case=cases[ctx.plan['sample_order'][step-1]];t.cuda.synchronize();start=time.perf_counter();obj=load_cache(ctx,case);car.configure(case['sequence']);prep=time.perf_counter()-start
            ev0=t.cuda.Event(enable_timing=True);ev1=t.cuda.Event(enable_timing=True);ev0.record();opt.zero_grad(set_to_none=True)
            result=micro_from_cache(ctx,car,module,obj);require(bool(t.isfinite(result['objective']).item()),'Nonfinite objective');scaler.scale(result['objective']).backward();scaler.unscale_(opt)
            grads=[v.grad for v in params if v.grad is not None];require(grads and all(bool(t.isfinite(v).all().item()) for v in grads),'Invalid micro gradient');require(all(not v.requires_grad and v.grad is None for v in car.base.model.parameters()),'Frozen model received gradients')
            norm=float(t.nn.utils.clip_grad_norm_(params,p['clip_grad_norm']).item());require(math.isfinite(norm) and norm>0,'Invalid gradient norm')
            scale=scaler.get_scale();scaler.step(opt);scaler.update();require(scaler.get_scale()>=scale,'AMP skipped update; recovery not advanced')
            require(all(bool(t.isfinite(v).all().item()) for v in params),'Nonfinite parameters');ev1.record();t.cuda.synchronize()
            row=dict(step=step,arm=arm,case_id=case['id'],sequence=case['sequence'],index=case['index'],phase=case['phase'],gradient_norm=norm,current_GT=1,future_GT=0,cuda_sec=ev0.elapsed_time(ev1)/1000.,step_wall_sec=time.perf_counter()-start,preparation_wall_sec=prep,**{k:float(v.detach().item()) for k,v in result.items()})
            row['timing_valid']=math.isfinite(row['cuda_sec']) and 0<row['cuda_sec']<=row['step_wall_sec']*1.1+.05
            if journal.exists() and journal.stat().st_size:
                with journal.open('rb') as f:f.seek(-1,2);needs_newline=f.read(1)!=b'\n'
                if needs_newline:
                    with journal.open('ab') as f:f.write(b'\n')
            with journal.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            require(row['timing_valid'],'CUDA interval inconsistent with wall; physical attempt kept, recovery not advanced');curve.append(row)
            if step%p['checkpoint_interval']==0 or step==ctx.plan['updates']:commit(step)
            if step==1 or step%10==0 or step==ctx.plan['updates']:print(f'[TRAIN {arm}] {step}/{ctx.plan["updates"]} current={row["current"]:.6f} phase={case["phase"]} wall={row["step_wall_sec"]:.2f}s',flush=True)
            del result,obj
        physical=[];malformed=0
        for line in journal.read_text().splitlines():
            if not line.strip():continue
            try:physical.append(json.loads(line))
            except json.JSONDecodeError:malformed+=1
        result=dict(signature=ctx.signature,status='TRAINING_COMPLETE',arm=arm,updates=len(curve),parameter_count=9244,initial_sha256=initial,final_file_sha256=sha(final),sample_order_sha256=ctx.plan['sample_order_sha256'],current_GT_exposures=len(curve),future_GT_exposures=0,
            committed_cuda_sec=sum(x['cuda_sec'] for x in curve),committed_wall_sec=sum(x['step_wall_sec'] for x in curve),physical_updates=len(physical),physical_cuda_sec=sum(x['cuda_sec'] for x in physical),malformed_journal_lines=malformed,current_invocation_sec=time.perf_counter()-started,
            compute_claim='same parameters, initialization, samples, masks, order, optimizer and updates; actual compute measured, not asserted equal')
        write(summary,result);return result
    finally:car.close();del module,opt,scaler;t.cuda.empty_cache()

def train_all(ctx,resume=False):
    real_smoke(ctx);build_cache(ctx);gradient_audit(ctx);summaries={}
    for arm in ctx.policy['arms']:summaries[arm]=train_one(ctx,arm,resume)
    require(summaries['N14_A0']['initial_sha256']==summaries['N14_A1']['initial_sha256'],'A0/A1 initialization differs');write(ctx.out/'training_summary.json',summaries)
